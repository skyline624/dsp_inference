`timescale 1ns/1ps
// =============================================================================
// vec_alu2 - SERIALIZED elementwise ALU for the sequencer's FFN glue ops.
//
// Same ops as vec_alu (MUL = a*b on DSP, ADD = shift-align + add, then requantize)
// but every elementwise pass runs ONE element per cycle instead of 64 in parallel.
// The 64-wide parallel adder/max/scaler that cost ~5000 LUT in v1 becomes a tiny
// serial loop. acc lives in a 64x32 LUT-RAM (cheap). Cost target: a few hundred LUT.
// The 64-cycle passes add ~200 cycles/FFN = ~0.7% of the matmul time -> negligible.
// =============================================================================
module vec_alu2 #(
    parameter W = 8,
    parameter D = 64
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire            op,          // 0 = MUL, 1 = ADD
    input  wire [W*D-1:0]  a,
    input  wire signed [7:0] sa,
    input  wire [W*D-1:0]  b,
    input  wire signed [7:0] sb,
    output reg  [W*D-1:0]  out,
    output reg  signed [7:0] out_sh,
    output reg             done
);
    localparam LAT = 4;
    reg signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));

    reg signed [31:0] acc [0:D-1];        // products / aligned sums (LUT-RAM)
    reg signed [31:0] max_abs, av, cur, rounded, shifted;
    reg signed [7:0]  base_sh, da, db_;
    reg [5:0] add_sh;
    reg [6:0] i;                          // 0..D element index
    reg [2:0] fcnt;                       // DSP pipeline drain counter
    reg [2:0] ph;                         // sub-phase: 0=MUL/ADD, 1=REQ, 2=FIN

    function [5:0] msb_idx;
        input [31:0] v; integer bb;
        begin msb_idx=0; for (bb=0;bb<32;bb=bb+1) if (v[bb]) msb_idx=bb[5:0]; end
    endfunction

    localparam IDLE=0, MUL_P=1, MUL_W=2, ADDC=3, MAXF=4, REQ=5, FIN=6;
    reg [2:0] st;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=IDLE; done<=0; mac_load<=0; i<=0; fcnt<=0; max_abs<=0; end
        else begin
            done<=0; mac_load<=0;
            case (st)
                IDLE: if (start) begin
                    base_sh <= (op==1'b0) ? (sa+sb) : ((sa<sb)?sa:sb);
                    da  <= (op==1'b0) ? 8'sd0 : (sa - ((sa<sb)?sa:sb));
                    db_ <= (op==1'b0) ? 8'sd0 : (sb - ((sa<sb)?sa:sb));
                    i<=0; max_abs<=0; fcnt<=0;
                    st <= (op==1'b0) ? MUL_P : ADDC;
                end

                // ---- MUL via DSP : acc[i] = a[i]*b[i], track max_abs serially ----
                MUL_P: begin
                    mac_a <= {{10{a[i*W+(W-1)]}}, a[i*W +: W]};
                    mac_b <= {{10{b[i*W+(W-1)]}}, b[i*W +: W]};
                    mac_load <= 1'b1; fcnt<=0; st<=MUL_W;
                end
                MUL_W: begin
                    mac_a <= 0; mac_b <= 0;   // drain a*b only : zero operands so the
                                              // accumulate cycles don't re-add a*b (2x bug)
                    if (fcnt==LAT) begin
                        cur <= mac_res[31:0];
                        acc[i] <= mac_res[31:0];
                        av = mac_res[31] ? -mac_res[31:0] : mac_res[31:0];
                        if (av > max_abs) max_abs <= av;
                        if (i==D-1) st<=REQ; else begin i<=i+1; st<=MUL_P; end
                    end else fcnt<=fcnt+1;
                end

                // ---- ADD : shift-align + add, one element/cycle, track max_abs ----
                ADDC: begin
                    cur = ($signed(a[i*W +: W]) <<< da) + ($signed(b[i*W +: W]) <<< db_);
                    acc[i] <= cur;
                    av = cur[31] ? -cur : cur;
                    if (av > max_abs) max_abs <= av;
                    if (i==D-1) st<=REQ; else i<=i+1;
                end

                // ---- requantize pass : scale acc[i] -> out[i], one/cycle ----
                REQ: begin
                    add_sh <= (max_abs==0) ? 6'd0 :
                              (msb_idx(max_abs) > 6) ? (msb_idx(max_abs)-6) : 6'd0;
                    i<=0;                 // restart element index for the requantize pass
                    st<=FIN;
                end
                FIN: begin
                    rounded = acc[i] + ((add_sh>0) ? (32'sd1 <<< (add_sh-1)) : 32'sd0);
                    shifted = rounded >>> add_sh;
                    out[i*W +: W] <= (shifted > 32'sd127)  ? 8'sd127  :
                                     (shifted < -32'sd128) ? -8'sd128 : shifted[7:0];
                    if (i==D-1) begin out_sh<=base_sh+$signed({2'b0,add_sh}); done<=1; st<=IDLE; end
                    else i<=i+1;
                end
            endcase
        end
    end
endmodule