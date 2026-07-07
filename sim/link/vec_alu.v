`timescale 1ns/1ps
// =============================================================================
// vec_alu - the sequencer's small ALU for the FFN "glue" ops (multiply, residual)
//
// Two elementwise vector ops on int8+pow2-shift operands, with integer requantize
// (round-half-up, consistent with the brick datapath):
//   op=MUL : out = a * b           (silu(h1) * h3)   -> uses the DSP (mac18)
//   op=ADD : out = a + b           (residual x + W·)  -> shift-align + adder
//
// The multiply is done on the real Gowin DSP (MULTALU18X18 via mac18, load mode
// = charge a*b, no accumulate) -- that's exactly what a DSP is for.
// =============================================================================
module vec_alu #(
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

    // DSP for the multiply
    reg  signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));

    reg signed [31:0] acc [0:D-1];      // products (MUL) or aligned sums (ADD)
    reg signed [7:0]  base_sh;

    function signed [7:0] smin2;
        input signed [7:0] p, q;
        smin2 = (p < q) ? p : q;
    endfunction
    function [5:0] msb_idx;
        input [31:0] v; integer bb;
        begin msb_idx=0; for (bb=0;bb<32;bb=bb+1) if (v[bb]) msb_idx=bb[5:0]; end
    endfunction

    localparam IDLE=0, MUL_P=1, MUL_W=2, ADDC=3, MAXF=4, REQ=5, FIN=6;
    reg [2:0] st;
    reg [9:0] i, fcnt;
    reg signed [31:0] max_abs, av, rounded, shifted;
    reg [5:0] add_sh;
    integer j;
    reg signed [7:0] da, db_;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=IDLE; done<=0; mac_load<=0; i<=0; fcnt<=0; end
        else begin
            done<=0;
            case (st)
                IDLE: if (start) begin
                    if (op==1'b0) begin base_sh <= sa+sb; i<=0; st<=MUL_P; end
                    else st<=ADDC;
                end

                // ---- MUL via DSP : acc[i] = a[i]*b[i] ----
                MUL_P: begin
                    mac_a <= {{10{a[i*W+(W-1)]}}, a[i*W +: W]};
                    mac_b <= {{10{b[i*W+(W-1)]}}, b[i*W +: W]};
                    mac_load <= 1'b1;       // charge a*b (no accumulate)
                    fcnt <= 0; st <= MUL_W;
                end
                MUL_W: begin
                    mac_load <= 1'b0;
                    mac_a <= 0; mac_b <= 0;     // accumulate 0 while draining (no re-mul)
                    if (fcnt==LAT) begin
                        acc[i] <= mac_res[31:0];
                        if (i==D-1) st<=MAXF; else begin i<=i+1; st<=MUL_P; end
                    end else fcnt<=fcnt+1;
                end

                // ---- ADD : shift-align then add ----
                ADDC: begin
                    base_sh = smin2(sa, sb);
                    da  = sa - base_sh;
                    db_ = sb - base_sh;
                    for (j=0;j<D;j=j+1)
                        acc[j] <= ($signed(a[j*W +: W]) <<< da)
                                + ($signed(b[j*W +: W]) <<< db_);
                    st <= MAXF;
                end

                // ---- requantize (round-half-up, like the bricks) ----
                MAXF: begin
                    max_abs = 0;
                    for (j=0;j<D;j=j+1) begin
                        av = acc[j][31] ? -acc[j] : acc[j];
                        if (av > max_abs) max_abs = av;
                    end
                    add_sh <= (max_abs==0) ? 6'd0 :
                              (msb_idx(max_abs) > 6) ? (msb_idx(max_abs)-6) : 6'd0;
                    st <= REQ;
                end
                REQ: begin
                    for (j=0;j<D;j=j+1) begin
                        rounded = acc[j] + ((add_sh>0) ? (32'sd1 <<< (add_sh-1)) : 32'sd0);
                        shifted = rounded >>> add_sh;
                        out[j*W +: W] <= (shifted > 32'sd127)  ? 8'sd127  :
                                         (shifted < -32'sd128) ? -8'sd128 : shifted[7:0];
                    end
                    out_sh <= base_sh + $signed({2'b0, add_sh});
                    done <= 1; st <= FIN;
                end
                FIN: st<=IDLE;
            endcase
        end
    end
endmodule
