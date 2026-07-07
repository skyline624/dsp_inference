`timescale 1ns/1ps
// =============================================================================
// tp_silu_node - autonomous chain integrating a REAL elementwise operator
//
// Demonstrates the missing FFN capability: chaining the real silu_op (with its
// BSRAM read/write interface) after a real-DSP matmul, inside one autonomous FSM
// with correct int8+pow2 shift bookkeeping:
//
//   matmul y=W@x (mac18 + fq_ref requant -> int8, shift = sx+sw+sh_used)
//        -> silu_op (real LUT operator) -> int8, shift_out
//
// Same operator-integration pattern works for rmsnorm_op; chaining several of
// these + the all-gather/all-reduce already built = the full FFN.
// =============================================================================
module tp_silu_node #(
    parameter W  = 8,
    parameter K  = 64,
    parameter NR = 32
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [W*K-1:0]   x,
    input  wire signed [7:0] sx,
    input  wire signed [7:0] sw,
    output reg  [W*NR-1:0]  out,
    output reg  signed [7:0] out_shift,
    output reg              done
);
    localparam LAT = 4;

    reg signed [7:0] wmem [0:NR*K-1];

    // real DSP
    reg  signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));
    reg signed [31:0] yi32 [0:NR-1];

    // buffers around the real silu_op
    reg  signed [7:0] sbuf [0:NR-1];        // silu input  (matmul result, int8)
    reg  signed [7:0] obuf [0:NR-1];        // silu output
    reg  signed [7:0] silu_x_rdata;
    wire [9:0]        silu_x_raddr;
    wire [9:0]        silu_out_waddr;
    wire signed [7:0] silu_out_wdata;
    wire              silu_out_we;
    wire              silu_done;
    reg               silu_start;
    reg  signed [7:0] silu_shift_x;
    wire signed [7:0] silu_shift_out;

    always @(posedge clk) silu_x_rdata <= sbuf[silu_x_raddr[5:0]];   // 1-cycle read
    silu_op #(.D(NR)) u_silu (
        .clk(clk), .rst(~rst_n), .start(silu_start), .done(silu_done),
        .shift_x(silu_shift_x), .shift_out(silu_shift_out),
        .x_raddr(silu_x_raddr), .x_rdata(silu_x_rdata),
        .out_waddr(silu_out_waddr), .out_wdata(silu_out_wdata), .out_we(silu_out_we),
        .dbg_lut_idx(), .dbg_silu_int());
    always @(posedge clk) if (silu_out_we) obuf[silu_out_waddr[5:0]] <= silu_out_wdata;

    localparam IDLE=0, MAC=1, FLUSH=2, MAXF=3, REQ=4, SSTART=5, SWAIT=6, EMIT=7, FIN=8;
    reg [3:0] st;
    reg [9:0] r, k, fcnt;
    reg signed [31:0] max_abs;
    reg [5:0] sh;
    integer i;
    reg signed [31:0] av, rounded, shifted;

    function [5:0] msb_idx;
        input [31:0] v; integer b;
        begin msb_idx=0; for (b=0;b<32;b=b+1) if (v[b]) msb_idx=b[5:0]; end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; silu_start<=0; mac_a<=0; mac_b<=0; mac_load<=0;
            r<=0; k<=0; fcnt<=0;
        end else begin
            silu_start<=0; done<=0;
            case (st)
                IDLE: if (start) begin r<=0; k<=0; st<=MAC; end

                MAC: begin
                    mac_a    <= {{10{wmem[r*K+k][7]}}, wmem[r*K+k]};
                    mac_b    <= {{10{x[k*W+(W-1)]}}, x[k*W +: W]};
                    mac_load <= (k==0);
                    if (k==K-1) begin k<=0; fcnt<=0; st<=FLUSH; end else k<=k+1;
                end
                FLUSH: begin
                    mac_a<=0; mac_b<=0; mac_load<=0;
                    if (fcnt==LAT) begin
                        yi32[r] <= mac_res[31:0];
                        if (r==NR-1) st<=MAXF; else begin r<=r+1; st<=MAC; end
                    end else fcnt<=fcnt+1;
                end
                MAXF: begin
                    max_abs = 0;
                    for (i=0;i<NR;i=i+1) begin
                        av = yi32[i][31] ? -yi32[i] : yi32[i];
                        if (av > max_abs) max_abs = av;
                    end
                    sh <= (max_abs==0) ? 6'd0 :
                          (msb_idx(max_abs) > 6) ? (msb_idx(max_abs)-6) : 6'd0;
                    st <= REQ;
                end
                REQ: begin                                   // requant -> sbuf (h1 int8)
                    for (i=0;i<NR;i=i+1) begin
                        rounded = yi32[i] + ((sh>0) ? (32'sd1 <<< (sh-1)) : 32'sd0);
                        shifted = rounded >>> sh;
                        sbuf[i] <= (shifted > 32'sd127)  ? 8'sd127  :
                                   (shifted < -32'sd128) ? -8'sd128 : shifted[7:0];
                    end
                    silu_shift_x <= sx + sw + $signed({2'b0, sh});   // matmul output shift
                    st <= SSTART;
                end
                SSTART: begin silu_start <= 1; st <= SWAIT; end
                SWAIT:  if (silu_done) begin out_shift <= silu_shift_out; st <= EMIT; end
                EMIT: begin
                    for (i=0;i<NR;i=i+1) out[i*W +: W] <= obuf[i];
                    done <= 1; st <= FIN;
                end
                FIN: st <= IDLE;
            endcase
        end
    end
endmodule
