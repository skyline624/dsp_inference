`timescale 1ns/1ps
// =============================================================================
// model_tp_top - autonomous MULTI-LAYER tensor-parallel forward pass (NN nodes)
//
// The layer controller loops the full TP layer (attn_tp_seq -> ffn_tp_seq2) over
// NL layers, zero PC:
//   for l in 0..NL-1:
//     base = l*STRIDE   (layer l weights live at that SDRAM offset)
//     x <- ffn( attn(x, layer l) , layer l )      // attention + FFN, residuals
// Per-layer weight shifts come in as packed buses (one byte per layer). All NL
// layers' weights sit in the NN SDRAMs at distinct offsets. NN=1 mono-card.
// =============================================================================
module model_tp_top #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2,
    parameter NL = 2,
    parameter [22:0] STRIDE = 23'h020000
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx,
    input  wire [NL*8-1:0] sw_rms_a, swq, swk, swv, swo,   // attention (per layer)
    input  wire [NL*8-1:0] sw_rms_f, sw1, sw3, sw2,        // FFN (per layer)
    output reg  [W*D-1:0] result,
    output reg  signed [7:0] result_sh,
    output reg            done
);
    wire [NN-1:0] n_tx;
    wire [NN-1:0] a_n_rx, f_n_rx;
    reg  phase;
    wire [NN-1:0] n_rx = phase ? f_n_rx : a_n_rx;

    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nodes
        node_top u_n (.clk(clk), .uart_rx(n_rx[g]), .uart_tx(n_tx[g]), .led());
    end endgenerate

    // current activation, layer index, per-layer base offset
    reg [3:0] l;
    reg [W*D-1:0] xcur; reg signed [7:0] sxcur;
    wire [22:0] base = l * STRIDE;

    // per-layer weight shifts (select current layer)
    wire signed [7:0] c_ra = sw_rms_a[l*8 +: 8], c_q = swq[l*8 +: 8], c_k = swk[l*8 +: 8],
                      c_v = swv[l*8 +: 8], c_o = swo[l*8 +: 8];
    wire signed [7:0] c_rf = sw_rms_f[l*8 +: 8], c_1 = sw1[l*8 +: 8],
                      c_3 = sw3[l*8 +: 8], c_2 = sw2[l*8 +: 8];

    reg            a_start; wire a_done;
    wire [W*D-1:0] a_res; wire signed [7:0] a_sh;
    attn_tp_seq #(.W(W), .D(D), .NN(NN)) u_attn (
        .clk(clk), .rst_n(rst_n), .start(a_start), .x_in(xcur), .sx_in(sxcur),
        .sw_rms(c_ra), .swq(c_q), .swk(c_k), .swv(c_v), .swo(c_o), .base(base),
        .n_rx(a_n_rx), .n_tx(n_tx),
        .result(a_res), .result_sh(a_sh), .done(a_done));

    reg            f_start; wire f_done;
    reg  [W*D-1:0] x1; reg signed [7:0] sx1;
    wire [W*D-1:0] f_res; wire signed [7:0] f_sh;
    ffn_tp_seq2 #(.W(W), .D(D), .NN(NN), .BOOT(16)) u_ffn (
        .clk(clk), .rst_n(rst_n), .start(f_start), .x_in(x1), .sx_in(sx1),
        .sw_rms(c_rf), .sw1(c_1), .sw3(c_3), .sw2(c_2), .base(base + 23'h10000),
        .n_rx(f_n_rx), .n_tx(n_tx),
        .result(f_res), .result_sh(f_sh), .done(f_done));

    localparam IDLE=0, ATTN=1, HANDOFF=2, FFN=3, NEXT=4, FIN=5;
    reg [2:0] cst;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin cst<=IDLE; a_start<=0; f_start<=0; phase<=0; done<=0; l<=0; end
        else begin
            a_start<=0; f_start<=0; done<=0;
            case (cst)
                IDLE: if (start) begin xcur<=x_in; sxcur<=sx; l<=0; phase<=0; a_start<=1; cst<=ATTN; end
                ATTN: if (a_done) begin x1<=a_res; sx1<=a_sh; cst<=HANDOFF; end
                HANDOFF: begin phase<=1; f_start<=1; cst<=FFN; end
                FFN: if (f_done) cst<=NEXT;
                NEXT: begin
                    if (l == NL-1) begin result<=f_res; result_sh<=f_sh; done<=1; cst<=FIN; end
                    else begin xcur<=f_res; sxcur<=f_sh; l<=l+1; phase<=0; a_start<=1; cst<=ATTN; end
                end
                FIN: cst<=IDLE;
            endcase
        end
    end
endmodule
