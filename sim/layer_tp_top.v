`timescale 1ns/1ps
// =============================================================================
// layer_tp_top - COMPLETE autonomous tensor-parallel transformer layer (NN nodes)
//
// Reuses the two NN-parameterized v2 sequencers as-is + a tiny layer controller:
//   1. run attn_tp_seq  -> x1 = x + attention_out
//   2. run ffn_tp_seq2 with x1 -> x2 = x1 + ffn_out   (= the layer output)
// The two sequencers drive the SAME NN nodes in turn (muxed by `phase`).
// Attention weights at 0x10xxxx, FFN weights at 0x11xxxx. Zero PC ; each node
// holds only its slice of every weight. NN=1 mono-card, NN>=2 cluster.
// =============================================================================
module layer_tp_top #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx,
    input  wire signed [7:0] sw_rms_a, swq, swk, swv, swo,   // attention
    input  wire signed [7:0] sw_rms_f, sw1, sw3, sw2,        // FFN
    output reg  [W*D-1:0] result,
    output reg  signed [7:0] result_sh,
    output reg            done
);
    wire [NN-1:0] n_tx;
    wire [NN-1:0] a_n_rx, f_n_rx;
    reg  phase;                          // 0 = attention, 1 = FFN
    wire [NN-1:0] n_rx = phase ? f_n_rx : a_n_rx;

    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nodes
        node_top u_n (.clk(clk), .uart_rx(n_rx[g]), .uart_tx(n_tx[g]), .led());
    end endgenerate

    // ---- attention block (weights at 0x10xxxx) ----
    reg               a_start; wire a_done;
    wire [W*D-1:0]    a_res; wire signed [7:0] a_sh;
    attn_tp_seq #(.W(W), .D(D), .NN(NN)) u_attn (
        .clk(clk), .rst_n(rst_n), .start(a_start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms_a), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .base(23'd0),
        .n_rx(a_n_rx), .n_tx(n_tx),
        .result(a_res), .result_sh(a_sh), .done(a_done));

    // ---- FFN block (weights at 0x11xxxx, boot wait already done) ----
    reg               f_start; wire f_done;
    reg  [W*D-1:0]    x1; reg signed [7:0] sx1;
    wire [W*D-1:0]    f_res; wire signed [7:0] f_sh;
    ffn_tp_seq2 #(.W(W), .D(D), .NN(NN), .BOOT(16),
                  .A_RMS(23'h110000), .A_W1(23'h111000), .A_W3(23'h112000), .A_W2(23'h113000)) u_ffn (
        .clk(clk), .rst_n(rst_n), .start(f_start), .x_in(x1), .sx_in(sx1),
        .sw_rms(sw_rms_f), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .base(23'd0),
        .n_rx(f_n_rx), .n_tx(n_tx),
        .result(f_res), .result_sh(f_sh), .done(f_done));

    localparam IDLE=0, ATTN=1, HANDOFF=2, FFN=3, FIN=4;
    reg [2:0] cst;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin cst<=IDLE; a_start<=0; f_start<=0; phase<=0; done<=0; end
        else begin
            a_start<=0; f_start<=0; done<=0;
            case (cst)
                IDLE: if (start) begin phase<=0; a_start<=1; cst<=ATTN; end
                ATTN: if (a_done) begin x1<=a_res; sx1<=a_sh; cst<=HANDOFF; end
                HANDOFF: begin phase<=1; f_start<=1; cst<=FFN; end
                FFN: if (f_done) begin result<=f_res; result_sh<=f_sh; done<=1; cst<=FIN; end
                FIN: cst<=IDLE;
            endcase
        end
    end
endmodule
