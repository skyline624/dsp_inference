`timescale 1ns/1ps
// attn_tp_top - NN brick nodes driven by the NN-parameterized TP attention seq
// (UART internal link). NN in {1,2,4} : NN=1 mono-card, NN>=2 head-parallel cluster.
module attn_tp_top #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx, sw_rms, swq, swk, swv, swo,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire           done
);
    wire [NN-1:0] n_rx, n_tx;
    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nodes
        node_top u_n (.clk(clk), .uart_rx(n_rx[g]), .uart_tx(n_tx[g]), .led());
    end endgenerate

    attn_tp_seq #(.W(W), .D(D), .NN(NN)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .base(23'd0),
        .n_rx(n_rx), .n_tx(n_tx),
        .result(result), .result_sh(result_sh), .done(done));
endmodule
