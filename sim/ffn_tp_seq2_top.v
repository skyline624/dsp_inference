`timescale 1ns/1ps
// ffn_tp_seq2_top - sim harness : NN brick nodes driven by the NN-parameterized
// LUT-lean FFN-TP sequencer (ffn_tp_seq2), UART internal link. NN=2 is the
// cluster parity case (hidden=128, 64/node); NN=1 is the mono-card degenerate
// case (hidden=64, single node, reduce skipped).
module ffn_tp_seq2_top #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx, sw_rms, sw1, sw3, sw2,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire           done
);
    wire [NN-1:0] n_rx, n_tx;
    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nodes
        node_top u_n (.clk(clk), .uart_rx(n_rx[g]), .uart_tx(n_tx[g]), .led());
    end endgenerate

    ffn_tp_seq2 #(.W(W), .D(D), .NN(NN), .DIV(27), .BOOT(40000),
                  .A_RMS(23'h100000), .A_W1(23'h101000),
                  .A_W3(23'h102000), .A_W2(23'h103000)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .base(23'd0),
        .n_rx(n_rx), .n_tx(n_tx),
        .result(result), .result_sh(result_sh), .done(done));
endmodule
