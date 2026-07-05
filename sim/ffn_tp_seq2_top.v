`timescale 1ns/1ps
// ffn_tp_seq2_top - Phase-0 sim harness : 2 unmodified brick nodes driven by the
// LUT-lean autonomous TP-FFN sequencer (ffn_tp_seq2), UART internal link kept as-is.
// Same structure as ffn_tp_top.v but instantiates ffn_tp_seq2 (register-file /
// vec_alu2 variant) so test_ffn_tp_seq2.py can prove it matches the float FFN.
module ffn_tp_seq2_top #(
    parameter W = 8,
    parameter D = 64
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx, sw_rms, sw1, sw3, sw2,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire           done
);
    wire n0rx, n0tx, n1rx, n1tx;
    node_top u_n0 (.clk(clk), .uart_rx(n0rx), .uart_tx(n0tx), .led());
    node_top u_n1 (.clk(clk), .uart_rx(n1rx), .uart_tx(n1tx), .led());
    // BOOT lowered vs the 40000 default : node_top has no SD boot wait here, so the
    // sequencer can start talking to the nodes soon after reset (matches ffn_tp_top).
    ffn_tp_seq2 #(.W(W), .D(D), .DIV(27), .BOOT(40000),
                  .A_RMS(23'h100000), .A_W1(23'h101000),
                  .A_W3(23'h102000), .A_W2(23'h103000)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .base(23'd0),
        .n0_rx(n0rx), .n1_rx(n1rx), .n0_tx(n0tx), .n1_tx(n1tx),
        .result(result), .result_sh(result_sh), .done(done));
endmodule
