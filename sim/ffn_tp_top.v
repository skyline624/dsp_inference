`timescale 1ns/1ps
// ffn_tp_top - 2 unmodified brick nodes driven by the autonomous TP-FFN sequencer.
module ffn_tp_top #(
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
    ffn_tp_seq #(W, D) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .base(23'd0),
        .n0_rx(n0rx), .n1_rx(n1rx), .n0_tx(n0tx), .n1_tx(n1tx),
        .result(result), .result_sh(result_sh), .done(done));
endmodule
