`timescale 1ns/1ps
// ffn_top - unmodified brick node driven by the autonomous FFN sequencer (no PC).
module ffn_top #(
    parameter W = 8,
    parameter D = 64
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx,
    input  wire signed [7:0] sw_rms,
    input  wire signed [7:0] sw1, sw3, sw2,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire           done
);
    wire n_rx, n_tx;
    node_top u_node (.clk(clk), .uart_rx(n_rx), .uart_tx(n_tx), .led());
    ffn_seq #(W, D) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .node_rx(n_rx), .node_tx(n_tx),
        .result(result), .result_sh(result_sh), .done(done));
endmodule
