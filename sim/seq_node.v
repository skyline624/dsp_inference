`timescale 1ns/1ps
// =============================================================================
// seq_node - a node driven by an on-chip sequencer instead of a PC
//
// node_top (the real FPGA node = top.v + SDRAM, UNMODIFIED) + the seq host,
// connected over the node's own UART command interface (on the same chip). The
// sequencer issues commands autonomously -> zero-PC, reusing the brick datapath.
// =============================================================================
module seq_node #(
    parameter W = 8,
    parameter D = 64,
    parameter N_OPS = 1
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx,
    output wire [W*D-1:0] result,
    output wire           done
);
    wire n_rx, n_tx;
    node_top u_node (.clk(clk), .uart_rx(n_rx), .uart_tx(n_tx), .led());
    seq #(.W(W), .D(D), .N_OPS(N_OPS)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx(sx),
        .node_rx(n_rx), .node_tx(n_tx), .result(result), .done(done));
endmodule
