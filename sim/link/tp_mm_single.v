`timescale 1ns/1ps
// =============================================================================
// tp_mm_single - one autonomous matmul node + a testbench peer, independent clks
//
// The testbench (node A) sends x[K] to the node and receives y[N]. The node runs
// on its own clock and does the whole matmul slice autonomously (no commands).
// =============================================================================
module tp_mm_single #(
    parameter W = 8,
    parameter K = 64,
    parameter N = 32
) (
    input  wire           clk_a, rst_a_n,
    input  wire           clk_b, rst_b_n,
    input  wire           start_a,
    input  wire [W*K-1:0] vec_a_tx,   // x
    output wire           busy_a,
    output wire [W*N-1:0] vec_a_rx,   // y
    output wire           done_a
);
    wire [W-1:0] ab_d, ab_rdd; wire ab_send, ab_full, ab_rd, ab_empty;
    wire [W-1:0] ba_d, ba_rdd; wire ba_send, ba_full, ba_rd, ba_empty;

    // A -> node : x
    link_send #(W, K) u_send_a (
        .clk(clk_a), .rst_n(rst_a_n), .start(start_a), .vec(vec_a_tx),
        .busy(busy_a), .tx_data(ab_d), .tx_send(ab_send), .tx_full(ab_full));
    ss_link #(W, 4) u_link_ab (
        .clk_tx(clk_a), .rst_tx_n(rst_a_n), .tx_data(ab_d), .tx_send(ab_send), .tx_full(ab_full),
        .clk_rx(clk_b), .rst_rx_n(rst_b_n), .rx_rd(ab_rd), .rx_data(ab_rdd), .rx_empty(ab_empty));

    // node -> A : y
    ss_link #(W, 4) u_link_ba (
        .clk_tx(clk_b), .rst_tx_n(rst_b_n), .tx_data(ba_d), .tx_send(ba_send), .tx_full(ba_full),
        .clk_rx(clk_a), .rst_rx_n(rst_a_n), .rx_rd(ba_rd), .rx_data(ba_rdd), .rx_empty(ba_empty));
    link_recv #(W, N) u_recv_a (
        .clk(clk_a), .rst_n(rst_a_n), .rx_rd(ba_rd), .rx_data(ba_rdd), .rx_empty(ba_empty),
        .vec(vec_a_rx), .done(done_a));

    // autonomous matmul node (clk_b)
    tp_mm_node #(W, K, N) u_node (
        .clk(clk_b), .rst_n(rst_b_n),
        .rx_rd(ab_rd), .rx_data(ab_rdd), .rx_empty(ab_empty),
        .tx_data(ba_d), .tx_send(ba_send), .tx_full(ba_full),
        .done());
endmodule
