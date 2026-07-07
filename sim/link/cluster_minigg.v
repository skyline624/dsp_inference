`timescale 1ns/1ps
// =============================================================================
// cluster_minigg - node A (peer) + node B (autonomous mini_gg), independent clocks
//
// Node A (driven by the testbench) sends a vector to B and receives B's reply.
// Node B is a self-contained mini_gg: it autonomously RECVs, runs its 2-stage
// op chain, and SENDs the result back -- with NO per-op command from anyone.
// This is the "zero-PC autonomy" goal, reborn per-node and small enough to route.
// =============================================================================
module cluster_minigg #(
    parameter W = 8,
    parameter L = 64
) (
    input  wire           clk_a, rst_a_n,
    input  wire           clk_b, rst_b_n,
    // node A (testbench peer)
    input  wire           start_a,
    input  wire [W*L-1:0] vec_a_tx,
    output wire           busy_a,
    output wire [W*L-1:0] vec_a_rx,
    output wire           done_a
);
    // A -> B link
    wire [W-1:0] ab_d, ab_rd_d; wire ab_send, ab_full, ab_rd, ab_empty;
    link_send #(W, L) u_send_a (
        .clk(clk_a), .rst_n(rst_a_n), .start(start_a), .vec(vec_a_tx),
        .busy(busy_a), .tx_data(ab_d), .tx_send(ab_send), .tx_full(ab_full));
    ss_link #(W, 4) u_link_ab (
        .clk_tx(clk_a), .rst_tx_n(rst_a_n), .tx_data(ab_d), .tx_send(ab_send), .tx_full(ab_full),
        .clk_rx(clk_b), .rst_rx_n(rst_b_n), .rx_rd(ab_rd), .rx_data(ab_rd_d), .rx_empty(ab_empty));

    // B -> A link
    wire [W-1:0] ba_d, ba_rd_d; wire ba_send, ba_full, ba_rd, ba_empty;
    ss_link #(W, 4) u_link_ba (
        .clk_tx(clk_b), .rst_tx_n(rst_b_n), .tx_data(ba_d), .tx_send(ba_send), .tx_full(ba_full),
        .clk_rx(clk_a), .rst_rx_n(rst_a_n), .rx_rd(ba_rd), .rx_data(ba_rd_d), .rx_empty(ba_empty));
    link_recv #(W, L) u_recv_a (
        .clk(clk_a), .rst_n(rst_a_n), .rx_rd(ba_rd), .rx_data(ba_rd_d), .rx_empty(ba_empty),
        .vec(vec_a_rx), .done(done_a));

    // node B : autonomous mini_gg (link in = A->B, link out = B->A)
    mini_gg #(W, L) u_node_b (
        .clk(clk_b), .rst_n(rst_b_n),
        .rx_rd(ab_rd), .rx_data(ab_rd_d), .rx_empty(ab_empty),
        .tx_data(ba_d), .tx_send(ba_send), .tx_full(ba_full),
        .dbg_state());
endmodule
