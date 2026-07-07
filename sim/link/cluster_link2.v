`timescale 1ns/1ps
// =============================================================================
// cluster_link2 - two nodes, full-duplex activation exchange over ss_link
//
// Models the real inter-node transport of the cluster: node A and node B run on
// INDEPENDENT clocks and exchange activation vectors directly (A->B and B->A),
// each direction = link_send -> ss_link (async FIFO CDC) -> link_recv.
//
// This is the parallel/source-synchronous replacement for the UART used in the
// functional cluster sims (M3..M4b): same byte payloads, ~270x faster, µs latency.
// =============================================================================
module cluster_link2 #(
    parameter W = 8,
    parameter L = 64
) (
    input  wire           clk_a,
    input  wire           rst_a_n,
    input  wire           clk_b,
    input  wire           rst_b_n,

    // node A: transmit vec_a_tx to B, receive vec_a_rx from B
    input  wire           start_a,
    input  wire [W*L-1:0] vec_a_tx,
    output wire           busy_a,
    output wire [W*L-1:0] vec_a_rx,
    output wire           done_a,        // A finished receiving from B

    // node B: transmit vec_b_tx to A, receive vec_b_rx from A
    input  wire           start_b,
    input  wire [W*L-1:0] vec_b_tx,
    output wire           busy_b,
    output wire [W*L-1:0] vec_b_rx,
    output wire           done_b         // B finished receiving from A
);
    // ---- link A -> B ----
    wire [W-1:0] ab_d, ab_rd_d; wire ab_send, ab_full, ab_rd, ab_empty;
    link_send #(W, L) u_send_a (
        .clk(clk_a), .rst_n(rst_a_n), .start(start_a), .vec(vec_a_tx),
        .busy(busy_a), .tx_data(ab_d), .tx_send(ab_send), .tx_full(ab_full));
    ss_link #(W, 4) u_link_ab (
        .clk_tx(clk_a), .rst_tx_n(rst_a_n), .tx_data(ab_d), .tx_send(ab_send), .tx_full(ab_full),
        .clk_rx(clk_b), .rst_rx_n(rst_b_n), .rx_rd(ab_rd), .rx_data(ab_rd_d), .rx_empty(ab_empty));
    link_recv #(W, L) u_recv_b (
        .clk(clk_b), .rst_n(rst_b_n), .rx_rd(ab_rd), .rx_data(ab_rd_d), .rx_empty(ab_empty),
        .vec(vec_b_rx), .done(done_b));

    // ---- link B -> A ----
    wire [W-1:0] ba_d, ba_rd_d; wire ba_send, ba_full, ba_rd, ba_empty;
    link_send #(W, L) u_send_b (
        .clk(clk_b), .rst_n(rst_b_n), .start(start_b), .vec(vec_b_tx),
        .busy(busy_b), .tx_data(ba_d), .tx_send(ba_send), .tx_full(ba_full));
    ss_link #(W, 4) u_link_ba (
        .clk_tx(clk_b), .rst_tx_n(rst_b_n), .tx_data(ba_d), .tx_send(ba_send), .tx_full(ba_full),
        .clk_rx(clk_a), .rst_rx_n(rst_a_n), .rx_rd(ba_rd), .rx_data(ba_rd_d), .rx_empty(ba_empty));
    link_recv #(W, L) u_recv_a (
        .clk(clk_a), .rst_n(rst_a_n), .rx_rd(ba_rd), .rx_data(ba_rd_d), .rx_empty(ba_empty),
        .vec(vec_a_rx), .done(done_a));
endmodule
