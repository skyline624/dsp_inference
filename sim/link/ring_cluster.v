`timescale 1ns/1ps
// =============================================================================
// ring_cluster - N nodes in a ring doing a scalable all-gather (any N >= 2)
//
// N ring_nodes, N ring links (edge g : node g -> node (g+1)%N). Each node has
// exactly one in + one out link => adding nodes does NOT add links per node.
// After the all-gather every node holds the full vector (all N slices).
//
// Single clock here to keep the N-scalability test simple; each link still uses
// the async FIFO (CDC-ready), and independent-clock operation is already proven
// in tasks 1/2 (cluster_link2 / bp_compare).
// =============================================================================
module ring_cluster #(
    parameter W = 8,
    parameter S = 32,
    parameter N = 2
) (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire [N*W*S-1:0]   my_slices,    // node g's input slice
    output wire [N*W*S*N-1:0] full_vecs,    // node g's assembled full vector
    output wire [N-1:0]       done
);
    localparam SB = W*S;

    wire [W-1:0] edata [0:N-1];
    wire         esend [0:N-1];
    wire         efull [0:N-1];
    wire [W-1:0] erdd  [0:N-1];
    wire         erd   [0:N-1];
    wire         eempty[0:N-1];

    genvar g;
    generate
        // ring links : edge g carries node g -> node (g+1)%N  (depth 64 > slice)
        for (g = 0; g < N; g = g + 1) begin : links
            ss_link #(W, 6) u_link (
                .clk_tx(clk), .rst_tx_n(rst_n),
                .tx_data(edata[g]), .tx_send(esend[g]), .tx_full(efull[g]),
                .clk_rx(clk), .rst_rx_n(rst_n),
                .rx_rd(erd[g]), .rx_data(erdd[g]), .rx_empty(eempty[g]));
        end
        // nodes : node g out=edge g, in=edge (g-1+N)%N
        for (g = 0; g < N; g = g + 1) begin : nodes
            localparam PREV = (g + N - 1) % N;
            ring_node #(W, S, N, g) u_node (
                .clk(clk), .rst_n(rst_n), .start(start),
                .my_slice(my_slices[g*SB +: SB]),
                .o_data(edata[g]), .o_send(esend[g]), .o_full(efull[g]),
                .i_rd(erd[PREV]), .i_data(erdd[PREV]), .i_empty(eempty[PREV]),
                .full_vec(full_vecs[g*SB*N +: SB*N]), .done(done[g]));
        end
    endgenerate
endmodule
