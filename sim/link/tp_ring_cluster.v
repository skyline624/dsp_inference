`timescale 1ns/1ps
// =============================================================================
// tp_ring_cluster - NN autonomous tp_nodes in a ring (real mini-GG, inc.3.1)
//
// Broadcast x to all nodes; each node computes its NR-row matmul slice with the
// real DSP and the ring all-gather distributes the slices -> every node holds
// the full NN*NR output. Fully autonomous (no PC), scalable in NN.
// =============================================================================
module tp_ring_cluster #(
    parameter W  = 8,
    parameter K  = 64,
    parameter NR = 16,
    parameter NN = 4
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,
    input  wire [W*K-1:0]      x,            // broadcast to all nodes
    output wire [NN*W*NR*NN-1:0] full_ys,
    output wire [NN-1:0]       done
);
    wire [W-1:0] edata [0:NN-1];
    wire         esend [0:NN-1];
    wire         efull [0:NN-1];
    wire [W-1:0] erdd  [0:NN-1];
    wire         erd   [0:NN-1];
    wire         eempty[0:NN-1];

    genvar g;
    generate
        for (g = 0; g < NN; g = g + 1) begin : links
            ss_link #(W, 6) u_link (
                .clk_tx(clk), .rst_tx_n(rst_n),
                .tx_data(edata[g]), .tx_send(esend[g]), .tx_full(efull[g]),
                .clk_rx(clk), .rst_rx_n(rst_n),
                .rx_rd(erd[g]), .rx_data(erdd[g]), .rx_empty(eempty[g]));
        end
        for (g = 0; g < NN; g = g + 1) begin : nodes
            localparam PREV = (g + NN - 1) % NN;
            tp_node #(W, K, NR, NN, g) u_node (
                .clk(clk), .rst_n(rst_n), .start(start), .my_x(x),
                .o_data(edata[g]), .o_send(esend[g]), .o_full(efull[g]),
                .i_rd(erd[PREV]), .i_data(erdd[PREV]), .i_empty(eempty[PREV]),
                .full_y(full_ys[g*W*NR*NN +: W*NR*NN]), .done(done[g]));
        end
    endgenerate
endmodule
