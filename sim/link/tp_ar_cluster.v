`timescale 1ns/1ps
// =============================================================================
// tp_ar_cluster - NN autonomous column-parallel matmul nodes + ring all-reduce
//
// Each node gets its input-columns slice and its W columns; computes a partial,
// and the ring all-reduce sums them so every node holds the full result. The
// ring carries 32-bit partial elements (depth 128 > D). Autonomous, scalable.
// =============================================================================
module tp_ar_cluster #(
    parameter W  = 8,
    parameter D  = 64,
    parameter KS = 32,
    parameter NN = 2
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [NN*W*KS-1:0] x_slices,     // node g's input-columns slice
    output wire [NN*W*D-1:0]  outs,         // node g's full result
    output wire [NN-1:0]     done
);
    wire [31:0] edata [0:NN-1];
    wire        esend [0:NN-1];
    wire        efull [0:NN-1];
    wire [31:0] erdd  [0:NN-1];
    wire        erd   [0:NN-1];
    wire        eempty[0:NN-1];

    genvar g;
    generate
        for (g = 0; g < NN; g = g + 1) begin : links
            ss_link #(32, 7) u_link (         // 32-bit elems, depth 128 > D
                .clk_tx(clk), .rst_tx_n(rst_n),
                .tx_data(edata[g]), .tx_send(esend[g]), .tx_full(efull[g]),
                .clk_rx(clk), .rst_rx_n(rst_n),
                .rx_rd(erd[g]), .rx_data(erdd[g]), .rx_empty(eempty[g]));
        end
        for (g = 0; g < NN; g = g + 1) begin : nodes
            localparam PREV = (g + NN - 1) % NN;
            tp_ar_node #(W, D, KS, NN, g) u_node (
                .clk(clk), .rst_n(rst_n), .start(start),
                .x_slice(x_slices[g*W*KS +: W*KS]),
                .o_data(edata[g]), .o_send(esend[g]), .o_full(efull[g]),
                .i_rd(erd[PREV]), .i_data(erdd[PREV]), .i_empty(eempty[PREV]),
                .out(outs[g*W*D +: W*D]), .done(done[g]));
        end
    endgenerate
endmodule
