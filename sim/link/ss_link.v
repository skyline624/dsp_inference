`timescale 1ns/1ps
// =============================================================================
// ss_link - source-synchronous parallel inter-FPGA link (one direction)
//
// Physical wires between the two boards (for DW=8):
//     A --> B :  lnk_clk (forwarded sender clock)   1
//                lnk_data[DW-1:0]                    DW
//                lnk_valid                           1
//     B --> A :  lnk_full (backpressure)            1     -> total DW+3 pins
//
// The sender drives data + valid + its own clock onto the pins. The receiver
// captures with that forwarded clock into an async FIFO, then reads on its own
// local clock -> the FIFO is the clock-domain-crossing. lnk_full crosses back in
// the sender's clock domain (same clock as the forwarded one), so no extra CDC.
//
// Drop-in replacement for the UART link: same byte granularity, but ~270x the
// throughput and µs-scale latency instead of ~650 µs per activation exchange.
// =============================================================================
module ss_link #(
    parameter DW = 8,
    parameter AW = 4
) (
    // sender side (node A, clk_tx domain)
    input  wire          clk_tx,
    input  wire          rst_tx_n,
    input  wire [DW-1:0] tx_data,
    input  wire          tx_send,
    output wire          tx_full,    // backpressure: do not send when high

    // receiver side (node B, clk_rx domain)
    input  wire          clk_rx,
    input  wire          rst_rx_n,
    input  wire          rx_rd,
    output wire [DW-1:0] rx_data,
    output wire          rx_empty
);
    // The sender's clock IS the FIFO write clock (forwarded over a pin);
    // tx_send & tx_data & ~tx_full model the data + valid pins.
    wire wr = tx_send & ~tx_full;

    async_fifo #(.DW(DW), .AW(AW)) u_fifo (
        .wclk   (clk_tx),
        .wrst_n (rst_tx_n),
        .winc   (wr),
        .wdata  (tx_data),
        .wfull  (tx_full),
        .rclk   (clk_rx),
        .rrst_n (rst_rx_n),
        .rinc   (rx_rd),
        .rdata  (rx_data),
        .rempty (rx_empty)
    );
endmodule
