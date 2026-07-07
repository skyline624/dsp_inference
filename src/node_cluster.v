`timescale 1ns/1ps
// =============================================================================
// node_cluster - INTEGRATED node "all-in" for LUT budget measurement (P&R only).
//
// Composes the 3 real pieces of a cluster node on ONE Tang Nano 20K, so a single
// gw_sh build gives the true LUT/BSRAM/DSP % of:
//   1. the brick node datapath      = top.v (NODE_ONLY + SD_BOOT + CLUSTER_NODE)
//   2. the on-chip mini-GG sequencer = ffn_tp_seq (drives the node over UART)
//   3. the inter-FPGA link           = link_send + async_fifo (source-synchronous CDC)
//
// This is a PLACE&ROUTE probe, not a functional test: an LFSR seeds pseudo-random
// inputs so the synthesizer cannot eliminate the sequencer/link logic, and the
// key outputs are XOR-reduced onto the LEDs so every module's logic reaches a
// pin. The % LUT is the official Gowin number for a cluster node "tout compris".
// No 2nd card needed: the link is looped back on-chip (its LUT cost is the same
// whether the wires leave the chip or not).
// =============================================================================
`define NODE_ONLY
`define SD_BOOT
`define CLUSTER_NODE
`include "top.v"

module node_cluster #(parameter W = 8, parameter D = 64) (
    input  wire        clk,             // 27 MHz crystal (pin 4)
    output wire [5:0]  led,
    // SDRAM
    output wire        O_sdram_clk,
    output wire        O_sdram_cke,
    output wire        O_sdram_cs_n,
    output wire        O_sdram_cas_n,
    output wire        O_sdram_ras_n,
    output wire        O_sdram_wen_n,
    inout  wire [31:0] IO_sdram_dq,
    output wire [10:0] O_sdram_addr,
    output wire [1:0]  O_sdram_ba,
    output wire [3:0]  O_sdram_dqm,
    // SD card
    output wire        sd_clk,
    output wire        sd_cmd,
    input  wire        sd_dat0,
    output wire        sd_dat3
);
    // ---- the brick node (own PLL -> clk_sys) ----
    wire        clk_sys;
    wire [5:0]  node_led;
    wire        n_uart_tx;              // node -> sequencer (node's UART tx)
    wire        n_uart_rx;              // sequencer -> node (node's UART rx)

    top #(.SD_HALF(135), .SD_NBLK(16'd2)) u_node (
        .clk(clk), .uart_rx(n_uart_rx), .uart_tx(n_uart_tx), .led(node_led),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke), .O_sdram_cs_n(O_sdram_cs_n),
        .O_sdram_cas_n(O_sdram_cas_n), .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .IO_sdram_dq(IO_sdram_dq), .O_sdram_addr(O_sdram_addr), .O_sdram_ba(O_sdram_ba),
        .O_sdram_dqm(O_sdram_dqm),
        .sd_clk(sd_clk), .sd_cmd(sd_cmd), .sd_dat0(sd_dat0), .sd_dat3(sd_dat3),
        .clk_sys_out(clk_sys));

    wire rst_n_sys = 1'b1;               // (probe) keep sequencer out of reset

    // ---- LFSR: pseudo-random seeds so logic is not eliminated ----
    reg [31:0] lfsr; wire [511:0] rnd;
    always @(posedge clk_sys) lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
    assign rnd = {lfsr, lfsr, lfsr, lfsr, lfsr, lfsr, lfsr, lfsr,
                  lfsr, lfsr, lfsr, lfsr, lfsr, lfsr, lfsr, lfsr};   // 512-bit

    // ---- on-chip mini-GG sequencer (FFN tensor-parallel, drives the node) ----
    wire [W*D-1:0] seq_result; wire signed [7:0] seq_result_sh; wire seq_done;
    // n1 is absent (single node): tie n1_tx to 0 so the mux reads idle.
    ffn_tp_seq2 #(.W(W), .D(D), .BOOT(20'd2000),
                 .A_RMS(23'h100000), .A_W1(23'h101000), .A_W3(23'h102000), .A_W2(23'h103000)) u_seq (
        .clk(clk_sys), .rst_n(rst_n_sys), .start(lfsr[0]), .x_in(rnd[W*D-1:0]),
        .sx_in(lfsr[7:0]), .sw_rms(lfsr[7:0]), .sw1(lfsr[7:0]), .sw3(lfsr[7:0]), .sw2(lfsr[7:0]),
        .base(lfsr[22:0]),
        .n0_rx(n_uart_rx), .n1_rx(),
        .n0_tx(n_uart_tx),  .n1_tx(1'b1),
        .result(seq_result), .result_sh(seq_result_sh), .done(seq_done));

    // ---- inter-FPGA link (source-synchronous, looped back on-chip) ----
    // write side (this node) streams an activation byte vector; the async_fifo
    // is the CDC into a divided read clock (stands in for the remote FPGA's
    // independent clock). The LUT cost of the CDC + framing is what we measure.
    wire [W-1:0]  tx_data;  wire tx_send; wire tx_full;
    wire [W-1:0]  rx_data;  wire rx_empty;
    wire          walmost_full;

    link_send #(.W(W), .L(D)) u_lsend (
        .clk(clk_sys), .rst_n(rst_n_sys), .start(lfsr[1]),
        .vec(rnd[W*D-1:0]), .busy(/*obs*/),
        .tx_data(tx_data), .tx_send(tx_send), .tx_full(tx_full));

    // divided clock for the CDC read domain (remote-FPGA stand-in)
    reg clk_div; always @(posedge clk_sys) clk_div <= ~clk_div;

    async_fifo #(.DW(W), .AW(4)) u_fifo (
        .wclk(clk_sys), .wrst_n(rst_n_sys),
        .winc(tx_send & ~tx_full), .wdata(tx_data), .wfull(tx_full), .walmost_full(walmost_full),
        .rclk(clk_div),  .rrst_n(rst_n_sys),
        .rinc(~rx_empty), .rdata(rx_data), .rempty(rx_empty));

    // ---- anti-elimination: drive every module's output onto the LEDs ----
    // XOR-reduce so each module contributes a non-constant function of its state.
    wire [5:0] extra = { seq_done,
                        ^seq_result[31:0],
                        ^seq_result[511:480],
                        seq_result_sh[3],
                        ^rx_data,
                        walmost_full };
    assign led = node_led ^ extra;
endmodule