`timescale 1ns/1ps
// =============================================================================
// node_top - one FPGA node = top.v + its own behavioral SDRAM chip
//
// Self-contained simulation unit representing a single Tang Nano 20K board.
// Exposes only clk + the UART pins, so nodes can be daisy-chained into a
// cluster (uart_tx of node i -> uart_rx of node i+1).
// =============================================================================
module node_top (
    input  wire       clk,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led
);
    wire        o_sdram_clk, o_sdram_cke, o_sdram_cs_n;
    wire        o_sdram_cas_n, o_sdram_ras_n, o_sdram_wen_n;
    wire [10:0] o_sdram_addr;
    wire [1:0]  o_sdram_ba;
    wire [3:0]  o_sdram_dqm;
    wire [31:0] io_sdram_dq;

    top u_fpga (
        .clk          (clk),
        .uart_rx      (uart_rx),
        .uart_tx      (uart_tx),
        .led          (led),
        .O_sdram_clk  (o_sdram_clk),
        .O_sdram_cke  (o_sdram_cke),
        .O_sdram_cs_n (o_sdram_cs_n),
        .O_sdram_cas_n(o_sdram_cas_n),
        .O_sdram_ras_n(o_sdram_ras_n),
        .O_sdram_wen_n(o_sdram_wen_n),
        .IO_sdram_dq  (io_sdram_dq),
        .O_sdram_addr (o_sdram_addr),
        .O_sdram_ba   (o_sdram_ba),
        .O_sdram_dqm  (o_sdram_dqm)
    );

    sdram_chip u_sdram (
        .SDRAM_CLK (o_sdram_clk),
        .SDRAM_CKE (o_sdram_cke),
        .SDRAM_nCS (o_sdram_cs_n),
        .SDRAM_nRAS(o_sdram_ras_n),
        .SDRAM_nCAS(o_sdram_cas_n),
        .SDRAM_nWE (o_sdram_wen_n),
        .SDRAM_BA  (o_sdram_ba),
        .SDRAM_A   (o_sdram_addr),
        .SDRAM_DQM (o_sdram_dqm),
        .SDRAM_DQ  (io_sdram_dq)
    );
endmodule
