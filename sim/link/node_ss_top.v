`timescale 1ns/1ps
// =============================================================================
// node_ss_top - one brick node (top.v, LINK_SS) driven over the parallel link
// instead of UART. Phase-1 node-side gate : proves top.v speaks the command/
// response protocol identically when its serial PHY is swapped for the fifo
// adapters. A testbench-side pair of fifos stands in for the host link :
//   host -> node : cmd fifo   (cocotb writes command bytes)
//   node -> host : resp fifo  (cocotb reads response bytes)
// Single clock here (the CDC itself is covered by test_uart_bridge).
// =============================================================================
module node_ss_top (
    input  wire       clk,
    input  wire       rst,           // active-high : reset the testbench-side fifos
    // command port : cocotb pushes bytes to the node
    input  wire [7:0] cmd_data,
    input  wire       cmd_wr,
    output wire       cmd_full,
    // response port : cocotb pulls bytes from the node
    output wire [7:0] resp_data,
    output wire       resp_empty,
    input  wire       resp_rd,
    output wire [5:0] led
);
    wire rst_n = ~rst;

    // host -> node command fifo (cocotb writes, node reads)
    wire [7:0] c_rdata; wire c_empty; wire c_rd;
    async_fifo #(.DW(8), .AW(4)) u_cmd_fifo (
        .wclk(clk), .wrst_n(rst_n), .winc(cmd_wr), .wdata(cmd_data), .wfull(cmd_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(c_rd),   .rdata(c_rdata),  .rempty(c_empty));

    // node -> host response fifo (node writes, cocotb reads)
    wire [7:0] r_wdata; wire r_wr; wire r_full;
    async_fifo #(.DW(8), .AW(4)) u_resp_fifo (
        .wclk(clk), .wrst_n(rst_n), .winc(r_wr), .wdata(r_wdata), .wfull(r_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(resp_rd), .rdata(resp_data), .rempty(resp_empty));

    // SDRAM wires
    wire        o_sdram_clk, o_sdram_cke, o_sdram_cs_n;
    wire        o_sdram_cas_n, o_sdram_ras_n, o_sdram_wen_n;
    wire [10:0] o_sdram_addr; wire [1:0] o_sdram_ba; wire [3:0] o_sdram_dqm;
    wire [31:0] io_sdram_dq;

    top u_fpga (
        .clk         (clk),
        .lk_rx_data  (c_rdata),
        .lk_rx_empty (c_empty),
        .lk_rx_rd    (c_rd),
        .lk_tx_data  (r_wdata),
        .lk_tx_wr    (r_wr),
        .lk_tx_full  (r_full),
        .led         (led),
        .O_sdram_clk (o_sdram_clk),  .O_sdram_cke (o_sdram_cke),  .O_sdram_cs_n(o_sdram_cs_n),
        .O_sdram_cas_n(o_sdram_cas_n),.O_sdram_ras_n(o_sdram_ras_n),.O_sdram_wen_n(o_sdram_wen_n),
        .IO_sdram_dq (io_sdram_dq),  .O_sdram_addr(o_sdram_addr),  .O_sdram_ba  (o_sdram_ba),
        .O_sdram_dqm (o_sdram_dqm));

    sdram_chip u_sdram (
        .SDRAM_CLK(o_sdram_clk), .SDRAM_CKE(o_sdram_cke), .SDRAM_nCS(o_sdram_cs_n),
        .SDRAM_nRAS(o_sdram_ras_n), .SDRAM_nCAS(o_sdram_cas_n), .SDRAM_nWE(o_sdram_wen_n),
        .SDRAM_BA(o_sdram_ba), .SDRAM_A(o_sdram_addr), .SDRAM_DQM(o_sdram_dqm),
        .SDRAM_DQ(io_sdram_dq));
endmodule
