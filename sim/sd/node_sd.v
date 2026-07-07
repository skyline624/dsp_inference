`timescale 1ns/1ps
// node_sd - a routable node that boots its model from SD (top+SD_BOOT) + SD card
//           model + SDRAM chip model. Full integration, for cocotb validation.
module node_sd #(
    parameter [15:0] NBLK = 16'd2,
    parameter        SDHALF = 4
) (
    input  wire       clk,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [5:0] led
);
    wire o_clk,o_cke,o_cs,o_cas,o_ras,o_wen; wire [10:0] o_addr; wire [1:0] o_ba; wire [3:0] o_dqm;
    wire [31:0] io_dq;
    wire s_clk, s_cmd, s_dat0, s_dat3;

    top #(.SD_HALF(SDHALF), .SD_NBLK(NBLK)) u_fpga (
        .clk(clk), .uart_rx(uart_rx), .uart_tx(uart_tx), .led(led),
        .O_sdram_clk(o_clk), .O_sdram_cke(o_cke), .O_sdram_cs_n(o_cs),
        .O_sdram_cas_n(o_cas), .O_sdram_ras_n(o_ras), .O_sdram_wen_n(o_wen),
        .IO_sdram_dq(io_dq), .O_sdram_addr(o_addr), .O_sdram_ba(o_ba), .O_sdram_dqm(o_dqm),
        .sd_clk(s_clk), .sd_cmd(s_cmd), .sd_dat0(s_dat0), .sd_dat3(s_dat3));

    sd_card #(.NBLK(NBLK)) u_card (.cs_n(s_dat3), .sclk(s_clk), .mosi(s_cmd), .miso(s_dat0));

    sdram_chip u_sdram (
        .SDRAM_CLK(o_clk), .SDRAM_CKE(o_cke), .SDRAM_nCS(o_cs), .SDRAM_nRAS(o_ras),
        .SDRAM_nCAS(o_cas), .SDRAM_nWE(o_wen), .SDRAM_BA(o_ba), .SDRAM_A(o_addr),
        .SDRAM_DQM(o_dqm), .SDRAM_DQ(io_dq));
endmodule
