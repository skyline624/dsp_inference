`timescale 1ns/1ps
// Standalone validation of sd_boot: SD card model -> real SDRAM (multi-block).
module sd_boot_top2 #(parameter NBLK = 2) (
    input wire clk, rstn, start, output wire done
);
    wire clk_sdram = ~clk;
    wire sd_clk, sd_cmd, sd_dat0, sd_dat3;
    wire [22:0] m_addr; wire [7:0] m_din; wire m_wr; wire sd_busy;

    sd_boot #(.HALF(4), .NBLK(NBLK[15:0])) u_boot (
        .clk(clk), .rstn(rstn), .start(start),
        .sd_clk(sd_clk), .sd_cmd(sd_cmd), .sd_dat0(sd_dat0), .sd_dat3(sd_dat3),
        .mem_addr(m_addr), .mem_din(m_din), .mem_wr(m_wr), .sdram_busy(sd_busy), .done(done));

    sd_card #(.NBLK(NBLK)) u_card (.cs_n(sd_dat3), .sclk(sd_clk), .mosi(sd_cmd), .miso(sd_dat0));

    wire [31:0] DQ; wire [10:0] A; wire [1:0] BA;
    wire nCS,nWE,nRAS,nCAS,SCLK,CKE; wire [3:0] DQM; wire [7:0] dout; wire dready;
    sdram #(.FREQ(2_000_000)) u_ctrl (
        .SDRAM_DQ(DQ), .SDRAM_A(A), .SDRAM_BA(BA), .SDRAM_nCS(nCS), .SDRAM_nWE(nWE),
        .SDRAM_nRAS(nRAS), .SDRAM_nCAS(nCAS), .SDRAM_CLK(SCLK), .SDRAM_CKE(CKE), .SDRAM_DQM(DQM),
        .clk(clk), .clk_sdram(clk_sdram), .resetn(rstn),
        .rd(1'b0), .wr(m_wr), .refresh(1'b0), .addr(m_addr), .din(m_din),
        .dout(dout), .dout32(), .data_ready(dready), .busy(sd_busy));
    sdram_chip u_chip (
        .SDRAM_CLK(SCLK), .SDRAM_CKE(CKE), .SDRAM_nCS(nCS), .SDRAM_nRAS(nRAS),
        .SDRAM_nCAS(nCAS), .SDRAM_nWE(nWE), .SDRAM_BA(BA), .SDRAM_A(A), .SDRAM_DQM(DQM), .SDRAM_DQ(DQ));
endmodule
