`timescale 1ns/1ps
// Sim wrapper: the real harness (src/sd_bringup) driving the SD card model.
module sd_bringup_tb (input wire clk);
    wire sd_clk, sd_cmd, sd_dat0, sd_dat3, uart_tx;
    wire [5:0] led;
    sd_bringup u_dut (
        .clk(clk), .uart_tx(uart_tx), .led(led),
        .sd_clk(sd_clk), .sd_cmd(sd_cmd), .sd_dat0(sd_dat0), .sd_dat3(sd_dat3));
    sd_card #(.NBLK(1)) u_card (
        .cs_n(sd_dat3), .sclk(sd_clk), .mosi(sd_cmd), .miso(sd_dat0));
endmodule
