`timescale 1ns/1ps
// sd_init_top - SD controller (with init) + SD card model (with init) + capture.
module sd_init_top #(parameter NBLK = 2) (
    input  wire clk, rstn, start,
    input  wire [31:0] rd_blk,
    output wire ready, done
);
    wire cs_n, sclk, mosi, miso;
    wire out_we; wire [7:0] out_data;
    sd_ctrl #(.HALF(2)) u_ctrl (
        .clk(clk), .rstn(rstn), .start(start), .rd_start(1'b0), .rd_blk(rd_blk),
        .cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso),
        .out_we(out_we), .out_data(out_data), .ready(ready), .done(done), .fail(), .crc_err());
    sd_card #(.NBLK(NBLK)) u_sd (.cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso));

    reg [7:0] cap [0:511];
    reg [9:0] widx;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) widx <= 0;
        else if (out_we) begin cap[widx] <= out_data; widx <= widx + 10'd1; end
    end
endmodule
