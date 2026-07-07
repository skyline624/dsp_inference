`timescale 1ns/1ps
// sd_stream_top - SD controller + SD card model + a capture buffer.
// Proves the read path: data stored on the (model) SD card streams out byte-exact.
module sd_stream_top #(parameter NBLK = 2) (
    input  wire clk, rstn, start,
    output wire done
);
    wire cs_n, sclk, mosi, miso;
    wire out_we; wire [7:0] out_data;

    sd_reader #(.HALF(2)) u_rd (
        .clk(clk), .rstn(rstn), .start(start),
        .blk_start(16'd0), .nblk(NBLK[15:0]),
        .cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso),
        .out_we(out_we), .out_data(out_data), .done(done));

    sd_spi_model #(.NBLK(NBLK)) u_sd (
        .cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso));

    reg [7:0] cap [0:NBLK*512-1];
    reg [19:0] widx;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) widx <= 0;
        else if (out_we) begin cap[widx] <= out_data; widx <= widx + 20'd1; end
    end
endmodule
