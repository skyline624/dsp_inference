`timescale 1ns/1ps
// =============================================================================
// sd_bringup - standalone SD-card bring-up harness for the Tang Nano 20K
//
// Validates the SD controller on REAL silicon before integrating into top.v:
// on power-on it initialises a real SD card (CMD0/CMD8/ACMD41/CMD58), reads
// block 0, and streams it to the PC over UART, framed as:  0xA5 0x5A <status>
// <512 data bytes>   (status 0x01 = OK, 0xEE = init/read failed/no card).
// Then it waits ~1.2 s and repeats, so the PC keeps receiving fresh frames.
//
// LEDs (active-low): [0]=init ready  [1]=last read OK  [2]=error  [5]=heartbeat
// SPI clock kept slow (~400 kHz, HALF=34) so init works with any card.
// Pins (Tang Nano 20K): clk=4, uart_tx=69, led=15..20,
//   sd_clk=83, sd_cmd(MOSI)=82, sd_dat0(MISO)=84, sd_dat3(CS)=81.
// =============================================================================
module sd_bringup (
    input  wire       clk,          // 27 MHz
    output wire       uart_tx,      // -> PC
    output reg  [5:0] led,          // active low
    output wire       sd_clk,       // SD_CLK
    output wire       sd_cmd,       // SD_CMD  (SPI MOSI)
    input  wire       sd_dat0,      // SD_DAT0 (SPI MISO)
    output wire       sd_dat3       // SD_DAT3 (SPI CS#)
);
    // power-on reset (~2.4 ms) + heartbeat
    reg [15:0] rstc = 16'd0;
    wire rstn = &rstc;
    always @(posedge clk) if (!(&rstc)) rstc <= rstc + 16'd1;
    reg [24:0] hb = 25'd0;
    always @(posedge clk) hb <= hb + 25'd1;

    // SD controller (slow SPI for robust init)
    reg  ctrl_start;
    wire ctrl_ready, ctrl_done, c_we; wire [7:0] c_data;
    sd_ctrl #(.HALF(34)) u_ctrl (
        .clk(clk), .rstn(rstn), .start(ctrl_start), .rd_start(1'b0), .rd_blk(32'd0),
        .cs_n(sd_dat3), .sclk(sd_clk), .mosi(sd_cmd), .miso(sd_dat0),
        .out_we(c_we), .out_data(c_data), .ready(ctrl_ready), .done(ctrl_done), .fail(), .crc_err());

    // captured block
    reg [7:0] cap [0:511];
    reg [9:0] widx;

    // UART tx @ 1 Mbaud
    reg [7:0] tx_data; reg tx_send; wire tx_busy;
    uart_tx_8n1 #(.DIV(27)) u_tx (.clk(clk), .rst(~rstn), .data(tx_data), .send(tx_send), .tx(uart_tx), .busy(tx_busy));

    reg [7:0]  status;
    reg [9:0]  scnt;
    reg        ready_l, err_l;
    reg [23:0] wdog;
    reg [24:0] dly;

    function [7:0] framebyte; input [9:0] i;
        framebyte = (i==10'd0) ? 8'hA5 : (i==10'd1) ? 8'h5A : (i==10'd2) ? status : cap[i-10'd3];
    endfunction

    localparam H_START=0, H_RUN=1, H_SET=2, H_BUSY=3, H_DRAIN=4, H_DLY=5;
    reg [2:0] hst;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            hst<=H_START; ctrl_start<=0; tx_send<=0; widx<=0; scnt<=0; wdog<=0; dly<=0;
            ready_l<=0; err_l<=0; status<=8'h00;
        end else begin
            ctrl_start<=0; tx_send<=0;
            if (ctrl_ready) ready_l<=1;
            case (hst)
                H_START: begin ctrl_start<=1; widx<=0; wdog<=0; hst<=H_RUN; end
                H_RUN: begin
                    if (c_we) begin cap[widx]<=c_data; widx<=widx+10'd1; end
                    wdog<=wdog+24'd1;
                    if (ctrl_done)            begin status<=8'h01; scnt<=0; hst<=H_SET; end
                    else if (wdog==24'hFFFFFF) begin status<=8'hEE; err_l<=1; scnt<=0; hst<=H_SET; end
                end
                H_SET:  begin tx_data<=framebyte(scnt); tx_send<=1; hst<=H_BUSY; end
                H_BUSY: if (tx_busy) hst<=H_DRAIN;
                H_DRAIN: if (!tx_busy) begin
                            if (scnt==10'd514) begin dly<=0; hst<=H_DLY; end
                            else begin scnt<=scnt+10'd1; hst<=H_SET; end
                         end
                H_DLY: begin dly<=dly+25'd1; if (dly==25'h1FFFFFF) begin err_l<=0; hst<=H_START; end end
            endcase
        end
    end

    always @(*) begin
        led = 6'b111111;              // active low : 0 = on
        led[0] = ~ready_l;            // init reached ready
        led[1] = ~(status==8'h01);    // last read OK
        led[2] = ~err_l;              // error / no card
        led[5] = ~hb[24];             // heartbeat
    end
endmodule
