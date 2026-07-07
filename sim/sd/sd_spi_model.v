`timescale 1ns/1ps
// =============================================================================
// sd_spi_model - behavioral SD card model (SPI mode), READ path (sim only)
//
// Mirrors a real SD card's SPI single-block read (CMD17) so we can validate the
// SD controller before touching hardware. Assumed already initialised (the SPI
// init handshake CMD0/ACMD41 is mechanical and added later for real HW).
//
// Protocol (SPI mode 0): master sends CMD17 = {0x51, addr[31:0], crc}; the card
// then streams, on the master's clocks: R1(0x00), data token 0xFE, 512 data
// bytes (from mem[addr*512 ..]), 2 CRC bytes. Byte-aligned, self-resyncing.
// =============================================================================
module sd_spi_model #(parameter NBLK = 2) (
    input  wire cs_n,
    input  wire sclk,
    input  wire mosi,
    output wire miso
);
    localparam NB = NBLK*512;
    reg [7:0] mem [0:NB-1];

    // ---- RX (sample MOSI on rising sclk) : collect 6-byte command ----
    reg [7:0] rxsh; reg [2:0] rxbit, ci;
    reg [7:0] cb0, cb1, cb2, cb3, cb4;
    reg       cmd_tgl; reg [31:0] blkaddr;
    wire [7:0] rxb = {rxsh[6:0], mosi};

    // ---- TX (drive MISO on falling sclk) : response sequencer ----
    reg [7:0] txsh; reg [2:0] txbit;
    reg       responding, cmd_seen; reg [9:0] residx; reg [31:0] dbase;
    assign miso = cs_n ? 1'b1 : txsh[7];

    always @(posedge sclk or posedge cs_n) begin
        if (cs_n) begin rxbit<=0; ci<=0; cmd_tgl<=0; end
        else begin
            rxsh <= {rxsh[6:0], mosi};
            if (rxbit==3'd7) begin
                rxbit<=0;
                if (!responding) begin
                    case (ci)
                        3'd0: cb0<=rxb; 3'd1: cb1<=rxb; 3'd2: cb2<=rxb; 3'd3: cb3<=rxb; 3'd4: cb4<=rxb;
                        default: if (cb0==8'h51) begin blkaddr<={cb1,cb2,cb3,cb4}; cmd_tgl<=~cmd_tgl; end
                    endcase
                    ci <= (ci==3'd5) ? 3'd0 : ci+3'd1;
                end
            end else rxbit<=rxbit+3'd1;
        end
    end

    function [7:0] rbyte; input [9:0] idx; input [31:0] base;
        if      (idx==10'd0) rbyte = 8'h00;          // R1 = ready
        else if (idx==10'd1) rbyte = 8'hFE;          // data start token
        else if (idx<=10'd513) rbyte = mem[base + (idx-10'd2)];
        else rbyte = 8'hFF;                          // CRC (don't care in SPI)
    endfunction

    always @(negedge sclk or posedge cs_n) begin
        if (cs_n) begin txsh<=8'hFF; txbit<=0; responding<=0; cmd_seen<=0; residx<=0; end
        else begin
            if (txbit==3'd7) begin
                txbit<=0;
                if (responding) begin
                    if (residx==10'd515) begin responding<=0; txsh<=8'hFF; end
                    else begin txsh<=rbyte(residx+10'd1, dbase); residx<=residx+10'd1; end
                end else if (cmd_tgl != cmd_seen) begin
                    cmd_seen<=cmd_tgl; responding<=1; dbase<=blkaddr*512;
                    residx<=10'd0; txsh<=rbyte(10'd0, blkaddr*512);
                end else txsh<=8'hFF;
            end else begin txsh<={txsh[6:0],1'b1}; txbit<=txbit+3'd1; end
        end
    end
endmodule
