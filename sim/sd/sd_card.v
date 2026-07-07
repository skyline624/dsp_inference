`timescale 1ns/1ps
// =============================================================================
// sd_card - behavioral SD card (SPI mode) with the INIT handshake (SD-5)
//
// Unlike sd_spi_model (which assumed an initialised card), this models the real
// power-up protocol so the controller's init sequence can be validated:
//   CMD0  (0x40) -> R1=0x01 (idle)
//   CMD8  (0x48) -> R7  : 0x01, 00 00 01 AA (voltage echo, SD v2)
//   CMD55 (0x77) -> R1=0x01 (app cmd)
//   CMD41 (0x69) -> R1=0x01 (busy) a few times, then 0x00 (ready)
//   CMD58 (0x7A) -> R3  : 0x00, OCR (CCS=1 -> SDHC, block addressing)
//   CMD17 (0x51) -> R1=0x00, 0xFE, 512 data, 2 CRC
// RX uses command-FRAMING (a command starts with bits [7:6]==01), so it resyncs
// regardless of how many 0xFF poll bytes the host inserts.
// =============================================================================
module sd_card #(parameter NBLK = 2) (
    input  wire cs_n, sclk, mosi, output wire miso
);
    localparam NB = NBLK*512;
    reg [7:0] mem [0:NB-1];

    // ---- RX : command framing ----
    reg [7:0] rxsh; reg [2:0] rxbit, ci; reg scanning;
    reg [7:0] cb1,cb2,cb3,cb4; reg [7:0] op; reg [31:0] blkaddr; reg cmd_tgl;
    wire [7:0] rxb = {rxsh[6:0], mosi};

    // ---- TX : response sequencer ----
    reg [7:0] txsh; reg [2:0] txbit;
    reg responding, cmd_seen; reg [9:0] residx; reg [7:0] rop; reg [31:0] dbase; reg [3:0] a41;
    assign miso = cs_n ? 1'b1 : txsh[7];

    always @(posedge sclk or posedge cs_n) begin
        if (cs_n) begin rxbit<=0; ci<=0; scanning<=1; cmd_tgl<=0; end
        else begin
            rxsh <= {rxsh[6:0], mosi};
            if (rxbit==3'd7) begin
                rxbit<=0;
                if (scanning) begin
                    if (rxb[7:6]==2'b01) begin op<=rxb; ci<=0; scanning<=0; end  // command start
                end else begin
                    case (ci) 3'd0:cb1<=rxb; 3'd1:cb2<=rxb; 3'd2:cb3<=rxb; default:cb4<=rxb; endcase
                    if (ci==3'd4) begin blkaddr<={cb1,cb2,cb3,cb4}; cmd_tgl<=~cmd_tgl; scanning<=1; ci<=0; end
                    else ci<=ci+3'd1;
                end
            end else rxbit<=rxbit+3'd1;
        end
    end

    function [9:0] rlen; input [7:0] o;
        case (o) 8'h48:rlen=5; 8'h7A:rlen=5; 8'h51:rlen=516; default:rlen=1; endcase
    endfunction
    function [15:0] blk_crc; input [31:0] base;
        reg [15:0] c; integer j, k; reg [7:0] d;
        begin
            c = 16'h0000;
            for (j=0;j<512;j=j+1) begin
                d = mem[base+j]; c = c ^ {d, 8'h00};
                for (k=0;k<8;k=k+1) c = c[15] ? ((c<<1)^16'h1021) : (c<<1);
            end
            blk_crc = c;
        end
    endfunction
    function [7:0] rbyte; input [7:0] o; input [9:0] i; input [31:0] base; input [3:0] a;
        reg [15:0] cc;
        begin
        cc = (o==8'h51) ? blk_crc(base) : 16'h0000;
        case (o)
            8'h40: rbyte = 8'h01;
            8'h48: case (i) 0:rbyte=8'h01;1:rbyte=8'h00;2:rbyte=8'h00;3:rbyte=8'h01;default:rbyte=8'hAA; endcase
            8'h77: rbyte = 8'h01;
            8'h69: rbyte = (a < 4'd2) ? 8'h01 : 8'h00;
            8'h7A: case (i) 0:rbyte=8'h00;1:rbyte=8'hC0;2:rbyte=8'hFF;3:rbyte=8'h80;default:rbyte=8'h00; endcase
            8'h51: rbyte = (i==0) ? 8'h00 : (i==1) ? 8'hFE : (i<=513) ? mem[base+(i-2)]
                         : (i==10'd514) ? cc[15:8] : cc[7:0];
            default: rbyte = 8'h01;
        endcase
        end
    endfunction

    always @(negedge sclk or posedge cs_n) begin
        if (cs_n) begin txsh<=8'hFF; txbit<=0; responding<=0; cmd_seen<=0; residx<=0; a41<=0; end
        else begin
            if (txbit==3'd7) begin
                txbit<=0;
                if (responding) begin
                    if (residx == rlen(rop)-10'd1) begin
                        responding<=0; txsh<=8'hFF;
                        if (rop==8'h69) a41<=a41+4'd1;          // advance ACMD41 readiness
                    end else begin txsh<=rbyte(rop, residx+10'd1, dbase, a41); residx<=residx+10'd1; end
                end else if (cmd_tgl != cmd_seen) begin
                    cmd_seen<=cmd_tgl; responding<=1; rop<=op; dbase<=blkaddr*512; residx<=0;
                    txsh<=rbyte(op, 10'd0, blkaddr*512, a41);
                end else txsh<=8'hFF;
            end else begin txsh<={txsh[6:0],1'b1}; txbit<=txbit+3'd1; end
        end
    end
endmodule
