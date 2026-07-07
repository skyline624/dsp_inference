`timescale 1ns/1ps
// =============================================================================
// sd_ctrl - SD controller with full SPI INIT sequence + block read (SD-5)
//
// Real-card-ready: on start it runs the power-up handshake, then reads a block.
//   power-on (>=74 clocks, CS high) -> CMD0 -> CMD8 -> {CMD55,CMD41}* -> CMD58
//   -> ready -> CMD17(rd_blk) -> stream 512 bytes.
// This is the RTL that drives a real SD card on the Tang Nano 20K's TF slot.
// =============================================================================
module sd_ctrl #(parameter HALF = 2) (
    input  wire clk, rstn, start,
    input  wire rd_start,          // read another block WITHOUT re-init (multi-block)
    input  wire [31:0] rd_blk,
    output reg  cs_n, output reg sclk, output reg mosi,
    input  wire miso,
    output reg  out_we, output reg [7:0] out_data,
    output reg  ready, done, fail, crc_err
);
    // ---- 8-bit SPI transfer engine (mode 0) ----
    reg eng_go, eng_done; reg [7:0] eng_tx, eng_rx;
    reg eng_busy; reg [3:0] bitn; reg [7:0] txs, rxs; reg [15:0] dv;
    reg miso_r, miso_r2;                       // register async MISO (clean sampling)
    always @(posedge clk) begin miso_r <= miso; miso_r2 <= miso_r; end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin eng_busy<=0; eng_done<=0; sclk<=0; mosi<=1; end
        else begin
            eng_done<=0;
            if (!eng_busy) begin
                if (eng_go) begin eng_busy<=1; txs<=eng_tx; rxs<=8'h00; bitn<=0; dv<=0; sclk<=0; mosi<=eng_tx[7]; end
            end else if (dv==HALF-1) begin
                dv<=0;
                if (sclk==1'b0) begin sclk<=1'b1; rxs<={rxs[6:0],miso_r}; end
                else begin
                    sclk<=1'b0;
                    if (bitn==4'd7) begin eng_busy<=0; eng_done<=1; eng_rx<=rxs; end
                    else begin bitn<=bitn+4'd1; txs<={txs[6:0],1'b0}; mosi<=txs[6]; end
                end
            end else dv<=dv+16'd1;
        end
    end

    reg [7:0] cmdbuf [0:5];
    reg [2:0] sidx; reg [10:0] pollc; reg [9:0] nread, rcnt; reg streamf; reg [7:0] r1; reg [3:0] pcnt;
    reg [5:0] st, ret;
    // block CRC16 (CCITT/XMODEM) : verify each 512-byte data block, re-read on error
    reg [15:0] crc16, rx_crc; reg [9:0] dcnt; reg ccnt; reg [4:0] rdretry;
    function [15:0] crc16_upd; input [15:0] c; input [7:0] d;
        reg [15:0] x; integer i;
        begin
            x = c ^ {d, 8'h00};
            for (i=0;i<8;i=i+1) x = x[15] ? ((x<<1)^16'h1021) : (x<<1);
            crc16_upd = x;
        end
    endfunction

    localparam S_IDLE=0, S_PWR=1, S_PWRW=2,
               SUB_SEND=3, SUB_SENDW=4, SUB_POLL=5, SUB_POLLW=6, SUB_TOK=7, SUB_TOKW=8, SUB_READ=9, SUB_READW=10,
               S_CMD0=11, S_CMD0P=12, S_CMD8=13, S_CMD8P=14, S_CMD8R=15,
               S_C55=16, S_C55P=17, S_C41=18, S_C41P=19, S_C41CHK=20,
               S_C58=21, S_C58P=22, S_C58R=23, S_READY=24,
               S_RD=25, S_RDP=26, S_RDTOK=27, S_RDDATA=28, S_RDCRC=29, S_DONE=30, S_FAIL=31,
               S_RIDLE=32, S_RGAP=33, S_RGAPW=34,
               S_DBYTE=35, S_DBYTE_W=36, S_CBYTE=37, S_CBYTE_W=38;

    task setcmd; input [7:0] b0,b1,b2,b3,b4,b5; begin
        cmdbuf[0]<=b0; cmdbuf[1]<=b1; cmdbuf[2]<=b2; cmdbuf[3]<=b3; cmdbuf[4]<=b4; cmdbuf[5]<=b5; sidx<=0;
    end endtask

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin st<=S_IDLE; eng_go<=0; cs_n<=1; out_we<=0; ready<=0; done<=0; fail<=0; crc_err<=0; rdretry<=0; end
        else begin
            eng_go<=0; out_we<=0; done<=0; crc_err<=0;
            case (st)
                S_IDLE: if (start) begin cs_n<=1; ready<=0; fail<=0; pcnt<=0; st<=S_PWR; end

                // power-on: 10 bytes 0xFF with CS high
                S_PWR:  begin eng_tx<=8'hFF; eng_go<=1; st<=S_PWRW; end
                S_PWRW: if (eng_done) begin if (pcnt==4'd9) begin cs_n<=0; st<=S_CMD0; end else begin pcnt<=pcnt+4'd1; st<=S_PWR; end end

                // ---- generic subroutines ----
                SUB_SEND:  begin eng_tx<=cmdbuf[sidx]; eng_go<=1; st<=SUB_SENDW; end
                SUB_SENDW: if (eng_done) begin if (sidx==3'd5) st<=ret; else begin sidx<=sidx+3'd1; st<=SUB_SEND; end end
                SUB_POLL:  begin eng_tx<=8'hFF; eng_go<=1; st<=SUB_POLLW; end
                SUB_POLLW: if (eng_done) begin
                               if (eng_rx[7]==1'b0) begin r1<=eng_rx; st<=ret; end
                               else if (pollc==11'd2000) st<=S_FAIL; else begin pollc<=pollc+11'd1; st<=SUB_POLL; end
                           end
                SUB_TOK:   begin eng_tx<=8'hFF; eng_go<=1; st<=SUB_TOKW; end
                SUB_TOKW:  if (eng_done) begin
                               if (eng_rx==8'hFE) st<=ret;
                               else if (pollc==11'd2000) st<=S_FAIL; else begin pollc<=pollc+11'd1; st<=SUB_TOK; end
                           end
                SUB_READ:  begin eng_tx<=8'hFF; eng_go<=1; st<=SUB_READW; end
                SUB_READW: if (eng_done) begin
                               if (streamf) begin out_data<=eng_rx; out_we<=1; end
                               if (rcnt==nread-10'd1) st<=ret; else begin rcnt<=rcnt+10'd1; st<=SUB_READ; end
                           end

                // ---- init sequence ----
                S_CMD0:  begin setcmd(8'h40,8'h00,8'h00,8'h00,8'h00,8'h95); ret<=S_CMD0P; st<=SUB_SEND; end
                S_CMD0P: begin pollc<=0; ret<=S_CMD8; st<=SUB_POLL; end
                S_CMD8:  begin setcmd(8'h48,8'h00,8'h00,8'h01,8'hAA,8'h87); ret<=S_CMD8P; st<=SUB_SEND; end
                S_CMD8P: begin pollc<=0; ret<=S_CMD8R; st<=SUB_POLL; end
                S_CMD8R: begin rcnt<=0; nread<=10'd4; streamf<=0; ret<=S_C55; st<=SUB_READ; end
                S_C55:   begin setcmd(8'h77,8'h00,8'h00,8'h00,8'h00,8'h01); ret<=S_C55P; st<=SUB_SEND; end
                S_C55P:  begin pollc<=0; ret<=S_C41; st<=SUB_POLL; end
                S_C41:   begin setcmd(8'h69,8'h40,8'h00,8'h00,8'h00,8'h01); ret<=S_C41P; st<=SUB_SEND; end
                S_C41P:  begin pollc<=0; ret<=S_C41CHK; st<=SUB_POLL; end
                S_C41CHK: st <= (r1==8'h00) ? S_C58 : S_C55;       // loop until ready
                S_C58:   begin setcmd(8'h7A,8'h00,8'h00,8'h00,8'h00,8'h01); ret<=S_C58P; st<=SUB_SEND; end
                S_C58P:  begin pollc<=0; ret<=S_C58R; st<=SUB_POLL; end
                S_C58R:  begin rcnt<=0; nread<=10'd4; streamf<=0; ret<=S_READY; st<=SUB_READ; end
                S_READY: begin ready<=1; rdretry<=4'd0; st<=S_RD; end

                // ---- block read ----
                S_RD:    begin setcmd(8'h51, rd_blk[31:24], rd_blk[23:16], rd_blk[15:8], rd_blk[7:0], 8'h01); ret<=S_RDP; st<=SUB_SEND; end
                S_RDP:   begin pollc<=0; ret<=S_RDTOK; st<=SUB_POLL; end
                S_RDTOK: begin pollc<=0; ret<=S_RDDATA; st<=SUB_TOK; end
                // read 512 data bytes (stream out + CRC16), then 2 CRC bytes, verify
                S_RDDATA: begin dcnt<=0; crc16<=16'h0000; st<=S_DBYTE; end
                S_DBYTE:  begin eng_tx<=8'hFF; eng_go<=1; st<=S_DBYTE_W; end
                S_DBYTE_W: if (eng_done) begin
                    out_data<=eng_rx; out_we<=1; crc16<=crc16_upd(crc16, eng_rx);
                    if (dcnt==10'd511) begin ccnt<=1'b0; st<=S_CBYTE; end
                    else begin dcnt<=dcnt+10'd1; st<=S_DBYTE; end
                end
                S_CBYTE:  begin eng_tx<=8'hFF; eng_go<=1; st<=S_CBYTE_W; end
                S_CBYTE_W: if (eng_done) begin
                    if (ccnt==1'b0) begin rx_crc[15:8]<=eng_rx; ccnt<=1'b1; st<=S_CBYTE; end
                    else if (crc16 == {rx_crc[15:8], eng_rx}) st<=S_DONE;        // block CRC OK
                    else if (rdretry != 5'd31) begin rdretry<=rdretry+5'd1; crc_err<=1'b1; st<=S_RGAP; end // re-read
                    else st<=S_FAIL;
                end
                S_DONE:  begin done<=1; st<=S_RIDLE; end               // keep CS low for more reads
                S_RIDLE: if (rd_start) begin pollc<=0; rdretry<=4'd0; st<=S_RGAP; end   // next block, no re-init
                         else if (start) begin cs_n<=1; pcnt<=0; st<=S_PWR; end  // re-init
                // 8 dummy clocks (Nrc) so the card is ready for the next CMD17
                S_RGAP:  begin eng_tx<=8'hFF; eng_go<=1; st<=S_RGAPW; end
                S_RGAPW: if (eng_done) st<=S_RD;
                S_FAIL:  begin cs_n<=1; fail<=1; st<=S_IDLE; end
            endcase
        end
    end
endmodule
