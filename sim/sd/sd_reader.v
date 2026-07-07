`timescale 1ns/1ps
// =============================================================================
// sd_reader - SD card controller (SPI master), multi-block read (synthesizable)
//
// This is the real RTL that would go on the FPGA. It reads NBLK consecutive
// 512-byte blocks from the SD card (CMD17 per block) and emits the bytes as a
// stream (out_we/out_data) - destined to fill an SDRAM region (the layer cache).
// SPI mode 0, sclk = clk/(2*HALF). Init handshake omitted (added for real HW).
// =============================================================================
module sd_reader #(parameter HALF = 2) (
    input  wire clk, rstn, start,
    input  wire [15:0] blk_start, nblk,
    output reg  cs_n, output reg sclk, output reg mosi,
    input  wire miso,
    output reg  out_we, output reg [7:0] out_data,
    output reg  done
);
    // ---- 8-bit SPI transfer engine ----
    reg eng_go, eng_done; reg [7:0] eng_tx, eng_rx;
    reg eng_busy; reg [3:0] bitn; reg [7:0] txs, rxs; reg [15:0] dv;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin eng_busy<=0; eng_done<=0; sclk<=0; mosi<=1; end
        else begin
            eng_done<=0;
            if (!eng_busy) begin
                if (eng_go) begin eng_busy<=1; txs<=eng_tx; rxs<=8'h00; bitn<=0; dv<=0; sclk<=0; mosi<=eng_tx[7]; end
            end else if (dv==HALF-1) begin
                dv<=0;
                if (sclk==1'b0) begin sclk<=1'b1; rxs<={rxs[6:0],miso}; end       // rising: sample
                else begin
                    sclk<=1'b0;                                                   // falling
                    if (bitn==4'd7) begin eng_busy<=0; eng_done<=1; eng_rx<=rxs; end
                    else begin bitn<=bitn+4'd1; txs<={txs[6:0],1'b0}; mosi<=txs[6]; end
                end
            end else dv<=dv+16'd1;
        end
    end

    // ---- command / read FSM ----
    localparam S_IDLE=0,S_CMD=1,S_CMDW=2,S_R1=3,S_R1W=4,S_TOK=5,S_TOKW=6,
               S_DAT=7,S_DATW=8,S_CRC=9,S_CRCW=10,S_DONE=11;
    reg [3:0] st; reg [2:0] cmdi; reg [15:0] dcnt, blk;

    function [7:0] cbyte; input [2:0] i; input [15:0] b;
        case (i) 3'd0:cbyte=8'h51; 3'd1:cbyte=8'h00; 3'd2:cbyte=8'h00;
                 3'd3:cbyte=b[15:8]; 3'd4:cbyte=b[7:0]; default:cbyte=8'h01; endcase
    endfunction

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin st<=S_IDLE; eng_go<=0; cs_n<=1; done<=0; out_we<=0; blk<=0; cmdi<=0; dcnt<=0; end
        else begin
            eng_go<=0; out_we<=0; done<=0;
            case (st)
                S_IDLE:  if (start) begin cs_n<=0; blk<=blk_start; cmdi<=0; st<=S_CMD; end
                S_CMD:   begin eng_tx<=cbyte(cmdi,blk); eng_go<=1; st<=S_CMDW; end
                S_CMDW:  if (eng_done) begin if (cmdi==3'd5) st<=S_R1; else begin cmdi<=cmdi+3'd1; st<=S_CMD; end end
                S_R1:    begin eng_tx<=8'hFF; eng_go<=1; st<=S_R1W; end
                S_R1W:   if (eng_done) st <= (eng_rx==8'h00) ? S_TOK : S_R1;
                S_TOK:   begin eng_tx<=8'hFF; eng_go<=1; st<=S_TOKW; end
                S_TOKW:  if (eng_done) begin if (eng_rx==8'hFE) begin dcnt<=0; st<=S_DAT; end else st<=S_TOK; end
                S_DAT:   begin eng_tx<=8'hFF; eng_go<=1; st<=S_DATW; end
                S_DATW:  if (eng_done) begin out_data<=eng_rx; out_we<=1;
                             if (dcnt==16'd511) begin dcnt<=0; st<=S_CRC; end else begin dcnt<=dcnt+16'd1; st<=S_DAT; end end
                S_CRC:   begin eng_tx<=8'hFF; eng_go<=1; st<=S_CRCW; end
                S_CRCW:  if (eng_done) begin
                             if (dcnt==16'd1) begin
                                 if (blk==blk_start+nblk-16'd1) begin cs_n<=1; st<=S_DONE; end
                                 else begin blk<=blk+16'd1; cmdi<=0; st<=S_CMD; end
                             end else begin dcnt<=dcnt+16'd1; st<=S_CRC; end
                         end
                S_DONE:  begin done<=1; st<=S_IDLE; end
            endcase
        end
    end
endmodule
