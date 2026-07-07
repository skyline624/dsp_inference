`timescale 1ns/1ps
// =============================================================================
// sd_boot - autonomous model loader : SD card -> SDRAM (init once, N blocks)
//
// Drives sd_ctrl to initialise the card once, then stream NBLK consecutive
// blocks, writing every byte into SDRAM through a simple write port (mem_wr /
// mem_addr / mem_din, honoring the SDRAM controller's `sdram_busy`). Meant to be
// wired into top.v: during boot it owns the SDRAM (via a mux), then `done` lets
// the normal command FSM take over for inference. The SD image is a linear dump
// of the SDRAM weight layout, so block b lands at SDRAM byte b*512.
// =============================================================================
module sd_boot #(parameter HALF = 34, parameter [15:0] NBLK = 16'd2) (
    input  wire clk, rstn, start,
    // SD SPI pins
    output wire sd_clk, sd_cmd, input wire sd_dat0, output wire sd_dat3,
    // SDRAM write port
    output reg [22:0] mem_addr, output reg [7:0] mem_din, output reg mem_wr,
    input  wire sdram_busy,
    output reg  done,
    output wire [15:0] dbg_blk,       // current block index (boot progress)
    output wire dbg_ready, dbg_fail, dbg_wpend, drained,
    output reg [19:0] dbg_nwe, dbg_nwr     // SD bytes received / SDRAM writes issued (debug)
);
    reg  ctrl_start, ctrl_rd; reg [31:0] rd_blk;
    wire ctrl_ready, ctrl_done, ctrl_fail, ctrl_crc_err, c_we; wire [7:0] c_data;
    sd_ctrl #(.HALF(HALF)) u_ctrl (
        .clk(clk), .rstn(rstn), .start(ctrl_start), .rd_start(ctrl_rd), .rd_blk(rd_blk),
        .cs_n(sd_dat3), .sclk(sd_clk), .mosi(sd_cmd), .miso(sd_dat0),
        .out_we(c_we), .out_data(c_data), .ready(ctrl_ready), .done(ctrl_done),
        .fail(ctrl_fail), .crc_err(ctrl_crc_err));
    assign dbg_ready = ctrl_ready;
    assign dbg_fail  = ctrl_fail;

    reg [7:0] wbyte; reg wpend; reg [22:0] waddr; reg blkdone; reg [15:0] blk;
    assign dbg_blk = blk;
    assign dbg_wpend = wpend;
    // FIFO decoupling the SD byte stream from SDRAM writes/refresh. Depth 32
    // absorbs the refresh burst; refresh is only requested when the FIFO is
    // empty (drained), so a refresh never catches a partially-full FIFO.
    reg [7:0] fbuf [0:31]; reg [4:0] fhead, ftail; reg [5:0] fcnt;
    wire fifo_full = (fcnt == 6'd32);
    wire fifo_empty = (fcnt == 6'd0);
    assign drained = fifo_empty && !wpend;     // write path idle -> safe to refresh
    localparam B_IDLE=0, B_KICK=1, B_LOAD=2, B_DONE=3;
    reg [1:0] bst;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            bst<=B_IDLE; ctrl_start<=0; ctrl_rd<=0; mem_wr<=0; wpend<=0; blkdone<=0;
            waddr<=0; blk<=0; done<=0; rd_blk<=32'd0; fhead<=0; ftail<=0; fcnt<=0; dbg_nwe<=0; dbg_nwr<=0;
        end else begin
            ctrl_start<=0; ctrl_rd<=0; mem_wr<=0;
            case (bst)
                B_IDLE: if (start) begin ctrl_start<=1; blk<=0; rd_blk<=32'd0; waddr<=0; wpend<=0; blkdone<=0; fhead<=0; ftail<=0; fcnt<=0; bst<=B_LOAD; end
                B_LOAD: if (ctrl_fail) begin           // init/read failed -> retry from block 0
                    ctrl_start<=1; blk<=0; rd_blk<=32'd0; waddr<=0; wpend<=0; blkdone<=0; fhead<=0; ftail<=0; fcnt<=0;
                end else begin
                    if (ctrl_crc_err) begin waddr <= {blk[13:0], 9'b0}; wpend<=0; fhead<=0; ftail<=0; fcnt<=0; end  // bad CRC: rewind + flush FIFO
                    // push SD byte into FIFO (depth 8 absorbs refresh bursts)
                    if (c_we) dbg_nwe<=dbg_nwe+20'd1;            // SD bytes emitted by the controller
                    if (c_we && !fifo_full) begin fbuf[fhead]<=c_data; fhead<=fhead+5'd1; fcnt<=fcnt+6'd1; end
                    if (ctrl_done) blkdone<=1;
                    // pop FIFO -> SDRAM write (wpend guards the 1-cycle mem_wr pulse)
                    if (!fifo_empty && !wpend && !sdram_busy) begin
                        wbyte<=fbuf[ftail]; wpend<=1'b1; ftail<=ftail+5'd1; fcnt<=fcnt-6'd1;
                    end
                    if (wpend && !sdram_busy) begin
                        mem_addr<=waddr; mem_din<=wbyte; mem_wr<=1; wpend<=0; waddr<=waddr+23'd1; dbg_nwr<=dbg_nwr+20'd1;
                    end
                    if (blkdone && fifo_empty && !wpend && !sdram_busy) begin       // block fully written
                        if (blk == NBLK-16'd1) begin done<=1; bst<=B_DONE; end
                        else begin blk<=blk+16'd1; rd_blk<={16'd0, blk+16'd1}; ctrl_rd<=1; blkdone<=0; end
                    end
                end
                B_DONE: done<=1;        // booting complete, SDRAM loaded
            endcase
        end
    end
endmodule
