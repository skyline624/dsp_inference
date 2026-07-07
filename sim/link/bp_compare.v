`timescale 1ns/1ps
// =============================================================================
// bp_compare - why almost_full matters with a real inter-chip round-trip delay
//
// On real hardware the backpressure signal travels sender<-receiver over a pin,
// so the sender sees it DELAYED. A free-running sender that throttles on the
// (delayed) plain `full` keeps writing into an already-full FIFO during the blind
// window -> data loss. `almost_full` asserts AFMARGIN slots early, absorbing the
// round-trip latency -> no loss.
//
// Two identical FIFOs run side by side under the same delayed backpressure:
//   plain : backpressure = wfull         -> loses bytes
//   afull : backpressure = walmost_full  -> lossless (AFMARGIN > DLY)
// =============================================================================
module bp_demo #(
    parameter AW = 4,
    parameter DLY = 3,        // backpressure round-trip delay (cycles)
    parameter AFMARGIN = 4,
    parameter USE_AFULL = 1
) (
    input  wire        wclk, wrst_n,
    input  wire        rclk, rrst_n,
    input  wire        rd_strobe,
    output reg [31:0]  wrote,
    output reg [31:0]  lost
);
    wire        wfull, walmost_full, rempty;
    wire [7:0]  rdata;
    reg  [7:0]  wcnt;

    wire bp_raw = USE_AFULL ? walmost_full : wfull;
    reg  [DLY-1:0] bp_pipe;
    always @(posedge wclk or negedge wrst_n)
        if (!wrst_n) bp_pipe <= 0;
        else         bp_pipe <= {bp_pipe[DLY-2:0], bp_raw};
    wire bp_seen    = bp_pipe[DLY-1];
    wire wr_attempt = ~bp_seen;          // free-running writer, throttled by delayed bp

    async_fifo #(.DW(8), .AW(AW), .AFMARGIN(AFMARGIN)) fifo (
        .wclk(wclk), .wrst_n(wrst_n), .winc(wr_attempt), .wdata(wcnt),
        .wfull(wfull), .walmost_full(walmost_full),
        .rclk(rclk), .rrst_n(rrst_n), .rinc(rd_strobe & ~rempty),
        .rdata(rdata), .rempty(rempty));

    always @(posedge wclk or negedge wrst_n)
        if (!wrst_n) begin wrote <= 0; lost <= 0; wcnt <= 0; end
        else if (wr_attempt) begin
            wcnt <= wcnt + 8'd1;
            if (wfull) lost  <= lost  + 32'd1;   // wanted to write, FIFO actually full -> lost
            else       wrote <= wrote + 32'd1;
        end
endmodule


module bp_compare #(
    parameter AW = 4,
    parameter DLY = 3,
    parameter AFMARGIN = 4
) (
    input  wire        wclk, wrst_n,
    input  wire        rclk, rrst_n,
    input  wire        rd_strobe,
    output wire [31:0] plain_wrote, plain_lost,
    output wire [31:0] afull_wrote, afull_lost
);
    bp_demo #(AW, DLY, AFMARGIN, 0) u_plain (
        .wclk(wclk), .wrst_n(wrst_n), .rclk(rclk), .rrst_n(rrst_n),
        .rd_strobe(rd_strobe), .wrote(plain_wrote), .lost(plain_lost));
    bp_demo #(AW, DLY, AFMARGIN, 1) u_afull (
        .wclk(wclk), .wrst_n(wrst_n), .rclk(rclk), .rrst_n(rrst_n),
        .rd_strobe(rd_strobe), .wrote(afull_wrote), .lost(afull_lost));
endmodule
