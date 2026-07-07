`timescale 1ns/1ps
// =============================================================================
// async_fifo - dual-clock (asynchronous) FIFO, Gray-pointer CDC
//
// Standard Cliff-Cummings style async FIFO: binary+Gray pointers, 2-flop
// synchronizers crossing each pointer into the opposite clock domain, registered
// full/empty. This is the safe primitive for moving data between two FPGAs that
// run on independent oscillators (no shared clock).
//
// In the source-synchronous inter-FPGA link, the WRITE side is clocked by the
// sender's FORWARDED clock (arriving on a pin), the READ side by the receiver's
// own clock. The whole FIFO physically lives on the receiver chip.
// =============================================================================
module async_fifo #(
    parameter DW = 8,        // data width (one activation byte)
    parameter AW = 4,        // address width -> depth = 2^AW
    parameter AFMARGIN = 4   // almost_full asserts this many slots before full
) (
    // write domain (= sender's forwarded clock)
    input  wire          wclk,
    input  wire          wrst_n,
    input  wire          winc,
    input  wire [DW-1:0] wdata,
    output reg           wfull,
    output wire          walmost_full,   // early backpressure (round-trip safe)
    // read domain (= receiver's local clock)
    input  wire          rclk,
    input  wire          rrst_n,
    input  wire          rinc,
    output wire [DW-1:0] rdata,
    output reg           rempty
);
    localparam DEPTH = (1 << AW);

    // gray -> binary (to estimate occupancy in the write domain)
    function [AW:0] gray2bin;
        input [AW:0] g;
        integer i;
        begin
            gray2bin[AW] = g[AW];
            for (i = AW-1; i >= 0; i = i - 1)
                gray2bin[i] = gray2bin[i+1] ^ g[i];
        end
    endfunction

    reg [DW-1:0] mem [0:DEPTH-1];

    reg  [AW:0] wbin, wptr;          // write binary + gray
    reg  [AW:0] rbin, rptr;          // read  binary + gray
    reg  [AW:0] wq1_rptr, wq2_rptr;  // read gray  synced into write domain
    reg  [AW:0] rq1_wptr, rq2_wptr;  // write gray synced into read domain

    // ---- memory ----
    wire [AW-1:0] waddr = wbin[AW-1:0];
    wire [AW-1:0] raddr = rbin[AW-1:0];
    always @(posedge wclk)
        if (winc && !wfull) mem[waddr] <= wdata;
    assign rdata = mem[raddr];       // current front word (valid when !rempty)

    // ---- pointer synchronizers (the CDC) ----
    always @(posedge wclk or negedge wrst_n)
        if (!wrst_n) {wq2_rptr, wq1_rptr} <= 0;
        else         {wq2_rptr, wq1_rptr} <= {wq1_rptr, rptr};
    always @(posedge rclk or negedge rrst_n)
        if (!rrst_n) {rq2_wptr, rq1_wptr} <= 0;
        else         {rq2_wptr, rq1_wptr} <= {rq1_wptr, wptr};

    // ---- write pointer + full ----
    wire [AW:0] wbnext = wbin + (winc & ~wfull);
    wire [AW:0] wgnext = (wbnext >> 1) ^ wbnext;
    always @(posedge wclk or negedge wrst_n)
        if (!wrst_n) {wbin, wptr} <= 0;
        else         {wbin, wptr} <= {wbnext, wgnext};
    // full when next write gray == read gray with the two MSBs inverted
    wire wfull_val = (wgnext == {~wq2_rptr[AW:AW-1], wq2_rptr[AW-2:0]});
    always @(posedge wclk or negedge wrst_n)
        if (!wrst_n) wfull <= 1'b0;
        else         wfull <= wfull_val;

    // almost_full : occupancy estimate in the write domain (gray read ptr -> bin)
    wire [AW:0] rbin_in_w   = gray2bin(wq2_rptr);
    wire [AW:0] w_occupancy = wbin - rbin_in_w;
    assign walmost_full = (w_occupancy >= (DEPTH - AFMARGIN));

    // ---- read pointer + empty ----
    wire [AW:0] rbnext = rbin + (rinc & ~rempty);
    wire [AW:0] rgnext = (rbnext >> 1) ^ rbnext;
    always @(posedge rclk or negedge rrst_n)
        if (!rrst_n) {rbin, rptr} <= 0;
        else         {rbin, rptr} <= {rbnext, rgnext};
    wire rempty_val = (rgnext == rq2_wptr);
    always @(posedge rclk or negedge rrst_n)
        if (!rrst_n) rempty <= 1'b1;
        else         rempty <= rempty_val;
endmodule
