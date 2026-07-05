`timescale 1ns/1ps
// Loopback bench for the UART->parallel-link adapters (Phase 1 transport gate).
// FSM-facing sender (data/send/busy) -> tx8_link -> ss_link (async_fifo) ->
// rx8_link -> FSM-facing receiver (data/valid). Two independent clocks exercise
// the CDC. cocotb pushes bytes and checks they come out in order, unchanged.
module uart_bridge_tb (
    input  wire       clk_tx,     // sender domain
    input  wire       clk_rx,     // receiver domain
    input  wire       rst,
    // FSM-facing transmit port (same as uart_tx_8n1)
    input  wire [7:0] tx_byte,
    input  wire       tx_send,
    output wire       tx_busy,
    // FSM-facing receive port (same as uart_rx_8n1)
    output wire [7:0] rx_byte,
    output wire       rx_valid
);
    wire       rst_n = ~rst;
    wire [7:0] w_data;  wire w_wr;  wire w_full;
    wire [7:0] r_data;  wire r_empty; wire r_rd;

    tx8_link u_tx (
        .clk(clk_tx), .rst(rst), .data(tx_byte), .send(tx_send), .busy(tx_busy),
        .o_data(w_data), .o_wr(w_wr), .i_full(w_full));

    ss_link #(.DW(8), .AW(4)) u_link (
        .clk_tx(clk_tx), .rst_tx_n(rst_n), .tx_data(w_data), .tx_send(w_wr), .tx_full(w_full),
        .clk_rx(clk_rx), .rst_rx_n(rst_n), .rx_rd(r_rd), .rx_data(r_data), .rx_empty(r_empty));

    // emulate a consumer that acknowledges each byte : rdy drops on valid (byte
    // accepted) and comes back high next cycle, giving rx8_link the rising edge it
    // needs to release the following byte (mirrors top.v's rx_pending/rx_consume).
    reg rx_rdy = 1'b1;
    always @(posedge clk_rx) rx_rdy <= rst ? 1'b1 : ~rx_valid;
    rx8_link u_rx (
        .clk(clk_rx), .rst(rst), .data(rx_byte), .valid(rx_valid), .rdy(rx_rdy),
        .i_data(r_data), .i_empty(r_empty), .o_rd(r_rd));
endmodule
