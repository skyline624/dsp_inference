`timescale 1ns/1ps
// =============================================================================
// uart_bridge - drop-in byte adapters that let a UART-style FSM drive the
// parallel source-synchronous link (async_fifo) instead of a serial 8N1 PHY.
//
// The FSM-facing side is IDENTICAL to uart_tx_8n1 / uart_rx_8n1 (send/busy,
// data/valid), so the command/response FSMs in top.v and ffn_tp_seq2.v do NOT
// change. Only the transport underneath does: one byte crosses in a few cycles
// (fifo write/read) instead of ~270 (10 bits x DIV). The fifo depth + full
// backpressure replace the UART's natural throttle.
//
//   tx8_link : FSM sees (data, send, busy)   -> fifo write port (o_data,o_wr,i_full)
//   rx8_link : FSM sees (data, valid)        <- fifo read  port (i_data,i_empty,o_rd)
// =============================================================================

// ---- transmit adapter : same interface as uart_tx_8n1 (data/send/busy) -------
module tx8_link #(
    parameter HOLD = 4            // cycles busy stays high after commit
) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       send,      // 1-cycle pulse : start sending 'data'
    output reg        busy,      // high until the byte is committed to the fifo
    // parallel link write port (to the receiver's async_fifo)
    output reg [7:0]  o_data,
    output reg        o_wr,      // 1-cycle write-enable pulse
    input  wire       i_full     // backpressure : fifo cannot accept a byte
);
    // States : 0 = idle, 1 = waiting for fifo room, 2 = draining HOLD cycles.
    // The HOLD tail matters : the driving FSM streams a byte, drops busy, then
    // does read-setup -> send for the next byte. A real UART's busy lasts ~270
    // cycles, which hid the node's obuf read-latency (S_TX_O_RD -> S_TX_O_W).
    // With an instant fifo write, busy must still linger a few cycles so the
    // node's registered obuf_rdata settles before the next send samples it.
    reg [1:0] ph;
    reg [3:0] hcnt;
    always @(posedge clk) begin
        o_wr <= 1'b0;
        if (rst) begin busy <= 1'b0; ph <= 2'd0; hcnt <= 4'd0; end
        else case (ph)
            2'd0: if (send) begin o_data <= data; busy <= 1'b1; ph <= 2'd1; end
            2'd1: if (!i_full) begin o_wr <= 1'b1; hcnt <= HOLD[3:0]; ph <= 2'd2; end
            default: if (hcnt == 4'd0) begin busy <= 1'b0; ph <= 2'd0; end
                     else hcnt <= hcnt - 4'd1;
        endcase
    end
endmodule

// ---- receive adapter : same interface as uart_rx_8n1 (data/valid) ------------
// Adds a 'rdy' handshake absent from a real UART : a UART is a push interface
// (bytes arrive ~270 cycles apart, the FSM always keeps up), but the parallel
// fifo can deliver a byte every 2 cycles -- far faster than a latching FSM
// (top.v's rx_pending, cleared only on rx_consume) can drain. So we only pull &
// pulse valid when the consumer asserts rdy, preventing dropped bytes.
module rx8_link (
    input  wire       clk,
    input  wire       rst,
    output reg  [7:0] data,
    output reg        valid,     // 1-cycle pulse when 'data' holds a fresh byte
    input  wire       rdy,       // consumer can accept a byte this cycle
    // parallel link read port (from this node's async_fifo)
    input  wire [7:0] i_data,
    input  wire       i_empty,
    output reg        o_rd       // 1-cycle read-enable pulse (advance the fifo)
);
    // One-byte-per-acknowledge handshake. A single 'rdy' level is not enough : if
    // rdy stays high while the consumer takes >1 cycle to register a byte, this
    // adapter would deliver a second byte the consumer never latched (dropped).
    // So we require rdy to have gone low (byte accepted) and come back high before
    // delivering the next : we track the previous rdy and fire only on its rising
    // edge, guaranteeing exactly one (data,valid) per consumer acknowledge.
    reg armed;                    // ready to deliver the next byte
    always @(posedge clk) begin
        valid <= 1'b0;
        o_rd  <= 1'b0;
        if (rst) begin
            armed <= 1'b1;
        end else if (armed && rdy && !i_empty && !o_rd) begin
            data  <= i_data;
            valid <= 1'b1;
            o_rd  <= 1'b1;
            armed <= 1'b0;        // wait for the consumer to drop rdy (accept) ...
        end else if (!rdy) begin
            armed <= 1'b1;        // ... then re-arm when it is ready again
        end
    end
endmodule
