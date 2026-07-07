`timescale 1ns/1ps
// =============================================================================
// link_send - stream an L-element activation vector over ss_link (sender side)
//
// On `start`, pushes vec[0..L-1] byte-by-byte into the link's TX port, honoring
// tx_full backpressure (atomic: the write commit and the index advance use the
// same tx_full at the same edge -> no byte loss/dup). This is the node-side
// framing that turns the raw link into "send one activation vector".
// =============================================================================
module link_send #(
    parameter W = 8,
    parameter L = 64
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start,
    input  wire [W*L-1:0] vec,        // flat vector to transmit
    output wire           busy,
    // ss_link TX interface
    output wire [W-1:0]   tx_data,
    output wire           tx_send,
    input  wire           tx_full
);
    localparam CW = (L <= 1) ? 1 : $clog2(L);
    reg            active;
    reg [CW:0]     idx;

    assign busy    = active;
    assign tx_send = active;                  // request a write while active
    assign tx_data = vec[idx*W +: W];         // current byte (combinational)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            idx    <= 0;
        end else if (!active) begin
            if (start) begin active <= 1'b1; idx <= 0; end
        end else if (!tx_full) begin          // a byte commits this edge
            if (idx == L-1) begin active <= 1'b0; idx <= 0; end
            else                idx <= idx + 1'b1;
        end
    end
endmodule
