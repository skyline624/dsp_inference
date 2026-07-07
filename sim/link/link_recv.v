`timescale 1ns/1ps
// =============================================================================
// link_recv - collect an L-element activation vector from ss_link (receiver side)
//
// Drains the link's RX FIFO into a flat vector register; pulses `done` for one
// clock when L elements have arrived. Receiver-side framing in the receiver's
// own clock domain (the CDC already handled by ss_link's async FIFO).
// =============================================================================
module link_recv #(
    parameter W = 8,
    parameter L = 64
) (
    input  wire           clk,
    input  wire           rst_n,
    // ss_link RX interface
    output wire           rx_rd,
    input  wire [W-1:0]   rx_data,
    input  wire           rx_empty,
    // collected vector
    output reg  [W*L-1:0] vec,
    output reg            done
);
    localparam CW = (L <= 1) ? 1 : $clog2(L);
    reg [CW:0] idx;

    assign rx_rd = ~rx_empty;                 // drain whenever data is available

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idx  <= 0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            if (!rx_empty) begin              // a front word is read this edge
                vec[idx*W +: W] <= rx_data;
                if (idx == L-1) begin idx <= 0; done <= 1'b1; end
                else                idx <= idx + 1'b1;
            end
        end
    end
endmodule
