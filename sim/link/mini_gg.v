`timescale 1ns/1ps
// =============================================================================
// mini_gg - per-node autonomous sequencer ("GG" reborn at cluster scale)
//
// The monolithic GG (whole layer in one chip) didn't route. A per-node mini-GG
// sequences only THIS node's slice: one small FSM chains, with NO external
// per-op command, the steps:
//     RECV  (pull an activation vector off the link)
//   -> COMP1 (chained op 1 : scale)
//   -> COMP2 (chained op 2 : bias + saturate)
//   -> SEND  (push the result onto the link)  -> back to RECV
//
// Link framing (link_recv/link_send) is internal -> the node is self-contained:
// give it a link in and a link out and it runs autonomously (zero-PC). Small =
// routes, unlike the monolithic GG. The compute here is a placeholder 2-stage
// chain standing in for the node's real slice ops (matmul/silu/...).
// =============================================================================
module mini_gg #(
    parameter W = 8,
    parameter L = 64
) (
    input  wire         clk,
    input  wire         rst_n,
    // link IN (from peer)
    output wire         rx_rd,
    input  wire [W-1:0] rx_data,
    input  wire         rx_empty,
    // link OUT (to peer)
    output wire [W-1:0] tx_data,
    output wire         tx_send,
    input  wire         tx_full,
    output reg  [2:0]   dbg_state
);
    wire [W*L-1:0]   rvec;
    wire             rdone;
    reg  [16*L-1:0]  tvec;       // stage-1 result (wide)
    reg  [W*L-1:0]   yvec;       // stage-2 result (sent)
    reg              send_start;
    wire             send_busy;

    link_recv #(W, L) u_rx (
        .clk(clk), .rst_n(rst_n), .rx_rd(rx_rd), .rx_data(rx_data),
        .rx_empty(rx_empty), .vec(rvec), .done(rdone));
    link_send #(W, L) u_tx (
        .clk(clk), .rst_n(rst_n), .start(send_start), .vec(yvec),
        .busy(send_busy), .tx_data(tx_data), .tx_send(tx_send), .tx_full(tx_full));

    function signed [7:0] sat8;
        input signed [15:0] t;
        sat8 = (t > 16'sd127)  ? 8'sd127  :
               (t < -16'sd128) ? -8'sd128 : t[7:0];
    endfunction

    localparam IDLE=3'd0, COMP1=3'd1, COMP2=3'd2, SEND=3'd3, WAITDONE=3'd4;
    reg [2:0] st;
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; send_start <= 1'b0; dbg_state <= 0;
        end else begin
            send_start <= 1'b0;
            case (st)
                IDLE: if (rdone) begin                       // vector received
                    for (i = 0; i < L; i = i + 1)            // chained op 1 : scale x2 + 1
                        tvec[i*16 +: 16] <= $signed(rvec[i*W +: W]) * 16'sd2 + 16'sd1;
                    st <= COMP1;
                end
                COMP1: begin                                 // chained op 2 : saturate to int8
                    for (i = 0; i < L; i = i + 1)
                        yvec[i*W +: W] <= sat8($signed(tvec[i*16 +: 16]));
                    st <= COMP2;
                end
                COMP2:    begin send_start <= 1'b1; st <= SEND; end
                SEND:     if (send_busy)  st <= WAITDONE;     // transmission started
                WAITDONE: if (!send_busy) st <= IDLE;         // transmission finished
            endcase
            dbg_state <= st;
        end
    end
endmodule
