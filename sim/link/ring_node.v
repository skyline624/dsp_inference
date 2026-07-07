`timescale 1ns/1ps
// =============================================================================
// ring_node - one node of a SCALABLE ring all-gather (any N nodes)
//
// Topology: a ring. Each node has ONE outgoing link (to node i+1) and ONE
// incoming link (from node i-1) -> pin count per node is CONSTANT, independent
// of N. This is what lets the cluster grow past 2 nodes.
//
// Ring all-gather (standard, bandwidth-optimal): each node starts holding its
// own slice; over N-1 rounds it forwards the slice it just received to its
// successor and stores incoming slices. After N-1 rounds EVERY node holds the
// full vector (all N slices). No central coordinator -- each node's small FSM
// just rendezvous with its neighbors via the links.
// =============================================================================
module ring_node #(
    parameter W = 8,
    parameter S = 32,    // slice length (elements)
    parameter N = 2,     // number of nodes
    parameter ID = 0     // this node's index
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [W*S-1:0]   my_slice,
    // outgoing link (to node ID+1)
    output wire [W-1:0]     o_data,
    output wire             o_send,
    input  wire             o_full,
    // incoming link (from node ID-1)
    output wire             i_rd,
    input  wire [W-1:0]     i_data,
    input  wire             i_empty,
    // assembled full vector (all N slices, slot j = node j's slice)
    output reg  [W*S*N-1:0] full_vec,
    output reg              done
);
    localparam SB = W*S;

    reg  [7:0] cur_send;                 // which slot we transmit this round
    wire [SB-1:0] slice_tx = full_vec[cur_send*SB +: SB];

    reg        send_start;
    wire       send_busy;
    link_send #(W, S) u_tx (
        .clk(clk), .rst_n(rst_n), .start(send_start), .vec(slice_tx),
        .busy(send_busy), .tx_data(o_data), .tx_send(o_send), .tx_full(o_full));

    wire [SB-1:0] rvec;
    wire          rdone;
    link_recv #(W, S) u_rx (
        .clk(clk), .rst_n(rst_n), .rx_rd(i_rd), .rx_data(i_data),
        .rx_empty(i_empty), .vec(rvec), .done(rdone));

    // latch incoming slice (link_recv drains continuously, FSM consumes later)
    reg [SB-1:0] rhold;
    reg          rgot;

    localparam IDLE=0, SEND=1, SBUSY=2, SWAIT=3, RECV=4, NEXT=5, FIN=6;
    reg [2:0] st;
    reg [7:0] round;
    reg [7:0] recv_slot;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; done <= 0; send_start <= 0; rgot <= 0; round <= 0; cur_send <= 0;
        end else begin
            send_start <= 0;
            done       <= 0;
            if (rdone) begin rhold <= rvec; rgot <= 1; end   // capture incoming

            case (st)
                IDLE: if (start) begin
                    full_vec[ID*SB +: SB] <= my_slice;       // start with own slice
                    cur_send <= ID[7:0];
                    round    <= 0;
                    st       <= SEND;
                end

                SEND: begin                                   // transmit current slot
                    send_start <= 1;
                    // recv_slot for this round r: (ID - 1 - r) mod N
                    recv_slot <= ((ID + N - 1 - round) >= N) ?
                                 (ID + N - 1 - round - N) : (ID + N - 1 - round);
                    st <= SBUSY;
                end
                SBUSY: if (send_busy) st <= SWAIT;             // wait it started
                SWAIT: if (!send_busy) st <= RECV;             // ...and finished

                RECV: if (rgot) begin                          // got neighbor's slice
                    full_vec[recv_slot*SB +: SB] <= rhold;
                    rgot <= 0;
                    st   <= NEXT;
                end

                NEXT: begin
                    cur_send <= recv_slot;                     // forward it next round
                    if (round == N-2) begin done <= 1; st <= FIN; end
                    else begin round <= round + 1; st <= SEND; end
                end

                FIN: st <= IDLE;
            endcase
        end
    end
endmodule
