`timescale 1ns/1ps
// =============================================================================
// seq - on-chip command sequencer (mini-GG) with op chaining + handoff
//
// Holds the "current activation" curx[D] on-chip. For each op in the program it
// issues a command to the (unmodified) brick node over UART, captures the
// response, and writes the result back into curx -> the output of one op becomes
// the INPUT of the next, with no host in the loop (the FFN handoff mechanism).
//
// inc.1 : N_OPS=1 (one SS command).
// inc.2 : N_OPS>1 (chain, e.g. SS->SS) -> proves the obuf->next-input handoff.
// The full FFN is the same loop with a varied program (FN,FQ,SS,...).
// =============================================================================
module seq #(
    parameter W = 8,
    parameter D = 64,
    parameter DIV  = 27,
    parameter BOOT = 40000,
    parameter N_OPS = 1
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire [W*D-1:0]  x_in,
    input  wire signed [7:0] sx,
    output wire            node_rx,
    input  wire            node_tx,
    output reg  [W*D-1:0]  result,
    output reg             done
);
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    uart_tx_8n1 #(.DIV(DIV)) u_htx (.clk(clk), .rst(~rst_n), .data(tx_data),
                                    .send(tx_send), .tx(node_rx), .busy(tx_busy));
    wire [7:0] rx_data; wire rx_valid;
    uart_rx_8n1 #(.DIV(DIV)) u_hrx (.clk(clk), .rst(~rst_n), .rx(node_tx),
                                    .data(rx_data), .valid(rx_valid));

    localparam PKT = 3 + D;       // SS : 'S' 'S' shift x[64]
    localparam RSP = 6 + D;       // SK : 'S' 'K' sh idx lo hi out[64]

    reg [W*D-1:0]    curx;        // current activation (carried op-to-op)
    reg signed [7:0] cursh;
    reg [3:0]        op_i;

    reg [9:0] idx;
    wire [7:0] emit_byte =
        (idx == 0) ? "S" :
        (idx == 1) ? "S" :
        (idx == 2) ? cursh[7:0] :
                     curx[(idx-3)*W +: W];

    reg [7:0] resp [0:RSP-1];
    reg [9:0] rcnt;
    integer i;

    localparam IDLE=0, BWAIT=1, E_SET=2, E_BUSY=3, E_DONE=4, RECV=5, NEXT=6, EMIT_OUT=7, FIN=8;
    reg [3:0] st;
    reg [15:0] bcnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; idx<=0; rcnt<=0; bcnt<=0; op_i<=0;
        end else begin
            tx_send<=0; done<=0;
            if (st==RECV && rx_valid) begin resp[rcnt] <= rx_data; rcnt <= rcnt + 1; end

            case (st)
                IDLE: if (start) begin
                    curx <= x_in; cursh <= sx; op_i <= 0; bcnt <= 0; st <= BWAIT;
                end
                BWAIT: if (bcnt==BOOT) begin idx<=0; st<=E_SET; end else bcnt<=bcnt+1;

                E_SET:  begin tx_data <= emit_byte; tx_send <= 1; st<=E_BUSY; end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx == PKT-1) begin rcnt<=0; st<=RECV; end
                    else begin idx<=idx+1; st<=E_SET; end
                end

                RECV: if (rcnt == RSP) begin
                    for (i=0;i<D;i=i+1) curx[i*W +: W] <= resp[6+i];   // handoff: out -> curx
                    cursh <= resp[2];                                   // and its shift
                    st <= NEXT;
                end
                NEXT: begin
                    op_i <= op_i + 1;
                    if (op_i == N_OPS-1) st <= EMIT_OUT;
                    else begin idx<=0; st<=E_SET; end                   // next op on curx
                end

                EMIT_OUT: begin result <= curx; done<=1; st<=FIN; end
                FIN: st<=IDLE;
            endcase
        end
    end
endmodule
