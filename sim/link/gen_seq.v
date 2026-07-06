`timescale 1ns/1ps
// =============================================================================
// gen_seq - autonomous GENERATION sequencer (mono-card, NN=1), LUT-lean.
//
// Forked from ffn_tp_seq2 (register-file vfile, the optimised -49% LUT style) and
// extended step by step into the full generation orchestrator (embed -> 5 causal
// layers -> lm_head -> argmax -> token, x N). Built INCREMENTALLY, one validated
// gate per step (see docs/PLAN_GEN_SEQUENCER.md).
//
//   STEP A (this file) : embedding only. On start, send EE(cur_tok) to the node,
//   write the returned x[64] into vfile[XB], expose it for the gate. The vfile
//   read/write plumbing and the ss_link byte adapters are the exact, validated
//   ones from ffn_tp_seq2 (LUT-lean, no 512-bit mux).
//
// Slot map here : only XB is used at step A. More slots added in later steps.
// =============================================================================
module gen_seq #(
    parameter W = 8,
    parameter D = 64,
    parameter BOOT = 40000,
    parameter A_EMB = 23'h000000     // tok_emb base (embedding table)
) (
    input  wire            clk, rst_n, start,
    input  wire [9:0]      cur_tok,          // token to embed
    input  wire [22:0]     base,
    // parallel link to the single node
    output wire [7:0]      lk_cmd_data,
    output wire            lk_cmd_wr,
    input  wire            lk_cmd_full,
    input  wire [7:0]      lk_resp_data,
    input  wire            lk_resp_empty,
    output wire            lk_resp_rd,
    // step-A observation port : the embedded vector, streamed out on done
    output reg  [W*D-1:0]  xb_out,
    output reg             done
);
    localparam NSLOTS = 4;               // XB, XNB, OUTV, YV (only XB used at step A)
    localparam SW = 2;
    localparam AW = SW + 6;
    localparam [SW-1:0] SB_XB=0;

    // ---- register file (LUT-RAM), byte-wide single port ----
    reg [7:0] vfile [0:NSLOTS*D-1];
    reg [AW-1:0] vaddr; reg [7:0] vdin; reg vwe;
    function [AW-1:0] vidx; input [SW-1:0] slot; input [5:0] b; vidx = (slot<<6)|b; endfunction

    // ---- ss_link byte adapters (validated pattern) ----
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    wire [7:0] rx_data; wire rx_valid; wire rx_rdy;
    tx8_link u_tx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                   .busy(tx_busy), .o_data(lk_cmd_data), .o_wr(lk_cmd_wr), .i_full(lk_cmd_full));
    rx8_link u_rx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                   .rdy(rx_rdy), .i_data(lk_resp_data), .i_empty(lk_resp_empty), .o_rd(lk_resp_rd));

    reg [7:0] pkt [0:15];
    reg [9:0] pkt_len, resp_len, idx, rcnt;
    reg [6:0] ldi;
    reg [23:0] bcnt;

    // phases (grows over the steps). Step A : just EMB.
    localparam [3:0] PH_EMB=0;
    reg [3:0] phase;

    localparam IDLE=0, BWAIT=1, BUILD=2, E_SET=3, E_BUSY=4, E_DONE=5, RECV=6,
               ADV=7, DONE_ST=8;
    reg [3:0] st;

    // one-byte-per-ack rx handshake (rx8_link fires on rising edge of rdy)
    reg rx_ack;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rx_ack <= 1'b1; else rx_ack <= ~rx_valid;
    assign rx_rdy = (st == RECV) & rx_ack;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; idx<=0; rcnt<=0; bcnt<=0; phase<=PH_EMB; ldi<=0; vwe<=0;
        end else begin
            tx_send<=0; done<=0; vwe<=0;
            if (vwe) vfile[vaddr] <= vdin;   // single write-port commit

            case (st)
                IDLE: if (start) begin phase<=PH_EMB; bcnt<=0; ldi<=0; st<=BWAIT; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    // EE : 'E''E' tok_lo tok_hi  (4 bytes) -> 'E''K' x[64]  (66 bytes)
                    pkt[0]<="E"; pkt[1]<="E"; pkt[2]<=cur_tok[7:0]; pkt[3]<={6'd0,cur_tok[9:8]};
                    pkt_len<=4; resp_len<=66;
                    idx<=0; st<=E_SET;
                end

                E_SET: begin tx_data<=pkt[idx]; tx_send<=1; st<=E_BUSY; end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin idx<=idx+1; st<=E_SET; end
                end

                // EE response : 'E''K' then x[64] at rcnt 2..65 -> vfile[XB]
                RECV: if (rx_valid) begin
                    if (rcnt>=2 && rcnt<2+D) begin vaddr<=vidx(SB_XB, rcnt[5:0]-2); vdin<=rx_data; vwe<=1'b1; end
                    if (rcnt==resp_len-1) st<=ADV; else rcnt<=rcnt+1;
                end

                ADV: begin ldi<=0; st<=DONE_ST; end

                // stream vfile[XB] out to the observation port
                DONE_ST: begin
                    xb_out[ldi*W +: W] <= vfile[vidx(SB_XB, ldi[5:0])];
                    if (ldi==D-1) begin done<=1; st<=IDLE; end else ldi<=ldi+1;
                end
            endcase
        end
    end
endmodule
