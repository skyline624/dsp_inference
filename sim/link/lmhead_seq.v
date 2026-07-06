`timescale 1ns/1ps
// =============================================================================
// lmhead_seq - autonomous generation HEAD for ONE node over ss_link.
// Takes x[64] (the post-5-layers activation) and produces the next token by:
//   1. FN(x, rms_final)                          -> xn[64]
//   2. for c in 0..NCHUNK-1 : FQ(N=64, tok_emb chunk c) -> logits[64], shift sy_c
//   3. integer re-align : right-shift each chunk to sref = max(sy_c)
//   4. argmax over the VOCAB aligned int logits -> token index
//
// tok_emb (shared classifier) sits at A_EMB, chunk c at A_EMB + c*64*64 (row-major
// [vocab,64], each chunk is 64 rows). VOCAB = NCHUNK*64. argmax is a running max
// at the common shift : a two-pass over the chunks (pass 1 finds sref, pass 2
// recomputes chunks and tracks the max) would double the FQ cost, so instead we
// keep every chunk's raw int8 logits in a small buffer (VOCAB bytes) plus its
// shift, then do the re-align + argmax in a final serial sweep.
// =============================================================================
module lmhead_seq #(
    parameter W = 8,
    parameter D = 64,
    parameter VOCAB  = 512,
    parameter NCHUNK = 8,            // VOCAB / 64
    parameter A_RMS = 23'h060000,    // rms_final address
    parameter A_EMB = 23'h000000     // tok_emb base (shared classifier)
) (
    input  wire            clk, rst_n, start,
    input  wire [W*D-1:0]  x_in,
    input  wire signed [7:0] sx_in,
    input  wire signed [7:0] sw_rms, sw_emb,
    input  wire [22:0]     base,
    output wire [7:0]      lk_cmd_data,
    output wire            lk_cmd_wr,
    input  wire            lk_cmd_full,
    input  wire [7:0]      lk_resp_data,
    input  wire            lk_resp_empty,
    output wire            lk_resp_rd,
    output reg  [9:0]      token,        // argmax index (0..VOCAB-1)
    output reg             done
);
    localparam BOOT = 40000;

    // ---- link byte adapters ----
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    wire [7:0] rx_data; wire rx_valid; wire rx_rdy;
    tx8_link u_tx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                   .busy(tx_busy), .o_data(lk_cmd_data), .o_wr(lk_cmd_wr), .i_full(lk_cmd_full));
    rx8_link u_rx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                   .rdy(rx_rdy), .i_data(lk_resp_data), .i_empty(lk_resp_empty), .o_rd(lk_resp_rd));

    reg [7:0] pkt [0:79];
    reg [7:0] resp [0:79];
    reg [9:0] pkt_len, resp_len, idx, rcnt;
    reg [15:0] bcnt;

    reg [W*D-1:0]   xb, xnb;
    reg signed [7:0] sx0, sxn;

    // logits store : VOCAB int8 + one shift per chunk
    reg signed [7:0] lgt [0:VOCAB-1];
    reg signed [7:0] csh [0:NCHUNK-1];
    reg [3:0] chunk;

    localparam PH_FN=0, PH_FQ=1;
    reg phase;
    localparam IDLE=0, BWAIT=1, BUILD=2, E_SET=3, E_BUSY=4, E_DONE=5, RECV=6,
               EXTRACT=7, AMAX_INIT=8, AMAX=9, DONE_ST=10;
    reg [3:0] st;

    // argmax sweep state
    reg signed [7:0] sref;
    reg signed [31:0] best_val;
    reg [9:0] amax_i;
    reg [9:0] best_idx;
    reg [3:0] scan_c;

    // one-byte-per-ack handshake : rx8_link fires on a rising edge of rdy, so rdy
    // must drop one cycle after each valid (same fix as the causal sequencer).
    reg rx_ack;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rx_ack <= 1'b1; else rx_ack <= ~rx_valid;
    assign rx_rdy = (st == RECV) & rx_ack;

    function [7:0] a0; input [22:0] a; a0=a[7:0]; endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8]; endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; idx<=0; rcnt<=0; bcnt<=0; phase<=PH_FN; chunk<=0;
        end else begin
            tx_send<=0; done<=0;
            if (st==RECV && rx_valid) begin resp[rcnt]<=rx_data; rcnt<=rcnt+1; end

            case (st)
                IDLE: if (start) begin xb<=x_in; sx0<=sx_in; sxn<=sx_in;
                        phase<=PH_FN; chunk<=0; bcnt<=0; st<=BWAIT; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    if (phase==PH_FN) begin
                        pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                        for(j=0;j<D;j=j+1) pkt[4+j]<=xb[j*W +: W];
                        pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                        pkt_len<=71; resp_len<=75;
                    end else begin // PH_FQ : chunk c of tok_emb
                        pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw_emb;
                        for(j=0;j<D;j=j+1) pkt[5+j]<=xnb[j*W +: W];
                        // chunk c starts at A_EMB + c*64*64  (64 rows * 64 cols)
                        pkt[69]<=a0(A_EMB+base + ({19'd0,chunk} <<< 12));
                        pkt[70]<=a1(A_EMB+base + ({19'd0,chunk} <<< 12));
                        pkt[71]<=a2(A_EMB+base + ({19'd0,chunk} <<< 12));
                        pkt_len<=72; resp_len<=3+D;
                    end
                    idx<=0; st<=E_SET;
                end

                E_SET: begin tx_data<=pkt[idx]; tx_send<=1; st<=E_BUSY; end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin idx<=idx+1; st<=E_SET; end
                end
                RECV: if (rcnt==resp_len) st<=EXTRACT;

                EXTRACT: begin
                    if (phase==PH_FN) begin
                        for(j=0;j<D;j=j+1) xnb[j*W +: W]<=resp[11+j]; sxn<=$signed(resp[2]);
                        phase<=PH_FQ; chunk<=0; st<=BUILD;
                    end else begin
                        // store this chunk's 64 logits + its shift
                        for(j=0;j<D;j=j+1) lgt[chunk*D + j] <= $signed(resp[3+j]);
                        csh[chunk] <= $signed(resp[2]);
                        if (chunk==NCHUNK-1) st<=AMAX_INIT;
                        else begin chunk<=chunk+1; st<=BUILD; end
                    end
                end

                // find sref = max chunk shift, then argmax over aligned logits
                AMAX_INIT: begin
                    // sref = max(csh) : serial over NCHUNK
                    sref <= csh[0]; scan_c<=1;
                    best_val <= -32'sd2147483647; best_idx<=0; amax_i<=0;
                    st<=AMAX;
                end
                AMAX: begin
                    if (scan_c < NCHUNK) begin
                        if (csh[scan_c] > sref) sref<=csh[scan_c];
                        scan_c<=scan_c+1;
                    end else begin
                        // sweep the VOCAB logits, aligning each to sref (right-shift by
                        // sref - chunk_shift), track the running max index.
                        begin: sweep
                            reg signed [31:0] v;
                            v = $signed(lgt[amax_i]) >>> (sref - csh[amax_i[9:6]]);
                            if (v > best_val) begin best_val<=v; best_idx<=amax_i; end
                        end
                        if (amax_i==VOCAB-1) begin token<=best_idx; st<=DONE_ST; end
                        else amax_i<=amax_i+1;
                    end
                end
                DONE_ST: begin done<=1; st<=IDLE; end
            endcase
        end
    end
endmodule
