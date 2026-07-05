`timescale 1ns/1ps
// =============================================================================
// attn_causal_seq - autonomous CAUSAL attention block for ONE node (NN=1), over
// the parallel ss_link. Orchestrates the node primitives to compute one causal
// attention step at position `pos`, reproducing prove_causal_orch.py byte-for-byte:
//
//   1. FN(x)                      -> xn                     (rmsnorm)
//   2. FQ(Wq) FQ(Wk) FQ(Wv)       -> Q[64], K[32], V[32]
//   3. RR per head on Q (8) and K (4)   [rope_op: shift preserved -> concat]
//   4. write K_roped,V into the KV-cache at slot `pos`
//   5. re-align K[0..pos],V[0..pos] to sKref=max(sK)/sVref=max(sV) by integer
//      right-shift; MM layout [t][kvh][hs] (stride 32)
//   6. MM(Q, Kpack, Vpack, T=pos+1)     -> attn[64]
//   7. FQ(Wo, attn)               -> out ; caller adds the residual x+out
//
// This is the causal core; the generation loop (embed, 5 layers, lm_head, argmax,
// KV persistence across tokens) is built on top later. Single node = no gather/
// reduce, so the sequencer is pure orchestration + the integer KV re-align.
//
// KV cache is INTERNAL here (kmem/vmem), one shift byte per (pos): the test seeds
// earlier positions, the sequencer fills slot `pos` and runs the MM over 0..pos.
// =============================================================================
module attn_causal_seq #(
    parameter W  = 8,
    parameter D  = 64,
    parameter H  = 8,       // query heads
    parameter KH = 4,       // kv heads (GQA)
    parameter HS = 8,       // head size
    parameter TMAX = 32,    // max sequence positions
    parameter A_RMS = 23'h100000,
    parameter A_WQ  = 23'h101000,
    parameter A_WK  = 23'h102000,
    parameter A_WV  = 23'h103000,
    parameter A_WO  = 23'h104000
) (
    input  wire            clk, rst_n, start,
    input  wire [W*D-1:0]  x_in,
    input  wire signed [7:0] sx_in,
    input  wire signed [7:0] sw_rms, swq, swk, swv, swo,
    input  wire [5:0]      pos,          // current position (0..TMAX-1)
    // rope cos/sin for THIS position : HS/2 pairs, Q15 signed 16-bit each
    input  wire [16*(HS/2)-1:0] cos_q15,
    input  wire [16*(HS/2)-1:0] sin_q15,
    input  wire [22:0]     base,
    // parallel link to the single node
    output wire [7:0]      lk_cmd_data,
    output wire            lk_cmd_wr,
    input  wire            lk_cmd_full,
    input  wire [7:0]      lk_resp_data,
    input  wire            lk_resp_empty,
    output wire            lk_resp_rd,
    output reg  [W*D-1:0]  result,       // attn_out[64] after Wo (before residual)
    output reg  signed [7:0] result_sh,
    output reg             done
);
    localparam KVW = KH*HS;              // 32 : kv bytes per position
    localparam NREP = H/KH;

    // ---- link byte adapters (same as the other ss_link sequencers) ----
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    wire [7:0] rx_data; wire rx_valid; wire rx_rdy;
    tx8_link u_tx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                   .busy(tx_busy), .o_data(lk_cmd_data), .o_wr(lk_cmd_wr), .i_full(lk_cmd_full));
    rx8_link u_rx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                   .rdy(rx_rdy), .i_data(lk_resp_data), .i_empty(lk_resp_empty), .o_rd(lk_resp_rd));

    // ---- staging buffers ----
    reg [7:0] pkt [0:159];
    reg [7:0] resp [0:79];
    reg [9:0] pkt_len, resp_len, idx, rcnt;
    reg [15:0] bcnt;

    // vectors held between steps
    reg [W*D-1:0]   xb, xnb, Qb, Kb, Vb, attnb;
    reg signed [7:0] sx0, sxn, sQ, sK, sV, sA, sOut;

    // KV cache : kmem/vmem [TMAX][KVW] bytes + a shift per position
    reg signed [7:0] kmem [0:TMAX*KVW-1];
    reg signed [7:0] vmem [0:TMAX*KVW-1];
    reg signed [7:0] ksh  [0:TMAX-1];
    reg signed [7:0] vsh  [0:TMAX-1];

    // ---- rope : Q15 fixed-point, one head (HS bytes) at a time ----
    // out[2i]   = (xr*cq - xi*sq + 16384) >>> 15
    // out[2i+1] = (xr*sq + xi*cq + 16384) >>> 15   ; shift preserved
    function signed [7:0] clip8; input signed [31:0] v;
        clip8 = (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0];
    endfunction

    localparam BOOT = 40000;
    // phases
    localparam PH_FN=0, PH_WQ=1, PH_WK=2, PH_WV=3, PH_MM=4, PH_WO=5;
    reg [2:0] phase;

    localparam IDLE=0, BWAIT=1, BUILD=2, E_SET=3, E_BUSY=4, E_DONE=5, RECV=6,
               EXTRACT=7, ROPEQ=8, ROPEK=9, KVWRITE=10, SCAN=11, DONE_ST=12;
    reg [3:0] st;
    reg [6:0] hidx;        // head index during rope
    reg [9:0] mmk;         // byte index while packing K/V for MM

    // rx8_link releases the next byte on a RISING edge of rdy, so rdy must drop for
    // one cycle after each delivered byte. rx_ack falls the cycle after a valid pulse
    // (mirrors the Phase-2 ffn/attn sequencers). Without this, RECV gets only 1 byte.
    reg rx_ack;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rx_ack <= 1'b1; else rx_ack <= ~rx_valid;
    assign rx_rdy = (st == RECV) & rx_ack;

    function [7:0] a0; input [22:0] a; a0=a[7:0]; endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8]; endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction

    integer j;

    // max shift over cached positions 0..pos : computed ONCE (registers) after the
    // KV write, by a small serial scan, not a permanent combinational reduction.
    reg signed [7:0] sKref, sVref;
    reg [5:0] scan_i;

    // MM K/V streaming counters (incremental, no div/mod in the datapath)
    reg [5:0] mmp;        // current cached position 0..pos
    reg [5:0] mmo;        // byte offset within a position 0..KVW-1
    reg       mmv;        // 0 = streaming K region, 1 = streaming V region
    reg [9:0] mm_total;   // total K||V bytes = 2*T*KVW

    // rope one head : x8 (8 bytes from a 64/32-wide reg) with the pos cos/sin
    task rope_head;
        input  [W*HS-1:0] in8;
        output [W*HS-1:0] out8;
        integer p; reg signed [15:0] cq, sq; reg signed [7:0] xr, xi;
        reg signed [31:0] nr, ni;
        begin
            for (p=0; p<HS/2; p=p+1) begin
                xr = $signed(in8[(2*p)*W +: W]);
                xi = $signed(in8[(2*p+1)*W +: W]);
                cq = $signed(cos_q15[16*p +: 16]);
                sq = $signed(sin_q15[16*p +: 16]);
                nr = (xr*cq - xi*sq + 32'sd16384) >>> 15;
                ni = (xr*sq + xi*cq + 32'sd16384) >>> 15;
                out8[(2*p)*W +: W]   = clip8(nr);
                out8[(2*p+1)*W +: W] = clip8(ni);
            end
        end
    endtask

    reg [W*HS-1:0] rope_in, rope_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; idx<=0; rcnt<=0; bcnt<=0; phase<=PH_FN; hidx<=0; mmk<=0;
        end else begin
            tx_send<=0; done<=0;
            if (st==RECV && rx_valid) begin resp[rcnt]<=rx_data; rcnt<=rcnt+1; end

            case (st)
                IDLE: if (start) begin xb<=x_in; sx0<=sx_in; sxn<=sx_in;
                        phase<=PH_FN; bcnt<=0; st<=BWAIT; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (phase)
                        PH_FN: begin
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                            for(j=0;j<D;j=j+1) pkt[4+j]<=xb[j*W +: W];
                            pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                            pkt_len<=71; resp_len<=75; end
                        PH_WQ: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=H*HS;pkt[3]<=sxn;pkt[4]<=swq;
                            for(j=0;j<D;j=j+1) pkt[5+j]<=xnb[j*W +: W];
                            pkt[69]<=a0(A_WQ+base);pkt[70]<=a1(A_WQ+base);pkt[71]<=a2(A_WQ+base);
                            pkt_len<=72; resp_len<=3+H*HS; end
                        PH_WK: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swk;
                            for(j=0;j<D;j=j+1) pkt[5+j]<=xnb[j*W +: W];
                            pkt[69]<=a0(A_WK+base);pkt[70]<=a1(A_WK+base);pkt[71]<=a2(A_WK+base);
                            pkt_len<=72; resp_len<=3+KH*HS; end
                        PH_WV: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swv;
                            for(j=0;j<D;j=j+1) pkt[5+j]<=xnb[j*W +: W];
                            pkt[69]<=a0(A_WV+base);pkt[70]<=a1(A_WV+base);pkt[71]<=a2(A_WV+base);
                            pkt_len<=72; resp_len<=3+KH*HS; end
                        PH_MM: begin
                            // header : 'M''M' sQ sKref sV(ref) T ; then Q[64], K[T*32], V[T*32]
                            pkt[0]<="M";pkt[1]<="M";pkt[2]<=sQ;pkt[3]<=sKref;pkt[4]<=sVref;
                            pkt[5]<=({2'd0,pos}+8'd1);        // T = pos+1
                            for(j=0;j<D;j=j+1) pkt[6+j]<=Qb[j*W +: W];   // Q[64]
                            // K/V bytes are streamed from kmem/vmem in E_SET (too big for pkt[])
                            pkt_len<=6+D+2*(({2'd0,pos}+10'd1)*KVW); // total len
                            resp_len<=3+D; end
                        default: begin // PH_WO
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sA;pkt[4]<=swo;
                            for(j=0;j<D;j=j+1) pkt[5+j]<=attnb[j*W +: W];
                            pkt[69]<=a0(A_WO+base);pkt[70]<=a1(A_WO+base);pkt[71]<=a2(A_WO+base);
                            pkt_len<=72; resp_len<=3+D; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                // stream bytes. For MM, the header+Q come from pkt[0..6+D-1]; the
                // K||V payload is streamed from kmem/vmem via incremental counters
                // (mmp position, mmo offset, mmv K/V region) with the integer
                // re-align (right-shift to sKref/sVref) applied per byte.
                E_SET: begin
                    if (phase==PH_MM && idx >= (6+D)) begin
                        if (!mmv)
                            tx_data <= clip8( $signed(kmem[mmp*KVW + mmo]) >>> (sKref - ksh[mmp]) );
                        else
                            tx_data <= clip8( $signed(vmem[mmp*KVW + mmo]) >>> (sVref - vsh[mmp]) );
                    end else begin
                        tx_data <= pkt[idx];
                    end
                    tx_send<=1; st<=E_BUSY;
                end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin
                        idx<=idx+1;
                        // advance the K/V streaming counters once we're in the payload
                        if (phase==PH_MM && idx >= (6+D)) begin
                            if (mmo==KVW-1) begin
                                mmo<=0;
                                if (mmp==pos) begin mmp<=0; mmv<=1'b1; end  // K done -> V region
                                else mmp<=mmp+1;
                            end else mmo<=mmo+1;
                        end
                        st<=E_SET;
                    end
                end
                RECV: if (rcnt==resp_len) st<=EXTRACT;

                EXTRACT: begin
                    case (phase)
                        PH_FN: begin for(j=0;j<D;j=j+1) xnb[j*W +: W]<=resp[11+j]; sxn<=$signed(resp[2]);
                                     phase<=PH_WQ; st<=BUILD; end
                        PH_WQ: begin for(j=0;j<H*HS;j=j+1) Qb[j*W +: W]<=resp[3+j]; sQ<=$signed(resp[2]);
                                     hidx<=0; st<=ROPEQ; end
                        PH_WK: begin for(j=0;j<KH*HS;j=j+1) Kb[j*W +: W]<=resp[3+j]; sK<=$signed(resp[2]);
                                     hidx<=0; st<=ROPEK; end
                        PH_WV: begin for(j=0;j<KH*HS;j=j+1) Vb[j*W +: W]<=resp[3+j]; sV<=$signed(resp[2]);
                                     st<=KVWRITE; mmk<=0; end
                        PH_MM: begin for(j=0;j<D;j=j+1) attnb[j*W +: W]<=resp[3+j]; sA<=$signed(resp[2]);
                                     phase<=PH_WO; st<=BUILD; end
                        default: begin for(j=0;j<D;j=j+1) result[j*W +: W]<=resp[3+j]; result_sh<=$signed(resp[2]);
                                     st<=DONE_ST; end
                    endcase
                end

                // rope Q : one head per cycle, in place in Qb ; then go fetch K
                ROPEQ: begin
                    rope_in = Qb[hidx*W*HS +: W*HS];
                    rope_head(rope_in, rope_out);
                    Qb[hidx*W*HS +: W*HS] <= rope_out;
                    if (hidx==H-1) begin phase<=PH_WK; st<=BUILD; end
                    else hidx<=hidx+1;
                end
                // rope K : one head per cycle in Kb ; then fetch V
                ROPEK: begin
                    rope_in = Kb[hidx*W*HS +: W*HS];
                    rope_head(rope_in, rope_out);
                    Kb[hidx*W*HS +: W*HS] <= rope_out;
                    if (hidx==KH-1) begin phase<=PH_WV; st<=BUILD; end
                    else hidx<=hidx+1;
                end

                // write K_roped and V into the cache at slot `pos`
                KVWRITE: begin
                    kmem[pos*KVW + mmk] <= $signed(Kb[mmk*W +: W]);
                    vmem[pos*KVW + mmk] <= $signed(Vb[mmk*W +: W]);
                    if (mmk==KVW-1) begin
                        ksh[pos]<=sK; vsh[pos]<=sV;
                        // start the serial max-shift scan over positions 0..pos.
                        // seed refs at the smallest int8 so the first real shift wins
                        // (ksh[0] may be uninitialised X at pos=0 -> must not read it).
                        sKref<=-8'sd128; sVref<=-8'sd128; scan_i<=0; st<=SCAN;
                    end else mmk<=mmk+1;
                end

                // serial scan : sKref/sVref = max shift over cached positions 0..pos.
                // (pos just written; ksh[pos]/vsh[pos] set combinationally below via
                //  the values we just latched, so include the current slot too.)
                SCAN: begin
                    if (ksh[scan_i] > sKref) sKref<=ksh[scan_i];
                    if (vsh[scan_i] > sVref) sVref<=vsh[scan_i];
                    if (scan_i==pos) begin
                        // prime MM streaming counters, then build the MM packet
                        mmp<=0; mmo<=0; mmv<=1'b0; phase<=PH_MM; st<=BUILD;
                    end else scan_i<=scan_i+1;
                end

                DONE_ST: begin done<=1; st<=IDLE; end
            endcase
        end
    end
endmodule
