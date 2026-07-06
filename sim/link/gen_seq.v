`timescale 1ns/1ps
// =============================================================================
// gen_seq - autonomous GENERATION sequencer (mono-card, NN=1), LUT-lean.
//
// Forked from ffn_tp_seq2 (register-file vfile, the optimised -49% LUT style) and
// extended step by step into the full generation orchestrator. Built INCREMENTALLY
// with one validated gate per step (docs/PLAN_GEN_SEQUENCER.md).
//
//   STEP A : embedding                                         [validated]
//   STEP B : causal attention block (this file)                <-- here
//     FN(XB,rms_att) -> XN ; FQ(XN)->Q/Kc/Vc ; RR per head on Q,Kc ;
//     write Kc_roped/Vc into kvmem[pos] ; scan max-shift ; MM(T=pos+1) -> ATT ;
//     FQ(ATT,wo) -> OUTV ; residual XB = XB + OUTV.  Ports the validated
//     attn_causal_seq (Phase 5b) into the vfile style (no 512-bit mux).
//
// The 3 Phase-5b pitfalls are avoided here : (1) rx_rdy pulses via rx_ack,
// (2) sKref/sVref seeded at -128 (never read an uninitialised shift),
// (3) MM K/V streamed by incremental counters (no div/mod).
//
// Slot map (vfile) : XB=0 XN=1 Q=2 Kc=3 Vc=4 ATT=5 OUTV=6 . Each slot is D=64
// bytes (Kc/Vc use only the first KVW=32). KV cache in a separate BSRAM (kvmem).
// =============================================================================
module gen_seq #(
    parameter W = 8,
    parameter D = 64,
    parameter H = 8, parameter KH = 4, parameter HS = 8,
    parameter TMAX = 32,
    parameter BOOT = 40000,
    parameter A_RMS = 23'h100000,
    parameter A_WQ  = 23'h101000,
    parameter A_WK  = 23'h102000,
    parameter A_WV  = 23'h103000,
    parameter A_WO  = 23'h104000
) (
    input  wire            clk, rst_n, start,
    input  wire [W*D-1:0]  x_in,          // step-B: external x (bypasses embed)
    input  wire signed [7:0] sx_in,
    input  wire signed [7:0] sw_rms, swq, swk, swv, swo,
    input  wire [5:0]      pos,
    input  wire [16*(HS/2)-1:0] cos_q15,
    input  wire [16*(HS/2)-1:0] sin_q15,
    input  wire [22:0]     base,
    output wire [7:0]      lk_cmd_data,
    output wire            lk_cmd_wr,
    input  wire            lk_cmd_full,
    input  wire [7:0]      lk_resp_data,
    input  wire            lk_resp_empty,
    output wire            lk_resp_rd,
    output reg  [W*D-1:0]  result,        // XB after residual (x + attn_out)
    output reg  signed [7:0] result_sh,
    output reg             done
);
    localparam KVW  = KH*HS;              // 32
    localparam NREP = H/KH;
    localparam NSLOTS = 8;
    localparam SW = 3;
    localparam AW = SW + 6;
    localparam [SW-1:0] SB_XB=0, SB_XN=1, SB_Q=2, SB_Kc=3, SB_Vc=4, SB_ATT=5, SB_OUTV=6;

    // ---- register file (LUT-RAM) ----
    reg [7:0] vfile [0:NSLOTS*D-1];
    reg [AW-1:0] vaddr; reg [7:0] vdin; reg vwe;
    function [AW-1:0] vidx; input [SW-1:0] slot; input [5:0] b; vidx = (slot<<6)|b; endfunction

    // ---- KV cache : separate BSRAM ----
    reg signed [7:0] kmem [0:TMAX*KVW-1];
    reg signed [7:0] vmem [0:TMAX*KVW-1];
    reg signed [7:0] ksh  [0:TMAX-1];
    reg signed [7:0] vsh  [0:TMAX-1];

    // ---- ss_link byte adapters ----
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    wire [7:0] rx_data; wire rx_valid; wire rx_rdy;
    tx8_link u_tx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                   .busy(tx_busy), .o_data(lk_cmd_data), .o_wr(lk_cmd_wr), .i_full(lk_cmd_full));
    rx8_link u_rx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                   .rdy(rx_rdy), .i_data(lk_resp_data), .i_empty(lk_resp_empty), .o_rd(lk_resp_rd));

    reg [7:0] pkt [0:79];
    reg [9:0] pkt_len, resp_len, idx, rcnt;
    reg [6:0] ldi;
    reg [23:0] bcnt;

    reg signed [7:0] sx0, sxn, sQ, sK, sV, sA, sOut;

    function signed [7:0] clip8; input signed [31:0] v;
        clip8 = (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0]; endfunction

    // phases
    localparam [3:0] PH_FN=0, PH_WQ=1, PH_WK=2, PH_WV=3, PH_MM=4, PH_WO=5;
    reg [3:0] phase;
    reg [3:0] src_slot, dst_slot;
    reg [7:0] Nfq;                          // FQ output count for current phase

    localparam IDLE=0, LOADX=1, BWAIT=2, BUILD=3, E_SET=4, E_BUSY=5, E_DONE=6, RECV=7,
               ROPEQ=8, ROPEK=9, KVWR=10, SCAN=11, RESID=12, DONE_ST=13;
    reg [3:0] st;

    // rx one-byte-per-ack handshake
    reg rx_ack;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rx_ack <= 1'b1; else rx_ack <= ~rx_valid;
    assign rx_rdy = (st == RECV) & rx_ack;

    function [7:0] a0; input [22:0] a; a0=a[7:0]; endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8]; endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction

    // rope one head : reads 8 bytes at (slot, h*8), writes them back roped
    reg [6:0] hidx;                          // head index during rope
    reg [3:0] rope_slot;                     // slot being roped (Q or Kc)
    reg signed [7:0] rh [0:HS-1];            // head bytes being roped

    // max shift scan + MM streaming counters
    reg signed [7:0] sKref, sVref;
    reg [5:0] scan_i;
    reg [5:0] mmp; reg [5:0] mmo; reg mmv;   // MM K/V stream position/offset/region

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; idx<=0; rcnt<=0; bcnt<=0; phase<=PH_FN;
            ldi<=0; vwe<=0; hidx<=0; mmp<=0; mmo<=0; mmv<=0;
        end else begin
            tx_send<=0; done<=0; vwe<=0;
            if (vwe) vfile[vaddr] <= vdin;

            case (st)
                // load external x into vfile[XB], then start the attention block
                IDLE: if (start) begin sx0<=sx_in; sxn<=sx_in; phase<=PH_FN; bcnt<=0; ldi<=0; st<=LOADX; end
                LOADX: begin vaddr<=vidx(SB_XB, ldi[5:0]); vdin<=x_in[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; st<=BWAIT; end else ldi<=ldi+1; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (phase)
                        PH_FN: begin
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                            pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                            pkt_len<=71; resp_len<=75; src_slot<=SB_XB; dst_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_WQ: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=H*HS;pkt[3]<=sxn;pkt[4]<=swq;
                            pkt[69]<=a0(A_WQ+base);pkt[70]<=a1(A_WQ+base);pkt[71]<=a2(A_WQ+base);
                            pkt_len<=72; resp_len<=3+H*HS; src_slot<=SB_XN; dst_slot<=SB_Q; Nfq<=H*HS; end
                        PH_WK: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swk;
                            pkt[69]<=a0(A_WK+base);pkt[70]<=a1(A_WK+base);pkt[71]<=a2(A_WK+base);
                            pkt_len<=72; resp_len<=3+KH*HS; src_slot<=SB_XN; dst_slot<=SB_Kc; Nfq<=KH*HS; end
                        PH_WV: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swv;
                            pkt[69]<=a0(A_WV+base);pkt[70]<=a1(A_WV+base);pkt[71]<=a2(A_WV+base);
                            pkt_len<=72; resp_len<=3+KH*HS; src_slot<=SB_XN; dst_slot<=SB_Vc; Nfq<=KH*HS; end
                        PH_MM: begin
                            pkt[0]<="M";pkt[1]<="M";pkt[2]<=sQ;pkt[3]<=sKref;pkt[4]<=sVref;
                            pkt[5]<=({2'd0,pos}+8'd1);
                            // Q from vfile[Q], then K/V streamed from kvmem
                            pkt_len<=6+D+2*(({2'd0,pos}+10'd1)*KVW); resp_len<=3+D;
                            src_slot<=SB_Q; end
                        default: begin // PH_WO
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sA;pkt[4]<=swo;
                            pkt[69]<=a0(A_WO+base);pkt[70]<=a1(A_WO+base);pkt[71]<=a2(A_WO+base);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_ATT; dst_slot<=SB_OUTV; Nfq<=D[7:0]; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                // stream packet; vector bytes read combinationally from vfile
                E_SET: begin
                    if (phase==PH_FN && idx>=4 && idx<4+D)          tx_data<=vfile[vidx(src_slot, idx[5:0]-4)];
                    else if ((phase==PH_WQ||phase==PH_WK||phase==PH_WV||phase==PH_WO) && idx>=5 && idx<5+D)
                                                                    tx_data<=vfile[vidx(src_slot, idx[5:0]-5)];
                    else if (phase==PH_MM && idx>=6 && idx<6+D)     tx_data<=vfile[vidx(SB_Q, idx[5:0]-6)];
                    else if (phase==PH_MM && idx>=(6+D)) begin
                        if (!mmv) tx_data <= clip8( $signed(kmem[mmp*KVW + mmo]) >>> (sKref - ksh[mmp]) );
                        else      tx_data <= clip8( $signed(vmem[mmp*KVW + mmo]) >>> (sVref - vsh[mmp]) );
                    end
                    else tx_data<=pkt[idx];
                    tx_send<=1; st<=E_BUSY;
                end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin
                        idx<=idx+1;
                        if (phase==PH_MM && idx>=(6+D)) begin
                            if (mmo==KVW-1) begin mmo<=0;
                                if (mmp==pos) begin mmp<=0; mmv<=1'b1; end else mmp<=mmp+1;
                            end else mmo<=mmo+1;
                        end
                        st<=E_SET;
                    end
                end

                // stream response; write vector bytes into dst_slot
                RECV: if (rx_valid) begin
                    if (phase==PH_FN && rcnt>=11 && rcnt<11+D)   begin vaddr<=vidx(SB_XN, rcnt[5:0]-11); vdin<=rx_data; vwe<=1'b1; end
                    else if ((phase==PH_WQ||phase==PH_WK||phase==PH_WV||phase==PH_WO) && rcnt>=3 && rcnt<3+Nfq)
                                                                  begin vaddr<=vidx(dst_slot, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_MM && rcnt>=3 && rcnt<3+D) begin vaddr<=vidx(SB_ATT, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    case (phase)
                        PH_FN: if (rcnt==2) sxn<=$signed(rx_data);
                        PH_WQ: if (rcnt==2) sQ<=$signed(rx_data);
                        PH_WK: if (rcnt==2) sK<=$signed(rx_data);
                        PH_WV: if (rcnt==2) sV<=$signed(rx_data);
                        PH_MM: if (rcnt==2) sA<=$signed(rx_data);
                        PH_WO: if (rcnt==2) sOut<=$signed(rx_data);
                    endcase
                    if (rcnt==resp_len-1) begin
                        case (phase)
                            PH_FN: begin phase<=PH_WQ; st<=BUILD; end
                            PH_WQ: begin phase<=PH_WK; st<=BUILD; end
                            PH_WK: begin phase<=PH_WV; st<=BUILD; end
                            PH_WV: begin hidx<=0; rope_slot<=SB_Q; st<=ROPEQ; end
                            PH_MM: begin phase<=PH_WO; st<=BUILD; end
                            default: begin ldi<=0; st<=RESID; end // PH_WO -> residual
                        endcase
                    end else rcnt<=rcnt+1;
                end

                // rope Q heads (H) then K heads (KH). Two-phase per head to avoid
                // read-after-write corruption : ldi==0 computes ALL 8 roped bytes into
                // rh from the ORIGINAL vfile bytes (rope preserves position, so writing
                // must not feed back into the compute). ldi 0..7 then write rh serially.
                // The compute reads vfile at hidx*8; the writes at hidx*8+ldi only start
                // taking effect from ldi>=1, but rh already holds the full head -> safe.
                ROPEQ: begin
                    if (ldi==0) begin: rqc
                        integer p; reg signed [15:0] cq, sq; reg signed [7:0] xr, xi;
                        reg signed [31:0] nr, ni;
                        for (p=0;p<HS/2;p=p+1) begin
                            xr = $signed(vfile[vidx(SB_Q, hidx[2:0]*HS + 2*p)]);
                            xi = $signed(vfile[vidx(SB_Q, hidx[2:0]*HS + 2*p + 1)]);
                            cq = $signed(cos_q15[16*p +: 16]); sq = $signed(sin_q15[16*p +: 16]);
                            nr = (xr*cq - xi*sq + 32'sd16384) >>> 15;
                            ni = (xr*sq + xi*cq + 32'sd16384) >>> 15;
                            rh[2*p]   <= clip8(nr); rh[2*p+1] <= clip8(ni);
                        end
                        ldi<=1;   // rh not yet written this cycle; start writing next cycle
                    end else begin
                        vaddr<=vidx(SB_Q, hidx[2:0]*HS + (ldi-1)); vdin<=rh[ldi-1]; vwe<=1'b1;
                        if (ldi==HS) begin ldi<=0;
                            if (hidx==H-1) begin hidx<=0; st<=ROPEK; end
                            else hidx<=hidx+1;
                        end else ldi<=ldi+1;
                    end
                end
                ROPEK: begin
                    if (ldi==0) begin: rkc
                        integer p; reg signed [15:0] cq, sq; reg signed [7:0] xr, xi;
                        reg signed [31:0] nr, ni;
                        for (p=0;p<HS/2;p=p+1) begin
                            xr = $signed(vfile[vidx(SB_Kc, hidx[2:0]*HS + 2*p)]);
                            xi = $signed(vfile[vidx(SB_Kc, hidx[2:0]*HS + 2*p + 1)]);
                            cq = $signed(cos_q15[16*p +: 16]); sq = $signed(sin_q15[16*p +: 16]);
                            nr = (xr*cq - xi*sq + 32'sd16384) >>> 15;
                            ni = (xr*sq + xi*cq + 32'sd16384) >>> 15;
                            rh[2*p]   <= clip8(nr); rh[2*p+1] <= clip8(ni);
                        end
                        ldi<=1;
                    end else begin
                        vaddr<=vidx(SB_Kc, hidx[2:0]*HS + (ldi-1)); vdin<=rh[ldi-1]; vwe<=1'b1;
                        if (ldi==HS) begin ldi<=0;
                            if (hidx==KH-1) begin hidx<=0; st<=KVWR; end
                            else hidx<=hidx+1;
                        end else ldi<=ldi+1;
                    end
                end

                // copy roped Kc / Vc into kvmem[pos], store shifts
                KVWR: begin
                    kmem[pos*KVW + ldi[5:0]] <= $signed(vfile[vidx(SB_Kc, ldi[5:0])]);
                    vmem[pos*KVW + ldi[5:0]] <= $signed(vfile[vidx(SB_Vc, ldi[5:0])]);
                    if (ldi==KVW-1) begin
                        ksh[pos]<=sK; vsh[pos]<=sV;
                        sKref<=-8'sd128; sVref<=-8'sd128; scan_i<=0; st<=SCAN;
                    end else ldi<=ldi+1;
                end
                // serial max-shift scan over positions 0..pos (seed -128)
                SCAN: begin
                    if (ksh[scan_i] > sKref) sKref<=ksh[scan_i];
                    if (vsh[scan_i] > sVref) sVref<=vsh[scan_i];
                    if (scan_i==pos) begin mmp<=0; mmo<=0; mmv<=1'b0; phase<=PH_MM; st<=BUILD; end
                    else scan_i<=scan_i+1;
                end

                // residual : XB = XB + OUTV  (needs an add; reuse vec via a serial int add)
                // XB and OUTV share shift sx0 (XB) and sOut (OUTV=Wo output). We align
                // to the smaller shift and add. sOut = sA+swo add-shift = the FQ shift.
                // residual XB = XB + OUTV, both aligned to the SMALLER shift (= more
                // fractional bits). shift diffs are non-negative, cast to unsigned so
                // <<< never gets a negative (undefined -> X) count.
                RESID: begin
                    begin: res
                        reg signed [7:0] xbv, ov;
                        reg [7:0] dx, dov;
                        reg signed [31:0] a32, b32, sum;
                        xbv = $signed(vfile[vidx(SB_XB, ldi[5:0])]);
                        ov  = $signed(vfile[vidx(SB_OUTV, ldi[5:0])]);
                        if (sx0 <= sOut) begin dx = 8'd0;          dov = sOut - sx0; end
                        else             begin dx = sx0 - sOut;    dov = 8'd0;       end
                        a32 = $signed({{24{xbv[7]}}, xbv}) <<< dx;
                        b32 = $signed({{24{ov[7]}},  ov })  <<< dov;
                        sum = a32 + b32;
                        vaddr<=vidx(SB_XB, ldi[5:0]); vdin<=clip8(sum); vwe<=1'b1;
                    end
                    if (ldi==D-1) begin ldi<=0; st<=DONE_ST; end else ldi<=ldi+1;
                end

                // stream XB (x + attn) to result
                DONE_ST: begin
                    result[ldi*W +: W] <= vfile[vidx(SB_XB, ldi[5:0])];
                    if (ldi==D-1) begin result_sh<=(sx0<sOut)?sx0:sOut; done<=1; st<=IDLE; end
                    else ldi<=ldi+1;
                end
            endcase
        end
    end
endmodule
