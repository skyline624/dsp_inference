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
    parameter NL   = 5,                   // number of transformer layers
    parameter LBASE = 23'h010000,         // layer 0 weight base (base_l = LBASE + layer*0x10000)
    // per-layer weight offsets (stories260K SDRAM layout)
    parameter OFF_RMS   = 23'h000000,
    parameter OFF_WQ    = 23'h000100,
    parameter OFF_WK    = 23'h001100,
    parameter OFF_WV    = 23'h001900,
    parameter OFF_WO    = 23'h002100,
    parameter OFF_RMSFF = 23'h003100,
    parameter OFF_W1    = 23'h003200,     // W1 [HID,64], chunk c @ +c*0x1000
    parameter OFF_W3    = 23'h006200,     // W3 [HID,64], chunk c @ +c*0x1000
    parameter OFF_W2    = 23'h009200      // W2 chunk c = [64,64] @ +c*0x1000
) (
    input  wire            clk, rst_n, start,
    input  wire [W*D-1:0]  x_in,          // step-B: external x (bypasses embed)
    input  wire signed [7:0] sx_in,
    input  wire signed [7:0] sw_rms, swq, swk, swv, swo,
    input  wire signed [7:0] sw_rmsf, sw1, sw3, sw2,   // ffn weight shifts
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
    localparam [SW-1:0] SB_XB=0, SB_XN=1, SB_Q=2, SB_Kc=3, SB_Vc=4, SB_ATT=5, SB_OUTV=6, SB_P=7;
    // FFN reuses the attention slots once attention is done : XN=rmsnorm,
    // H1=Q, H3=Kc, SG=Vc, HG=ATT, W2-accumulator=OUTV, W2-partial=P, residual=XB.

    // ---- register file (LUT-RAM) ----
    reg [7:0] vfile [0:NSLOTS*D-1];
    reg [AW-1:0] vaddr; reg [7:0] vdin; reg vwe;
    function [AW-1:0] vidx; input [SW-1:0] slot; input [5:0] b; vidx = (slot<<6)|b; endfunction

    // ---- KV cache : separate BSRAM, indexed by (layer, pos) ----
    reg signed [7:0] kmem [0:NL*TMAX*KVW-1];
    reg signed [7:0] vmem [0:NL*TMAX*KVW-1];
    reg signed [7:0] ksh  [0:NL*TMAX-1];
    reg signed [7:0] vsh  [0:NL*TMAX-1];

    // ---- ss_link byte adapters ----
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    wire [7:0] rx_data; wire rx_valid; wire rx_rdy;
    tx8_link u_tx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                   .busy(tx_busy), .o_data(lk_cmd_data), .o_wr(lk_cmd_wr), .i_full(lk_cmd_full));
    rx8_link u_rx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                   .rdy(rx_rdy), .i_data(lk_resp_data), .i_empty(lk_resp_empty), .o_rd(lk_resp_rd));

    // ---- serialized elementwise ALU (FFN MUL, W2 reduce ADD, residual ADD) ----
    reg            alu_start, alu_op;
    reg  [1:0]     alu_next;                  // 0=MUL->HG, 1=reduce->OUTV, 2=residual->XB
    reg  [W*D-1:0] alu_a, alu_b; reg signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out; wire signed [7:0] alu_out_sh; wire alu_done;
    reg  [SW-1:0]  alu_aslot, alu_bslot, alu_dslot;
    vec_alu2 #(W, D) u_alu (.clk(clk), .rst_n(rst_n), .start(alu_start), .op(alu_op),
        .a(alu_a), .sa(alu_sa), .b(alu_b), .sb(alu_sb),
        .out(alu_out), .out_sh(alu_out_sh), .done(alu_done));

    reg [7:0] pkt [0:79];
    reg [9:0] pkt_len, resp_len, idx, rcnt;
    reg [6:0] ldi;
    reg [23:0] bcnt;

    reg signed [7:0] sx0, sxn, sQ, sK, sV, sA, sOut;
    // FFN running/per-chunk shifts + chunk counter
    reg signed [7:0] sxb, s1c, s3c, ssgc, shc, spc, sov;
    reg [1:0] fc;                            // ffn hidden chunk 0..2 (64+64+44)
    reg in_ffn;                              // 0 while attention, 1 during FFN
    reg [22:0] waddr;                        // blocking temp : chunked weight address
    // FFN hidden dim HID=172 is chunked 3x64 with W1/W3 rows 172..191 zero-padded
    // (test preload) so every chunk is a clean 64-wide op : silu(0)=0, no stale
    // value contaminates the per-chunk output shift, and W2 cols 172..191 are 0.
    reg [2:0] layer;                         // current transformer layer 0..NL-1
    wire [22:0] base_l = base + LBASE + {layer,16'b0};   // = base + LBASE + layer*0x10000

    function signed [7:0] clip8; input signed [31:0] v;
        clip8 = (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0]; endfunction

    // phases : attention (FN..WO) then FFN (FN2, W1, W3, SS, W2)
    localparam [3:0] PH_FN=0, PH_WQ=1, PH_WK=2, PH_WV=3, PH_MM=4, PH_WO=5,
                     PH_FN2=6, PH_FW1=7, PH_FW3=8, PH_SS=9, PH_FW2=10;
    reg [3:0] phase;
    reg [3:0] src_slot, dst_slot;
    reg [7:0] Nfq;                          // FQ output count for current phase

    localparam IDLE=0, LOADX=1, BWAIT=2, BUILD=3, E_SET=4, E_BUSY=5, E_DONE=6, RECV=7,
               ROPEQ=8, ROPEK=9, KVWR=10, SCAN=11, RESID=12, DONE_ST=13,
               ALD_A=14, ALD_B=15, ALU_WAIT=16, AST_A=17, FRED_COPY=18;
    reg [4:0] st;

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
            alu_start<=0; fc<=0; in_ffn<=0; layer<=0;
        end else begin
            tx_send<=0; done<=0; vwe<=0; alu_start<=0;
            if (vwe) vfile[vaddr] <= vdin;

            case (st)
                // load external x into vfile[XB], then start the attention block
                IDLE: if (start) begin sx0<=sx_in; sxn<=sx_in; sxb<=sx_in;
                        phase<=PH_FN; fc<=0; in_ffn<=0; layer<=0; bcnt<=0; ldi<=0; st<=LOADX; end
                LOADX: begin vaddr<=vidx(SB_XB, ldi[5:0]); vdin<=x_in[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; st<=BWAIT; end else ldi<=ldi+1; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (phase)
                        PH_FN: begin
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sxb;pkt[3]<=sw_rms;
                            pkt[68]<=a0(base_l+OFF_RMS);pkt[69]<=a1(base_l+OFF_RMS);pkt[70]<=a2(base_l+OFF_RMS);
                            pkt_len<=71; resp_len<=75; src_slot<=SB_XB; dst_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_WQ: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=H*HS;pkt[3]<=sxn;pkt[4]<=swq;
                            pkt[69]<=a0(base_l+OFF_WQ);pkt[70]<=a1(base_l+OFF_WQ);pkt[71]<=a2(base_l+OFF_WQ);
                            pkt_len<=72; resp_len<=3+H*HS; src_slot<=SB_XN; dst_slot<=SB_Q; Nfq<=H*HS; end
                        PH_WK: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swk;
                            pkt[69]<=a0(base_l+OFF_WK);pkt[70]<=a1(base_l+OFF_WK);pkt[71]<=a2(base_l+OFF_WK);
                            pkt_len<=72; resp_len<=3+KH*HS; src_slot<=SB_XN; dst_slot<=SB_Kc; Nfq<=KH*HS; end
                        PH_WV: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swv;
                            pkt[69]<=a0(base_l+OFF_WV);pkt[70]<=a1(base_l+OFF_WV);pkt[71]<=a2(base_l+OFF_WV);
                            pkt_len<=72; resp_len<=3+KH*HS; src_slot<=SB_XN; dst_slot<=SB_Vc; Nfq<=KH*HS; end
                        PH_MM: begin
                            pkt[0]<="M";pkt[1]<="M";pkt[2]<=sQ;pkt[3]<=sKref;pkt[4]<=sVref;
                            pkt[5]<=({2'd0,pos}+8'd1);
                            // Q from vfile[Q], then K/V streamed from kvmem
                            pkt_len<=6+D+2*(({2'd0,pos}+10'd1)*KVW); resp_len<=3+D;
                            src_slot<=SB_Q; end
                        PH_WO: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sA;pkt[4]<=swo;
                            pkt[69]<=a0(base_l+OFF_WO);pkt[70]<=a1(base_l+OFF_WO);pkt[71]<=a2(base_l+OFF_WO);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_ATT; dst_slot<=SB_OUTV; Nfq<=D[7:0]; end
                        // ---- FFN ----
                        PH_FN2: begin  // rmsnorm(XB, rms_ffn) -> XN
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sxb;pkt[3]<=sw_rmsf;
                            pkt[68]<=a0(base_l+OFF_RMSFF);pkt[69]<=a1(base_l+OFF_RMSFF);pkt[70]<=a2(base_l+OFF_RMSFF);
                            pkt_len<=71; resp_len<=75; src_slot<=SB_XB; dst_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_FW1: begin  // W1_chunk(XN) -> H1 (SB_Q), N_out=64 (zero-padded)
                            waddr = base_l + OFF_W1 + {fc,12'b0};
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw1;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_XN; dst_slot<=SB_Q; Nfq<=D[7:0]; end
                        PH_FW3: begin  // W3_chunk(XN) -> H3 (SB_Kc)
                            waddr = base_l + OFF_W3 + {fc,12'b0};
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw3;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_XN; dst_slot<=SB_Kc; Nfq<=D[7:0]; end
                        PH_SS: begin   // silu(H1) -> SG (SB_Vc)
                            pkt[0]<="S";pkt[1]<="S";pkt[2]<=s1c;
                            pkt_len<=67; resp_len<=70; src_slot<=SB_Q; dst_slot<=SB_Vc; Nfq<=D[7:0]; end
                        default: begin // PH_FW2 : W2_chunk(HG) -> P (SB_P), N_out=64
                            waddr = base_l + OFF_W2 + {fc,12'b0};
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=shc;pkt[4]<=sw2;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_ATT; dst_slot<=SB_P; Nfq<=D[7:0]; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                // stream packet; vector bytes read combinationally from vfile
                E_SET: begin
                    if ((phase==PH_FN||phase==PH_FN2) && idx>=4 && idx<4+D)
                                                                    tx_data<=vfile[vidx(src_slot, idx[5:0]-4)];
                    else if ((phase==PH_WQ||phase==PH_WK||phase==PH_WV||phase==PH_WO||
                              phase==PH_FW1||phase==PH_FW3||phase==PH_FW2) && idx>=5 && idx<5+D)
                                                                    tx_data<=vfile[vidx(src_slot, idx[5:0]-5)];
                    else if (phase==PH_SS && idx>=3 && idx<3+D)     tx_data<=vfile[vidx(src_slot, idx[5:0]-3)];
                    else if (phase==PH_MM && idx>=6 && idx<6+D)     tx_data<=vfile[vidx(SB_Q, idx[5:0]-6)];
                    else if (phase==PH_MM && idx>=(6+D)) begin
                        if (!mmv) tx_data <= clip8( $signed(kmem[layer*TMAX*KVW + mmp*KVW + mmo]) >>> (sKref - ksh[layer*TMAX + mmp]) );
                        else      tx_data <= clip8( $signed(vmem[layer*TMAX*KVW + mmp*KVW + mmo]) >>> (sVref - vsh[layer*TMAX + mmp]) );
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
                    if ((phase==PH_FN||phase==PH_FN2) && rcnt>=11 && rcnt<11+D)
                                                                  begin vaddr<=vidx(SB_XN, rcnt[5:0]-11); vdin<=rx_data; vwe<=1'b1; end
                    else if ((phase==PH_WQ||phase==PH_WK||phase==PH_WV||phase==PH_WO||
                              phase==PH_FW1||phase==PH_FW3||phase==PH_FW2) && rcnt>=3 && rcnt<3+Nfq)
                                                                  begin vaddr<=vidx(dst_slot, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_SS && rcnt>=6 && rcnt<6+D) begin vaddr<=vidx(dst_slot, rcnt[5:0]-6); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_MM && rcnt>=3 && rcnt<3+D) begin vaddr<=vidx(SB_ATT, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    case (phase)
                        PH_FN:  if (rcnt==2) sxn<=$signed(rx_data);
                        PH_WQ:  if (rcnt==2) sQ<=$signed(rx_data);
                        PH_WK:  if (rcnt==2) sK<=$signed(rx_data);
                        PH_WV:  if (rcnt==2) sV<=$signed(rx_data);
                        PH_MM:  if (rcnt==2) sA<=$signed(rx_data);
                        PH_WO:  if (rcnt==2) sOut<=$signed(rx_data);
                        PH_FN2: if (rcnt==2) sxn<=$signed(rx_data);
                        PH_FW1: if (rcnt==2) s1c<=$signed(rx_data);
                        PH_FW3: if (rcnt==2) s3c<=$signed(rx_data);
                        PH_SS:  if (rcnt==2) ssgc<=$signed(rx_data);
                        PH_FW2: if (rcnt==2) spc<=$signed(rx_data);
                    endcase
                    if (rcnt==resp_len-1) begin
                        case (phase)
                            PH_FN:  begin phase<=PH_WQ; st<=BUILD; end
                            PH_WQ:  begin phase<=PH_WK; st<=BUILD; end
                            PH_WK:  begin phase<=PH_WV; st<=BUILD; end
                            PH_WV:  begin hidx<=0; rope_slot<=SB_Q; st<=ROPEQ; end
                            PH_MM:  begin phase<=PH_WO; st<=BUILD; end
                            PH_WO:  begin  // attention residual : XB = XB + OUTV (requantizing add)
                                    alu_aslot<=SB_XB; alu_bslot<=SB_OUTV; alu_dslot<=SB_XB;
                                    alu_op<=1'b1; alu_sa<=sxb; alu_sb<=sOut; alu_next<=2'd2;
                                    ldi<=0; st<=ALD_A; end
                            PH_FN2: begin phase<=PH_FW1; fc<=0; st<=BUILD; end
                            PH_FW1: begin phase<=PH_FW3; st<=BUILD; end
                            PH_FW3: begin phase<=PH_SS; st<=BUILD; end
                            PH_SS:  begin  // MUL : SG(Vc) * H3(Kc) -> HG(ATT)
                                    alu_aslot<=SB_Vc; alu_bslot<=SB_Kc; alu_dslot<=SB_ATT;
                                    alu_op<=1'b0; alu_sa<=ssgc; alu_sb<=s3c; alu_next<=2'd0;
                                    ldi<=0; st<=ALD_A; end
                            default: begin  // PH_FW2 : reduce P into OUTV accumulator
                                    if (fc==2'd0) begin ldi<=0; st<=FRED_COPY; end
                                    else begin alu_aslot<=SB_OUTV; alu_bslot<=SB_P; alu_dslot<=SB_OUTV;
                                               alu_op<=1'b1; alu_sa<=sov; alu_sb<=spc; alu_next<=2'd1;
                                               ldi<=0; st<=ALD_A; end
                            end
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
                    kmem[layer*TMAX*KVW + pos*KVW + ldi[5:0]] <= $signed(vfile[vidx(SB_Kc, ldi[5:0])]);
                    vmem[layer*TMAX*KVW + pos*KVW + ldi[5:0]] <= $signed(vfile[vidx(SB_Vc, ldi[5:0])]);
                    if (ldi==KVW-1) begin
                        ksh[layer*TMAX + pos]<=sK; vsh[layer*TMAX + pos]<=sV;
                        sKref<=-8'sd128; sVref<=-8'sd128; scan_i<=0; st<=SCAN;
                    end else ldi<=ldi+1;
                end
                // serial max-shift scan over positions 0..pos of the CURRENT layer (seed -128)
                SCAN: begin
                    if (ksh[layer*TMAX + scan_i] > sKref) sKref<=ksh[layer*TMAX + scan_i];
                    if (vsh[layer*TMAX + scan_i] > sVref) sVref<=vsh[layer*TMAX + scan_i];
                    if (scan_i==pos) begin mmp<=0; mmo<=0; mmv<=1'b0; phase<=PH_MM; st<=BUILD; end
                    else scan_i<=scan_i+1;
                end

                // stream XB (x + attn + ffn) to result
                DONE_ST: begin
                    result[ldi*W +: W] <= vfile[vidx(SB_XB, ldi[5:0])];
                    if (ldi==D-1) begin result_sh<=sxb; done<=1; st<=IDLE; end
                    else ldi<=ldi+1;
                end

                // ---- serialized ALU sequence (MUL for SwiGLU, ADD for W2 reduce) ----
                // load alu_a (D bytes, combinational vfile read), then alu_b, pulse start.
                ALD_A: begin alu_a[ldi*W +: W] <= vfile[vidx(alu_aslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0; st<=ALD_B; end else ldi<=ldi+1; end
                ALD_B: begin alu_b[ldi*W +: W] <= vfile[vidx(alu_bslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0; alu_start<=1'b1; st<=ALU_WAIT; end else ldi<=ldi+1; end
                ALU_WAIT: if (alu_done) begin
                        case (alu_next)
                            2'd0: shc<=alu_out_sh;    // MUL -> HG shift
                            2'd1: sov<=alu_out_sh;    // reduce -> accumulator shift
                            default: sxb<=alu_out_sh; // residual -> new running XB shift
                        endcase
                        ldi<=0; st<=AST_A;
                    end
                // serial-store alu_out into alu_dslot, then dispatch by alu_next
                AST_A: begin vaddr<=vidx(alu_dslot, ldi[5:0]); vdin<=alu_out[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0;
                            case (alu_next)
                                2'd0: begin phase<=PH_FW2; st<=BUILD; end               // MUL done -> W2
                                2'd1: if (fc==2'd2) begin                               // last reduce -> FFN residual
                                          alu_aslot<=SB_XB; alu_bslot<=SB_OUTV; alu_dslot<=SB_XB;
                                          alu_op<=1'b1; alu_sa<=sxb; alu_sb<=sov; alu_next<=2'd2;
                                          ldi<=0; st<=ALD_A;
                                      end else begin fc<=fc+1; phase<=PH_FW1; st<=BUILD; end
                                default: begin  // residual done
                                          if (!in_ffn) begin in_ffn<=1'b1; phase<=PH_FN2; st<=BUILD; end  // attn -> FFN
                                          else if (layer==NL-1) st<=DONE_ST;                              // last layer
                                          else begin  // next layer : XB carries over, KV persists
                                              layer<=layer+1; in_ffn<=1'b0; fc<=2'd0;
                                              phase<=PH_FN; ldi<=0; st<=BUILD;
                                          end
                                      end
                            endcase
                        end else ldi<=ldi+1; end
                // fc==0 : the first W2 partial simply seeds the accumulator (no add)
                FRED_COPY: begin vaddr<=vidx(SB_OUTV, ldi[5:0]); vdin<=vfile[vidx(SB_P, ldi[5:0])]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; sov<=spc; fc<=fc+1; phase<=PH_FW1; st<=BUILD; end
                        else ldi<=ldi+1; end
            endcase
        end
    end
endmodule
