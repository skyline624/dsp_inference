`timescale 1ns/1ps
// =============================================================================
// ffn_tp_seq2 - autonomous FFN-TP sequencer, LUT-lean AND NN-parameterized.
//
// Zero-PC coordinator that splits a SwiGLU FFN across NN brick nodes (tensor-
// parallel). Same LUT-lean register file (vfile) as before, but the fixed 2-node
// A/B structure is replaced by a (phase, chunk) FSM that loops each per-node op
// over chunk = 0..NN-1. NN=1 => single-card autonomous; NN>=2 => cluster. The
// hidden dim is NN*D (each node computes one D-wide chunk of the hidden layer).
//
// Phases : FN (rmsnorm broadcast) -> W1[c] -> W3[c] -> SS[c] (silu) ->
//          MUL[c] (silu*W3, local ALU) -> W2[c] (col-parallel matmul) ->
//          REDUCE (all-reduce the NN partials) -> RESID (x + out) -> done.
//
// Slot map (in vfile): 0=XB 1=XNB 2=OUTV 3=YV, then per-chunk banks of NN each:
//   H1[c]=4+c  H3[c]=4+NN+c  SG[c]=4+2NN+c  HG[c]=4+3NN+c  P[c]=4+4NN+c
// Nodes are addressed over NN internal UART lines, muxed by 'tgt' (= chunk).
// =============================================================================
module ffn_tp_seq2 #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2,                 // tensor-parallel degree (nodes)
    parameter DIV = 27,
    parameter BOOT = 40000,
    parameter A_RMS = 23'h100000,
    parameter A_W1  = 23'h101000,
    parameter A_W3  = 23'h102000,
    parameter A_W2  = 23'h103000
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire [W*D-1:0]  x_in,
    input  wire signed [7:0] sx_in,
    input  wire signed [7:0] sw_rms, sw1, sw3, sw2,
    input  wire [22:0]     base,
    output wire [NN-1:0]   n_rx,      // sequencer -> node i  (UART line)
    input  wire [NN-1:0]   n_tx,      // node i -> sequencer
    output reg  [W*D-1:0]  result,
    output reg  signed [7:0] result_sh,
    output reg             done
);
    localparam NSLOTS = 4 + 5*NN;
    localparam SW = (NSLOTS <= 2) ? 1 : $clog2(NSLOTS);   // slot-index width
    localparam CW = (NN <= 1) ? 1 : $clog2(NN);           // chunk/node-index width
    localparam AW = SW + 6;                               // D=64 -> 6 byte bits
    localparam [SW-1:0] SB_XB=0, SB_XNB=1, SB_OUTV=2, SB_YV=3;

    // per-chunk slot bases (functions of the chunk index)
    function [SW-1:0] slH1; input [7:0] i; slH1 = 4          + i; endfunction
    function [SW-1:0] slH3; input [7:0] i; slH3 = 4 +   NN   + i; endfunction
    function [SW-1:0] slSG; input [7:0] i; slSG = 4 + 2*NN   + i; endfunction
    function [SW-1:0] slHG; input [7:0] i; slHG = 4 + 3*NN   + i; endfunction
    function [SW-1:0] slP;  input [7:0] i; slP  = 4 + 4*NN   + i; endfunction

    // ---- single-port vector register file (LUT-RAM) ----
    reg [7:0] vfile [0:NSLOTS*D-1];
    reg [AW-1:0] vaddr;
    reg [7:0]  vdin;
    reg        vwe;
    function [AW-1:0] vidx; input [SW-1:0] slot; input [5:0] b; vidx = (slot<<6)|b; endfunction

    // shift registers : scalars + per-chunk banks
    reg signed [7:0] sx0, sxn, sov, syv;
    reg signed [7:0] s1 [0:NN-1];    // shift of H1[c]  (from W1)
    reg signed [7:0] s3 [0:NN-1];    // shift of H3[c]  (from W3)
    reg signed [7:0] ss [0:NN-1];    // shift of SG[c]  (from SiLU)
    reg signed [7:0] sg_[0:NN-1];    // shift of HG[c]  (from MUL, feeds W2 sx)
    reg signed [7:0] sp [0:NN-1];    // shift of P[c]   (from W2)

    // one UART host, muxed to the target node (tgt = chunk index)
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    reg  [CW-1:0] tgt;
    wire host_tx; wire host_rx;
    uart_tx_8n1 #(.DIV(DIV)) u_htx (.clk(clk), .rst(~rst_n), .data(tx_data),
                                    .send(tx_send), .tx(host_tx), .busy(tx_busy));
    wire [7:0] rx_data; wire rx_valid;
    uart_rx_8n1 #(.DIV(DIV)) u_hrx (.clk(clk), .rst(~rst_n), .rx(host_rx),
                                    .data(rx_data), .valid(rx_valid));
    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nmux
        assign n_rx[g] = (tgt==g) ? host_tx : 1'b1;
    end endgenerate
    assign host_rx = n_tx[tgt];

    // glue ALU (parallel DSP multiply + add, fed via serial staging)
    reg            alu_start, alu_op;
    reg  [W*D-1:0] alu_a, alu_b; reg signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out; wire signed [7:0] alu_out_sh; wire alu_done;
    vec_alu2 #(W, D) u_alu (.clk(clk), .rst_n(rst_n), .start(alu_start), .op(alu_op),
        .a(alu_a), .sa(alu_sa), .b(alu_b), .sb(alu_sb),
        .out(alu_out), .out_sh(alu_out_sh), .done(alu_done));

    reg [7:0] pkt [0:79];
    reg [9:0] pkt_len, resp_len, rcnt, idx;
    reg [6:0] ldi;                    // 0..D load/store index

    // phases
    localparam [3:0] PH_FN=0, PH_W1=1, PH_W3=2, PH_SS=3, PH_MUL=4, PH_W2=5,
                     PH_RED=6, PH_RES=7, PH_FIN=8;
    reg [3:0] phase;
    reg [CW-1:0] chunk;               // 0..NN-1 (per-chunk loop)
    reg [CW-1:0] rk;                  // reduce accumulation index (1..NN-1)

    reg [SW-1:0] src_slot, dst_slot, alu_aslot, alu_bslot, alu_dslot;
    reg [23:0] bcnt;

    function [7:0] a0; input [22:0] a; a0=a[7:0];   endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8];  endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction

    localparam IDLE=0,LOADX=1,BWAIT=2,BUILD=3,E_SET=4,E_BUSY=5,E_DONE=6,RECV=7,
               ALD_A=8,ALD_B=9,ALU_WAIT=10,AST_A=11,RED_COPY=12,DONE_ST=13;
    reg [3:0] st;

    // ---- per-(phase,chunk,rk) slot selection (combinational) ----
    always @(*) begin
        src_slot=SB_XB; dst_slot=SB_XNB;
        alu_aslot=SB_XB; alu_bslot=SB_OUTV; alu_dslot=SB_YV;
        case (phase)
            PH_FN:  begin src_slot=SB_XB;         dst_slot=SB_XNB;       end
            PH_W1:  begin src_slot=SB_XNB;        dst_slot=slH1(chunk);  end
            PH_W3:  begin src_slot=SB_XNB;        dst_slot=slH3(chunk);  end
            PH_SS:  begin src_slot=slH1(chunk);   dst_slot=slSG(chunk);  end
            PH_W2:  begin src_slot=slHG(chunk);   dst_slot=slP(chunk);   end
            PH_MUL: begin alu_aslot=slSG(chunk);  alu_bslot=slH3(chunk); alu_dslot=slHG(chunk); end
            PH_RED: begin alu_aslot=(rk<=1)?slP(0):SB_OUTV;
                          alu_bslot=(rk<=1)?slP(1):slP(rk);
                          alu_dslot=SB_OUTV; end
            PH_RES: begin alu_aslot=SB_XB; alu_bslot=SB_OUTV; alu_dslot=SB_YV; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; alu_start<=0; idx<=0; rcnt<=0; bcnt<=0;
            phase<=PH_FN; chunk<=0; rk<=1; tgt<=0; ldi<=0; vwe<=0;
        end else begin
            tx_send<=0; done<=0; alu_start<=0; vwe<=0;

            // single write port : commit vdin into the slot-file when vwe is set.
            if (vwe) vfile[vaddr] <= vdin;

            case (st)
                IDLE: if (start) begin sx0<=sx_in; sxn<=sx_in;
                        phase<=PH_FN; chunk<=0; rk<=1; bcnt<=0; ldi<=0; st<=LOADX; end

                // serial-load external x_in into vfile[XB]
                LOADX: begin vaddr<=vidx(SB_XB, ldi[5:0]); vdin<=x_in[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; st<=BWAIT; end else ldi<=ldi+1; end

                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (phase)
                        PH_FN: begin tgt<=0;
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                            pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                            pkt_len<=71; resp_len<=75; end
                        PH_W1: begin tgt<=chunk;
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw1;
                            pkt[69]<=a0(A_W1+base);pkt[70]<=a1(A_W1+base);pkt[71]<=a2(A_W1+base);
                            pkt_len<=72; resp_len<=3+D; end
                        PH_W3: begin tgt<=chunk;
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw3;
                            pkt[69]<=a0(A_W3+base);pkt[70]<=a1(A_W3+base);pkt[71]<=a2(A_W3+base);
                            pkt_len<=72; resp_len<=3+D; end
                        PH_SS: begin tgt<=chunk;
                            pkt[0]<="S";pkt[1]<="S";pkt[2]<=s1[chunk];
                            pkt_len<=67; resp_len<=70; end
                        default: begin // PH_W2
                            tgt<=chunk;
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sg_[chunk];pkt[4]<=sw2;
                            pkt[69]<=a0(A_W2+base);pkt[70]<=a1(A_W2+base);pkt[71]<=a2(A_W2+base);
                            pkt_len<=72; resp_len<=3+D; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                // stream packet bytes; vector bytes read COMBINATIONALLY from vfile
                E_SET: begin
                    if (phase==PH_FN && idx>=4 && idx<4+D)        tx_data<=vfile[vidx(src_slot, idx[5:0]-4)];
                    else if ((phase==PH_W1||phase==PH_W3||phase==PH_W2) && idx>=5 && idx<5+D)
                                                                  tx_data<=vfile[vidx(src_slot, idx[5:0]-5)];
                    else if (phase==PH_SS && idx>=3 && idx<3+D)   tx_data<=vfile[vidx(src_slot, idx[5:0]-3)];
                    else tx_data<=pkt[idx];
                    tx_send<=1; st<=E_BUSY;
                end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin idx<=idx+1; st<=E_SET; end
                end

                // stream response bytes; vector bytes written into vfile[dst_slot]
                RECV: if (rx_valid) begin
                    if (phase==PH_FN && rcnt>=11 && rcnt<11+D)    begin vaddr<=vidx(SB_XNB, rcnt[5:0]-11); vdin<=rx_data; vwe<=1'b1; end
                    else if ((phase==PH_W1||phase==PH_W3||phase==PH_W2) && rcnt>=3 && rcnt<3+D)
                                                                  begin vaddr<=vidx(dst_slot, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_SS && rcnt>=6 && rcnt<6+D)  begin vaddr<=vidx(dst_slot, rcnt[5:0]-6); vdin<=rx_data; vwe<=1'b1; end
                    case (phase)
                        PH_FN: if (rcnt==2) sxn<=rx_data;
                        PH_W1: if (rcnt==2) s1[chunk]<=rx_data;
                        PH_W3: if (rcnt==2) s3[chunk]<=rx_data;
                        PH_SS: if (rcnt==2) ss[chunk]<=rx_data;
                        PH_W2: if (rcnt==2) sp[chunk]<=rx_data;
                    endcase
                    if (rcnt==resp_len-1) begin
                        // advance node-op phase / chunk loop
                        case (phase)
                            PH_FN: begin phase<=PH_W1; chunk<=0; st<=BUILD; end
                            PH_W1: if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                                   else begin phase<=PH_W3; chunk<=0; st<=BUILD; end
                            PH_W3: if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                                   else begin phase<=PH_SS; chunk<=0; st<=BUILD; end
                            PH_SS: if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                                   else begin phase<=PH_MUL; chunk<=0; st<=ALD_A; end
                            default: /* PH_W2 */
                                   if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                                   else begin phase<=PH_RED; chunk<=0; rk<=1;
                                              st<= (NN==1) ? RED_COPY : ALD_A; end
                        endcase
                    end else rcnt<=rcnt+1;
                end

                // ALU: serial-load alu_a / alu_b from vfile (combinational read)
                ALD_A: begin
                        alu_a[ldi*W +: W] <= vfile[vidx(alu_aslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0; st<=ALD_B; end else ldi<=ldi+1; end
                ALD_B: begin
                        alu_b[ldi*W +: W] <= vfile[vidx(alu_bslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0;
                            alu_op<=(phase==PH_MUL)?1'b0:1'b1;
                            case (phase)
                                PH_MUL:  begin alu_sa<=ss[chunk]; alu_sb<=s3[chunk]; end
                                PH_RED:  begin alu_sa<=(rk<=1)?sp[0]:sov; alu_sb<=(rk<=1)?sp[1]:sp[rk]; end
                                default: begin alu_sa<=sx0;       alu_sb<=sov;       end // PH_RES
                            endcase
                            alu_start<=1; st<=ALU_WAIT; end else ldi<=ldi+1; end
                ALU_WAIT: if (alu_done) begin
                    case (phase)
                        PH_MUL: sg_[chunk] <= alu_out_sh;
                        PH_RED: sov        <= alu_out_sh;
                        default: syv       <= alu_out_sh; // PH_RES
                    endcase
                    ldi<=0; st<=AST_A;
                end
                // serial-store alu_out into vfile[alu_dslot]
                AST_A: begin vaddr<=vidx(alu_dslot, ldi[5:0]); vdin<=alu_out[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0;
                            case (phase)
                                PH_MUL: if (chunk!=NN-1) begin chunk<=chunk+1; st<=ALD_A; end
                                        else begin phase<=PH_W2; chunk<=0; st<=BUILD; end
                                PH_RED: if (rk==NN-1) begin phase<=PH_RES; st<=ALD_A; end
                                        else begin rk<=rk+1; st<=ALD_A; end
                                default: begin phase<=PH_FIN; st<=DONE_ST; end // PH_RES
                            endcase
                        end else ldi<=ldi+1; end

                // NN==1 : no reduce, just copy P[0] -> OUTV (serial, sov = sp[0])
                RED_COPY: begin vaddr<=vidx(SB_OUTV, ldi[5:0]); vdin<=vfile[vidx(slP(0), ldi[5:0])]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; sov<=sp[0]; phase<=PH_RES; st<=ALD_A; end
                        else ldi<=ldi+1; end

                DONE_ST: begin
                        result[ldi*W +: W] <= vfile[vidx(SB_YV, ldi[5:0])];
                        if (ldi==D-1) begin result_sh<=syv; done<=1; st<=IDLE; end else ldi<=ldi+1; end
            endcase
        end
    end
endmodule
