`timescale 1ns/1ps
// =============================================================================
// ffn_tp_seq2 - REFACTORED autonomous FFN-TP sequencer (LUT-lean)
//
// Same role as ffn_tp_seq (zero-PC coordinator driving 2 brick nodes, tensor-
// parallel FFN), but the 14 activation vectors live in a byte-wide LUT-RAM
// register file (vfile) accessed 8 bits/cycle through ONE read port and ONE
// write port (single-port RAM, driven by a unified addr/we/din/dout muxed by
// the FSM state). The step selects a SLOT INDEX (4-bit constant), not a
// 512-bit mux -> the 512-bit vector muxing that cost ~36000 LUT in v1 is gone.
// The glue ALU (vec_alu, already DSP-sequential) is fed via 64-cycle serial
// staging. Net target: sequencer LUT small enough to fit beside the node.
//
// Slot map: 0=XB 1=XNB 2=H1A 3=H1B 4=H3A 5=H3B 6=SGA 7=SGB
//           8=HGA 9=HGB 10=PA 11=PB 12=OUTV 13=YV
// =============================================================================
module ffn_tp_seq2 #(
    parameter W = 8,
    parameter D = 64,
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
    output wire            n0_rx, output wire n1_rx,
    input  wire            n0_tx, input  wire n1_tx,
    output reg  [W*D-1:0]  result,
    output reg  signed [7:0] result_sh,
    output reg             done
);
    localparam NSLOTS = 14;
    localparam [3:0] SB_XB=0, SB_XNB=1, SB_H1A=2, SB_H1B=3, SB_H3A=4, SB_H3B=5,
               SB_SGA=6, SB_SGB=7, SB_HGA=8, SB_HGB=9, SB_PA=10, SB_PB=11,
               SB_OUTV=12, SB_YV=13;
    localparam AW = 10;   // 14*64=896 entries

    // ---- single-port vector register file (LUT-RAM) ----
    reg [7:0] vfile [0:NSLOTS*D-1];
    reg [AW-1:0] vaddr;            // unified read/write address
    reg [7:0]  vdin;               // unified write data
    reg        vwe;                // unified write enable
    wire [7:0] vdout = vfile[vaddr]; // async read (LUT-RAM)
    function [AW-1:0] vidx; input [3:0] slot; input [5:0] b; vidx = {slot, b}; endfunction

    // shift registers (small FFs)
    reg signed [7:0] sx0, sxn, s1a, s1b, s3a, s3b, ssa, ssb, sga_, sgb_, spa, spb, sov, syv;

    // one UART host, muxed to the target node
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    reg  tgt;
    wire host_tx; wire host_rx;
    uart_tx_8n1 #(.DIV(DIV)) u_htx (.clk(clk), .rst(~rst_n), .data(tx_data),
                                    .send(tx_send), .tx(host_tx), .busy(tx_busy));
    wire [7:0] rx_data; wire rx_valid;
    uart_rx_8n1 #(.DIV(DIV)) u_hrx (.clk(clk), .rst(~rst_n), .rx(host_rx),
                                    .data(rx_data), .valid(rx_valid));
    assign n0_rx  = (tgt==1'b0) ? host_tx : 1'b1;
    assign n1_rx  = (tgt==1'b1) ? host_tx : 1'b1;
    assign host_rx = (tgt==1'b0) ? n0_tx : n1_tx;

    // glue ALU (parallel DSP multiply + add, fed via serial staging)
    reg            alu_start, alu_op;
    reg  [W*D-1:0] alu_a, alu_b; reg signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out; wire signed [7:0] alu_out_sh; wire alu_done;
    vec_alu2 #(W, D) u_alu (.clk(clk), .rst_n(rst_n), .start(alu_start), .op(alu_op),
        .a(alu_a), .sa(alu_sa), .b(alu_b), .sb(alu_sb),
        .out(alu_out), .out_sh(alu_out_sh), .done(alu_done));

    reg [7:0] pkt [0:79];
    reg [9:0] pkt_len, resp_len, rcnt, idx;
    reg [6:0] ldi;     // 0..D load/store index

    localparam FN=0, W1A=1,W1B=2, W3A=3,W3B=4, SSA=5,SSB=6,
               MULA=7,MULB=8, W2A=9,W2B=10, REDUCE=11, RESID=12, FINI=13;
    reg [3:0] step;
    reg [3:0] src_slot, dst_slot, alu_aslot, alu_bslot, alu_dslot;

    localparam IDLE=0,LOADX=1,BWAIT=2,BUILD=3,E_SET=4,E_BUSY=5,E_DONE=6,RECV=7,
               ALD_A=8,ALD_B=9,ALU_GO=10,ALU_WAIT=11,AST_A=12,DONE_ST=13;
    reg [3:0] st;
    reg [23:0] bcnt;

    function [7:0] a0; input [22:0] a; a0=a[7:0];   endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8];  endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction
    function is_alu; input [3:0] s; is_alu=(s==MULA)||(s==MULB)||(s==REDUCE)||(s==RESID); endfunction
    task adv; input [3:0] s; begin
        if (s==RESID) st<=DONE_ST;
        else begin step<=s+1; st<= is_alu(s+1) ? ALD_A : BUILD; end
    end endtask

    // per-step slot selection (combinational constants)
    always @(*) begin
        src_slot=0; dst_slot=0; alu_aslot=0; alu_bslot=0; alu_dslot=0;
        case (step)
            FN:    begin src_slot=SB_XB;  dst_slot=SB_XNB; end
            W1A:   begin src_slot=SB_XNB; dst_slot=SB_H1A; end
            W1B:   begin src_slot=SB_XNB; dst_slot=SB_H1B; end
            W3A:   begin src_slot=SB_XNB; dst_slot=SB_H3A; end
            W3B:   begin src_slot=SB_XNB; dst_slot=SB_H3B; end
            SSA:   begin src_slot=SB_H1A; dst_slot=SB_SGA; end
            SSB:   begin src_slot=SB_H1B; dst_slot=SB_SGB; end
            W2A:   begin src_slot=SB_HGA; dst_slot=SB_PA;  end
            W2B:   begin src_slot=SB_HGB; dst_slot=SB_PB;  end
            MULA:  begin alu_aslot=SB_SGA; alu_bslot=SB_H3A; alu_dslot=SB_HGA; end
            MULB:  begin alu_aslot=SB_SGB; alu_bslot=SB_H3B; alu_dslot=SB_HGB; end
            REDUCE:begin alu_aslot=SB_PA;  alu_bslot=SB_PB;  alu_dslot=SB_OUTV;end
            RESID: begin alu_aslot=SB_XB;  alu_bslot=SB_OUTV;alu_dslot=SB_YV; end
        endcase
    end

    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=IDLE; done<=0; tx_send<=0; alu_start<=0; idx<=0; rcnt<=0; bcnt<=0; step<=FN; tgt<=0; ldi<=0; vwe<=0; end
        else begin
            tx_send<=0; done<=0; alu_start<=0; vwe<=0;

            // single write port : commit vdin into the slot-file when vwe is set.
            // (vaddr/vdin/vwe are driven by the FSM states below.)
            if (vwe) vfile[vaddr] <= vdin;

            case (st)
                IDLE: if (start) begin sx0<=sx_in; sxn<=sx_in; step<=FN; bcnt<=0; ldi<=0; st<=LOADX; end

                // serial-load external x_in into vfile[SB_XB]
                LOADX: begin vaddr<=vidx(SB_XB, ldi[5:0]); vdin<=x_in[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; st<=BWAIT; end else ldi<=ldi+1; end

                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (step)
                        FN: begin tgt<=0;
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                            pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                            pkt_len<=71; resp_len<=75; end
                        W1A,W1B,W3A,W3B: begin tgt<=(step==W1B||step==W3B);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;
                            pkt[4]<= (step==W3A||step==W3B) ? sw3 : sw1;
                            if (step==W1A||step==W1B) begin pkt[69]<=a0(A_W1+base);pkt[70]<=a1(A_W1+base);pkt[71]<=a2(A_W1+base); end
                            else begin pkt[69]<=a0(A_W3+base);pkt[70]<=a1(A_W3+base);pkt[71]<=a2(A_W3+base); end
                            pkt_len<=72; resp_len<=3+D; end
                        SSA,SSB: begin tgt<=(step==SSB);
                            pkt[0]<="S";pkt[1]<="S";pkt[2]<= (step==SSA)?s1a:s1b;
                            pkt_len<=67; resp_len<=70; end
                        default: begin // W2A, W2B
                            tgt<=(step==W2B);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];
                            pkt[3]<= (step==W2A)?sga_:sgb_; pkt[4]<=sw2;
                            pkt[69]<=a0(A_W2+base);pkt[70]<=a1(A_W2+base);pkt[71]<=a2(A_W2+base);
                            pkt_len<=72; resp_len<=3+D; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                // stream packet bytes; vector bytes read from vfile[src_slot] on the fly
                // vector bytes are read COMBINATIONALLY at the current index (LUT-RAM
                // async read) : reading vfile[] directly avoids the 1-cycle latency of
                // the registered vaddr (which caused an off-by-one rotation of the vector).
                E_SET: begin
                    if (step==FN && idx>=4 && idx<68)             tx_data<=vfile[vidx(SB_XB,  idx[5:0]-4)];
                    else if ((step==W1A||step==W1B||step==W3A||step==W3B) && idx>=5 && idx<69)
                                                                 tx_data<=vfile[vidx(src_slot, idx[5:0]-5)];
                    else if (step==SSA && idx>=3 && idx<67)        tx_data<=vfile[vidx(SB_H1A, idx[5:0]-3)];
                    else if (step==SSB && idx>=3 && idx<67)        tx_data<=vfile[vidx(SB_H1B, idx[5:0]-3)];
                    else if ((step==W2A||step==W2B) && idx>=5 && idx<69)
                                                                 tx_data<=vfile[vidx(src_slot, idx[5:0]-5)];
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
                    if (step==FN && rcnt>=11 && rcnt<11+D)        begin vaddr<=vidx(SB_XNB, rcnt[5:0]-11); vdin<=rx_data; vwe<=1'b1; end
                    else if ((step==W1A||step==W1B||step==W3A||step==W3B||step==W2A||step==W2B) && rcnt>=3 && rcnt<3+D)
                                                                 begin vaddr<=vidx(dst_slot, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    else if ((step==SSA||step==SSB) && rcnt>=6 && rcnt<6+D)
                                                                 begin vaddr<=vidx(dst_slot, rcnt[5:0]-6); vdin<=rx_data; vwe<=1'b1; end
                    case (step)
                        FN:  if (rcnt==2) sxn<=rx_data;
                        W1A: if (rcnt==2) s1a<=rx_data; W1B: if (rcnt==2) s1b<=rx_data;
                        W3A: if (rcnt==2) s3a<=rx_data; W3B: if (rcnt==2) s3b<=rx_data;
                        SSA: if (rcnt==2) ssa<=rx_data; SSB: if (rcnt==2) ssb<=rx_data;
                        W2A: if (rcnt==2) spa<=rx_data; W2B: if (rcnt==2) spb<=rx_data;
                    endcase
                    if (rcnt==resp_len-1) adv(step); else rcnt<=rcnt+1;
                end

                // ALU: serial-load alu_a from vfile[alu_aslot] (combinational read)
                ALD_A: begin
                        alu_a[ldi*W +: W] <= vfile[vidx(alu_aslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0; st<=ALD_B; end else ldi<=ldi+1; end
                ALD_B: begin
                        alu_b[ldi*W +: W] <= vfile[vidx(alu_bslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0;
                            alu_op<=(step==MULA||step==MULB)?1'b0:1'b1;
                            alu_sa<=(step==MULA)?ssa:(step==MULB)?ssb:(step==REDUCE)?spa:sx0;
                            alu_sb<=(step==MULA)?s3a:(step==MULB)?s3b:(step==REDUCE)?spb:sov;
                            alu_start<=1; st<=ALU_WAIT; end else ldi<=ldi+1; end
                ALU_WAIT: if (alu_done) begin
                    case (step)
                        MULA: sga_<=alu_out_sh; MULB: sgb_<=alu_out_sh;
                        REDUCE: sov<=alu_out_sh; RESID: syv<=alu_out_sh;
                    endcase
                    ldi<=0; st<=AST_A;
                end
                // serial-store alu_out into vfile[alu_dslot]
                AST_A: begin vaddr<=vidx(alu_dslot, ldi[5:0]); vdin<=alu_out[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; adv(step); end else ldi<=ldi+1; end

                DONE_ST: begin
                        result[ldi*W +: W] <= vfile[vidx(SB_YV, ldi[5:0])];
                        if (ldi==D-1) begin result_sh<=syv; done<=1; st<=IDLE; end else ldi<=ldi+1; end
            endcase
        end
    end
endmodule