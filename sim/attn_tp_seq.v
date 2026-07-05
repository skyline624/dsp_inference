`timescale 1ns/1ps
// =============================================================================
// attn_tp_seq - autonomous HEAD-PARALLEL attention, LUT-lean AND NN-parameterized.
//
// Zero-PC coordinator, pos=0/T=1 (rope=identity, attn core = GQA V-passthrough,
// still run through the real MM on node 0). The fixed 2-node structure is
// generalized to NN nodes : Wq/Wk/Wv/Wo loop over chunk=0..NN-1, and the gathers
// concatenate the NN slices via a cascade of shift-aligned padded ADDs (vec_alu2).
// NN=1 => single-card autonomous ; NN in {1,2,4} keep KPN>=HS for GQA.
//
//   FN(rmsnorm) on node 0 -> xn (broadcast)
//   Wq/Wk/Wv chunk c : node c computes its QPN/KPN rows
//   gather Q/K/V : concat the NN chunks (padded ADD cascade)
//   MM (central, real) on node 0 -> attn[64]
//   Wo chunk c : node c computes its OPN rows ; gather -> out
//   residual (ADD) -> y
// H=8, KH=4, HS=8 (GQA n_rep=2). Q rows 64 (QPN/node), K/V rows 32 (KPN/node).
// =============================================================================
module attn_tp_seq #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2,
    parameter DIV = 27,
    parameter BOOT = 40000,
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
    input  wire [22:0]     base,
`ifdef LINK_SS
    output wire [8*NN-1:0] lk_cmd_data,
    output wire [NN-1:0]   lk_cmd_wr,
    input  wire [NN-1:0]   lk_cmd_full,
    input  wire [8*NN-1:0] lk_resp_data,
    input  wire [NN-1:0]   lk_resp_empty,
    output wire [NN-1:0]   lk_resp_rd,
`else
    output wire [NN-1:0]   n_rx,
    input  wire [NN-1:0]   n_tx,
`endif
    output reg  [W*D-1:0]  result,
    output reg  signed [7:0] result_sh,
    output reg             done
);
    localparam QPN = 64 / NN;   // query rows per node  (H*HS / NN)
    localparam KPN = 32 / NN;   // kv rows per node     (KH*HS / NN)
    localparam OPN = 64 / NN;   // Wo output rows per node
    localparam HS  = 8, NREP = 2;
    localparam CW = (NN <= 1) ? 1 : $clog2(NN);

    reg [7:0] tx_data; reg tx_send; wire tx_busy; reg [CW-1:0] tgt;
    wire [7:0] rx_data; wire rx_valid;
    wire rx_rdy;
    genvar g;
`ifdef LINK_SS
    wire [7:0] htx_data; wire htx_wr; wire htx_full;
    wire       hrx_rd;   wire [7:0] hrx_data; wire hrx_empty;
    tx8_link u_htx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                    .busy(tx_busy), .o_data(htx_data), .o_wr(htx_wr), .i_full(htx_full));
    rx8_link u_hrx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                    .rdy(rx_rdy), .i_data(hrx_data), .i_empty(hrx_empty), .o_rd(hrx_rd));
    generate for (g=0; g<NN; g=g+1) begin: lmux
        assign lk_cmd_data[8*g +: 8] = htx_data;
        assign lk_cmd_wr[g]  = (tgt==g) ? htx_wr : 1'b0;
        assign lk_resp_rd[g] = (tgt==g) ? hrx_rd : 1'b0;
    end endgenerate
    assign htx_full  = lk_cmd_full[tgt];
    assign hrx_data  = lk_resp_data[8*tgt +: 8];
    assign hrx_empty = lk_resp_empty[tgt];
`else
    wire host_tx, host_rx;
    uart_tx_8n1 #(.DIV(DIV)) u_htx(.clk(clk),.rst(~rst_n),.data(tx_data),.send(tx_send),.tx(host_tx),.busy(tx_busy));
    uart_rx_8n1 #(.DIV(DIV)) u_hrx(.clk(clk),.rst(~rst_n),.rx(host_rx),.data(rx_data),.valid(rx_valid));
    generate for (g=0; g<NN; g=g+1) begin: nmux
        assign n_rx[g] = (tgt==g) ? host_tx : 1'b1;
    end endgenerate
    assign host_rx = n_tx[tgt];
`endif

    reg alu_start, alu_op; reg [W*D-1:0] alu_a, alu_b; reg signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out; wire signed [7:0] alu_out_sh; wire alu_done;
    vec_alu2 #(W,D) u_alu(.clk(clk),.rst_n(rst_n),.start(alu_start),.op(alu_op),
        .a(alu_a),.sa(alu_sa),.b(alu_b),.sb(alu_sb),.out(alu_out),.out_sh(alu_out_sh),.done(alu_done));

    reg [7:0] pkt[0:159]; reg [7:0] resp[0:79]; reg [9:0] pkt_len, resp_len;

    // per-chunk banks (NN slices) for Q/K/V/Wo results
    reg [W*D-1:0] xb, xnb, Qb, Kb, Vb, attnb, outb, yb;
    reg [W*QPN-1:0] qc [0:NN-1];   // Wq slice per node
    reg [W*KPN-1:0] kc [0:NN-1];   // Wk slice per node
    reg [W*KPN-1:0] vc [0:NN-1];   // Wv slice per node
    reg [W*OPN-1:0] oc [0:NN-1];   // Wo slice per node
    reg signed [7:0] sqc[0:NN-1], skc[0:NN-1], svc[0:NN-1], soc[0:NN-1];
    reg signed [7:0] sx0, sxn, sQ, sK, sV, sa_, sout, syv, sacc;
    integer j;

    // phases
    localparam [3:0] PH_FN=0, PH_WQ=1, PH_WK=2, PH_WV=3,
                     PH_GQ=4, PH_GK=5, PH_GV=6, PH_MM=7,
                     PH_WO=8, PH_GO=9, PH_RES=10, PH_FIN=11;
    reg [3:0] phase;
    reg [CW-1:0] chunk;    // 0..NN-1
    reg [CW-1:0] gk;       // gather accumulation index

    localparam IDLE=0,BWAIT=1,BUILD=2,E_SET=3,E_BUSY=4,E_DONE=5,RECV=6,EXTRACT=7,
               ALU_GO=8,ALU_WAIT=9,DONE_ST=10;
    reg [3:0] st; reg [9:0] idx, rcnt; reg [15:0] bcnt;

    function [7:0] a0; input [22:0] a; a0=a[7:0]; endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8]; endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction

`ifdef LINK_SS
    // pull one response byte per acknowledge (rx8_link fires on a rising edge of
    // rdy) : drop rdy for one cycle after each valid pulse. Mirrors the FFN seq.
    reg rx_ack;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rx_ack <= 1'b1; else rx_ack <= ~rx_valid;
    assign rx_rdy = (st == RECV) & rx_ack;
`endif

    // build an FQ packet (header 'F''Q' N sx sw, x[64], addr[3]) -> len 72, resp 3+N
    task build_fq; input [7:0] n; input signed [7:0] sxf, swf; input [22:0] addr; input [W*D-1:0] data;
        begin
            pkt[0]<="F"; pkt[1]<="Q"; pkt[2]<=n; pkt[3]<=sxf; pkt[4]<=swf;
            for (j=0;j<D;j=j+1) pkt[5+j] <= data[j*W +: W];
            pkt[69]<=a0(addr); pkt[70]<=a1(addr); pkt[71]<=a2(addr);
            pkt_len<=72; resp_len<=3+n;
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=IDLE; done<=0; tx_send<=0; alu_start<=0; idx<=0; rcnt<=0;
                          bcnt<=0; phase<=PH_FN; chunk<=0; gk<=0; tgt<=0; end
        else begin
            tx_send<=0; done<=0; alu_start<=0;
            if (st==RECV && rx_valid) begin resp[rcnt]<=rx_data; rcnt<=rcnt+1; end
            case (st)
                IDLE: if (start) begin xb<=x_in; sx0<=sx_in; sxn<=sx_in;
                        phase<=PH_FN; chunk<=0; gk<=0; bcnt<=0; st<=BWAIT; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (phase)
                        PH_FN: begin tgt<=0;
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                            for(j=0;j<D;j=j+1) pkt[4+j]<=xb[j*W +: W];
                            pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                            pkt_len<=71; resp_len<=75; end
                        PH_WQ: begin tgt<=chunk; build_fq(QPN[7:0], sxn, swq, A_WQ+base, xnb); end
                        PH_WK: begin tgt<=chunk; build_fq(KPN[7:0], sxn, swk, A_WK+base, xnb); end
                        PH_WV: begin tgt<=chunk; build_fq(KPN[7:0], sxn, swv, A_WV+base, xnb); end
                        PH_MM: begin tgt<=0;
                            pkt[0]<="M";pkt[1]<="M";pkt[2]<=sQ;pkt[3]<=sK;pkt[4]<=sV;pkt[5]<=8'd1;
                            for(j=0;j<D;j=j+1)  pkt[6+j]   <= Qb[j*W +: W];   // Q[64]
                            for(j=0;j<32;j=j+1) pkt[70+j]  <= Kb[j*W +: W];   // K[32]
                            for(j=0;j<32;j=j+1) pkt[102+j] <= Vb[j*W +: W];   // V[32]
                            pkt_len<=134; resp_len<=67; end
                        default: begin tgt<=chunk; build_fq(OPN[7:0], sa_, swo, A_WO+base, attnb); end // PH_WO
                    endcase
                    idx<=0; st<=E_SET;
                end

                E_SET:  begin tx_data<=pkt[idx]; tx_send<=1; st<=E_BUSY; end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end else begin idx<=idx+1; st<=E_SET; end
                end
                RECV: if (rcnt==resp_len) st<=EXTRACT;

                EXTRACT: begin
                    case (phase)
                        PH_FN:  begin for(j=0;j<D;j=j+1) xnb[j*W +: W]<=resp[11+j]; sxn<=resp[2]; end
                        PH_WQ:  begin for(j=0;j<QPN;j=j+1) qc[chunk][j*W +: W]<=resp[3+j]; sqc[chunk]<=resp[2]; end
                        PH_WK:  begin for(j=0;j<KPN;j=j+1) kc[chunk][j*W +: W]<=resp[3+j]; skc[chunk]<=resp[2]; end
                        PH_WV:  begin for(j=0;j<KPN;j=j+1) vc[chunk][j*W +: W]<=resp[3+j]; svc[chunk]<=resp[2]; end
                        PH_MM:  begin for(j=0;j<D;j=j+1) attnb[j*W +: W]<=resp[3+j]; sa_<=sV; end
                        default:begin for(j=0;j<OPN;j=j+1) oc[chunk][j*W +: W]<=resp[3+j]; soc[chunk]<=resp[2]; end // PH_WO
                    endcase
                    // advance phase / chunk loop
                    case (phase)
                        PH_FN: begin phase<=PH_WQ; chunk<=0; st<=BUILD; end
                        PH_WQ: if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                               else begin phase<=PH_WK; chunk<=0; st<=BUILD; end
                        PH_WK: if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                               else begin phase<=PH_WV; chunk<=0; st<=BUILD; end
                        PH_WV: if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                               else begin phase<=PH_GQ; gk<=0; st<=ALU_GO; end
                        PH_MM: begin phase<=PH_WO; chunk<=0; st<=BUILD; end
                        default: /* PH_WO */
                               if (chunk!=NN-1) begin chunk<=chunk+1; st<=BUILD; end
                               else begin phase<=PH_GO; gk<=0; st<=ALU_GO; end
                    endcase
                end

                // gather = concat NN chunks via a cascade of padded shift-aligned ADDs.
                // gk=0 : SEED the accumulator with chunk 0 at offset 0 (alu_b = 0, so
                //        the ADD is chunk0 + 0). gk>=1 : accumulator + chunk gk placed
                //        at offset gk*PN. This avoids double-counting chunk 0.
                ALU_GO: begin
                    alu_op<=1;
                    case (phase)
                        PH_GQ: begin
                            for(j=0;j<D;j=j+1) alu_a[j*W +: W] <= (gk==0) ? ((j<QPN)?qc[0][j*W +: W]:8'd0) : Qb[j*W +: W];
                            for(j=0;j<D;j=j+1) alu_b[j*W +: W] <= (gk!=0 && j>=gk*QPN && j<(gk+1)*QPN) ? qc[gk][(j-gk*QPN)*W +: W] : 8'd0;
                            alu_sa<=(gk==0)?sqc[0]:sQ; alu_sb<=(gk==0)?sqc[0]:sqc[gk];
                        end
                        PH_GK: begin
                            for(j=0;j<D;j=j+1) alu_a[j*W +: W] <= (gk==0) ? ((j<KPN)?kc[0][j*W +: W]:8'd0) : Kb[j*W +: W];
                            for(j=0;j<D;j=j+1) alu_b[j*W +: W] <= (gk!=0 && j>=gk*KPN && j<(gk+1)*KPN) ? kc[gk][(j-gk*KPN)*W +: W] : 8'd0;
                            alu_sa<=(gk==0)?skc[0]:sK; alu_sb<=(gk==0)?skc[0]:skc[gk];
                        end
                        PH_GV: begin
                            for(j=0;j<D;j=j+1) alu_a[j*W +: W] <= (gk==0) ? ((j<KPN)?vc[0][j*W +: W]:8'd0) : Vb[j*W +: W];
                            for(j=0;j<D;j=j+1) alu_b[j*W +: W] <= (gk!=0 && j>=gk*KPN && j<(gk+1)*KPN) ? vc[gk][(j-gk*KPN)*W +: W] : 8'd0;
                            alu_sa<=(gk==0)?svc[0]:sV; alu_sb<=(gk==0)?svc[0]:svc[gk];
                        end
                        PH_GO: begin
                            for(j=0;j<D;j=j+1) alu_a[j*W +: W] <= (gk==0) ? ((j<OPN)?oc[0][j*W +: W]:8'd0) : outb[j*W +: W];
                            for(j=0;j<D;j=j+1) alu_b[j*W +: W] <= (gk!=0 && j>=gk*OPN && j<(gk+1)*OPN) ? oc[gk][(j-gk*OPN)*W +: W] : 8'd0;
                            alu_sa<=(gk==0)?soc[0]:sout; alu_sb<=(gk==0)?soc[0]:soc[gk];
                        end
                        default: begin alu_a<=xb; alu_sa<=sx0; alu_b<=outb; alu_sb<=sout; end // PH_RES
                    endcase
                    alu_start<=1; st<=ALU_WAIT;
                end
                ALU_WAIT: if (alu_done) begin
                    case (phase)
                        PH_GQ: begin Qb<=alu_out; sQ<=alu_out_sh; end
                        PH_GK: begin Kb<=alu_out; sK<=alu_out_sh; end
                        PH_GV: begin Vb<=alu_out; sV<=alu_out_sh; end
                        PH_GO: begin outb<=alu_out; sout<=alu_out_sh; end
                        default: begin yb<=alu_out; syv<=alu_out_sh; end // PH_RES
                    endcase
                    // gather loop : fold NN chunks (gk = 0 seeds chunk0, then 1..NN-1).
                    // NN==1 : one pass (gk goes 0 -> done) is enough.
                    case (phase)
                        PH_GQ,PH_GK,PH_GV,PH_GO: begin
                            if (gk == NN-1) begin
                                // gather done : advance to next phase
                                case (phase)
                                    PH_GQ: begin phase<=PH_GK; gk<=0; st<=ALU_GO; end
                                    PH_GK: begin phase<=PH_GV; gk<=0; st<=ALU_GO; end
                                    PH_GV: begin phase<=PH_MM; st<=BUILD; end
                                    default: begin phase<=PH_RES; st<=ALU_GO; end // PH_GO
                                endcase
                            end else begin gk<=gk+1; st<=ALU_GO; end
                        end
                        default: begin phase<=PH_FIN; st<=DONE_ST; end // PH_RES
                    endcase
                end

                DONE_ST: begin result<=yb; result_sh<=syv; done<=1; st<=IDLE; end
            endcase
        end
    end
endmodule
