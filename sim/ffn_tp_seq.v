`timescale 1ns/1ps
// =============================================================================
// ffn_tp_seq - autonomous TENSOR-PARALLEL FFN sequencer over 2 brick nodes
//
// On-chip coordinator (zero PC) driving 2 unmodified nodes via one UART host
// muxed by `tgt`, splitting the FFN across them (hidden=128 -> 64/node):
//   FN(rmsnorm) on n0 -> xn (broadcast)
//   W1/W3 row-parallel : n0 = rows[0:64], n1 = rows[64:128]
//   silu + multiply (vec_alu) per node's hidden slice
//   W2 column-parallel : n0 = cols[0:64], n1 = cols[64:128] -> partials
//   all-reduce (vec_alu ADD of partials) -> out ;  residual (vec_alu ADD) -> y
// Weights split across the two SDRAMs (backdoor). Each node only ever holds/
// computes its slice; only activations move between the coordinator and nodes.
// =============================================================================
module ffn_tp_seq #(
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

    // glue ALU (multiply on DSP + add)
    reg            alu_start, alu_op;
    reg  [W*D-1:0] alu_a, alu_b; reg signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out; wire signed [7:0] alu_out_sh; wire alu_done;
    vec_alu #(W, D) u_alu (.clk(clk), .rst_n(rst_n), .start(alu_start), .op(alu_op),
        .a(alu_a), .sa(alu_sa), .b(alu_b), .sb(alu_sb),
        .out(alu_out), .out_sh(alu_out_sh), .done(alu_done));

    reg [7:0] pkt [0:79];
    reg [7:0] resp [0:79];
    reg [9:0] pkt_len, resp_len;

    reg [W*D-1:0]    xb, xnb, h1a,h1b, h3a,h3b, sga,sgb, hga,hgb, pa,pb, outv, yv;
    reg signed [7:0] sx0, sxn, s1a,s1b, s3a,s3b, ssa,ssb, sga_,sgb_, spa,spb, sov, syv;
    integer j;

    // steps
    localparam FN=0, W1A=1,W1B=2, W3A=3,W3B=4, SSA=5,SSB=6,
               MULA=7,MULB=8, W2A=9,W2B=10, REDUCE=11, RESID=12, FINI=13;
    reg [3:0] step;

    localparam IDLE=0,BWAIT=1,BUILD=2,E_SET=3,E_BUSY=4,E_DONE=5,RECV=6,EXTRACT=7,
               ALU_GO=8,ALU_WAIT=9,DONE_ST=10;
    reg [3:0] st;
    reg [9:0] idx, rcnt; reg [23:0] bcnt;   // wide enough for large BOOT waits (SD-boot nodes)

    function [7:0] a0; input [22:0] a; a0=a[7:0];   endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8];  endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction
    function is_alu; input [3:0] s; is_alu=(s==MULA)||(s==MULB)||(s==REDUCE)||(s==RESID); endfunction

    task adv; input [3:0] s; begin
        if (s==RESID) st<=DONE_ST;
        else begin step<=s+1; st<= is_alu(s+1) ? ALU_GO : BUILD; end
    end endtask

    // build an FQ packet from a given input vector / shift / addr (helper via task-like inline)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=IDLE; done<=0; tx_send<=0; alu_start<=0; idx<=0; rcnt<=0; bcnt<=0; step<=FN; tgt<=0; end
        else begin
            tx_send<=0; done<=0; alu_start<=0;
            if (st==RECV && rx_valid) begin resp[rcnt]<=rx_data; rcnt<=rcnt+1; end

            case (st)
                IDLE: if (start) begin xb<=x_in; sx0<=sx_in; sxn<=sx_in; step<=FN; bcnt<=0; st<=BWAIT; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (step)
                        FN: begin tgt<=0;
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sx0;pkt[3]<=sw_rms;
                            for(j=0;j<D;j=j+1) pkt[4+j]<=xb[j*W +: W];
                            pkt[68]<=a0(A_RMS+base);pkt[69]<=a1(A_RMS+base);pkt[70]<=a2(A_RMS+base);
                            pkt_len<=71; resp_len<=75; end
                        W1A,W1B,W3A,W3B: begin
                            tgt<= (step==W1B||step==W3B);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;
                            pkt[4]<= (step==W3A||step==W3B) ? sw3 : sw1;
                            for(j=0;j<D;j=j+1) pkt[5+j]<=xnb[j*W +: W];
                            if (step==W1A||step==W1B) begin pkt[69]<=a0(A_W1+base);pkt[70]<=a1(A_W1+base);pkt[71]<=a2(A_W1+base); end
                            else begin pkt[69]<=a0(A_W3+base);pkt[70]<=a1(A_W3+base);pkt[71]<=a2(A_W3+base); end
                            pkt_len<=72; resp_len<=3+D; end
                        SSA,SSB: begin tgt<=(step==SSB);
                            pkt[0]<="S";pkt[1]<="S";pkt[2]<= (step==SSA)?s1a:s1b;
                            for(j=0;j<D;j=j+1) pkt[3+j]<= (step==SSA)?h1a[j*W +: W]:h1b[j*W +: W];
                            pkt_len<=67; resp_len<=70; end
                        default: begin // W2A, W2B
                            tgt<=(step==W2B);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];
                            pkt[3]<= (step==W2A)?sga_:sgb_; pkt[4]<=sw2;
                            for(j=0;j<D;j=j+1) pkt[5+j]<= (step==W2A)?hga[j*W +: W]:hgb[j*W +: W];
                            pkt[69]<=a0(A_W2+base);pkt[70]<=a1(A_W2+base);pkt[71]<=a2(A_W2+base);
                            pkt_len<=72; resp_len<=3+D; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                E_SET:  begin tx_data<=pkt[idx]; tx_send<=1; st<=E_BUSY; end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin idx<=idx+1; st<=E_SET; end
                end
                RECV: if (rcnt==resp_len) st<=EXTRACT;

                EXTRACT: begin
                    case (step)
                        FN:  begin for(j=0;j<D;j=j+1) xnb[j*W +: W]<=resp[11+j]; sxn<=resp[2]; end
                        W1A: begin for(j=0;j<D;j=j+1) h1a[j*W +: W]<=resp[3+j];  s1a<=resp[2]; end
                        W1B: begin for(j=0;j<D;j=j+1) h1b[j*W +: W]<=resp[3+j];  s1b<=resp[2]; end
                        W3A: begin for(j=0;j<D;j=j+1) h3a[j*W +: W]<=resp[3+j];  s3a<=resp[2]; end
                        W3B: begin for(j=0;j<D;j=j+1) h3b[j*W +: W]<=resp[3+j];  s3b<=resp[2]; end
                        SSA: begin for(j=0;j<D;j=j+1) sga[j*W +: W]<=resp[6+j];  ssa<=resp[2]; end
                        SSB: begin for(j=0;j<D;j=j+1) sgb[j*W +: W]<=resp[6+j];  ssb<=resp[2]; end
                        W2A: begin for(j=0;j<D;j=j+1) pa[j*W +: W]<=resp[3+j];   spa<=resp[2]; end
                        default: begin for(j=0;j<D;j=j+1) pb[j*W +: W]<=resp[3+j]; spb<=resp[2]; end // W2B
                    endcase
                    adv(step);
                end

                ALU_GO: begin
                    case (step)
                        MULA: begin alu_op<=0; alu_a<=sga; alu_sa<=ssa; alu_b<=h3a; alu_sb<=s3a; end
                        MULB: begin alu_op<=0; alu_a<=sgb; alu_sa<=ssb; alu_b<=h3b; alu_sb<=s3b; end
                        REDUCE: begin alu_op<=1; alu_a<=pa; alu_sa<=spa; alu_b<=pb; alu_sb<=spb; end
                        default: begin alu_op<=1; alu_a<=xb; alu_sa<=sx0; alu_b<=outv; alu_sb<=sov; end // RESID
                    endcase
                    alu_start<=1; st<=ALU_WAIT;
                end
                ALU_WAIT: if (alu_done) begin
                    case (step)
                        MULA: begin hga<=alu_out; sga_<=alu_out_sh; end
                        MULB: begin hgb<=alu_out; sgb_<=alu_out_sh; end
                        REDUCE: begin outv<=alu_out; sov<=alu_out_sh; end
                        default: begin yv<=alu_out; syv<=alu_out_sh; end // RESID
                    endcase
                    adv(step);
                end

                DONE_ST: begin result<=yv; result_sh<=syv; done<=1; st<=IDLE; end
            endcase
        end
    end
endmodule
