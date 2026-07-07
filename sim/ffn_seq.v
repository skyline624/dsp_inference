`timescale 1ns/1ps
// =============================================================================
// ffn_seq - autonomous COMPLETE FFN sequencer on a real brick node (final)
//
// On-chip "FFN microcontroller": drives the unmodified brick node over UART and
// uses vec_alu for the glue, running the whole FFN by itself (zero PC):
//
//   FN(rmsnorm) -> FQ(W1) -> FQ(W3) -> SS(silu) -> vec_alu MUL(silu.h3)
//               -> FQ(W2) -> vec_alu ADD(residual x + .)  -> y
//
// Command/handoff/shift mechanisms validated in steps 1/2; vec_alu (multiply on
// the DSP + residual) validated separately. This file assembles them.
// Weights (rms,W1,W3,W2) are in SDRAM (backdoor-loaded). hidden = D (no chunk).
// =============================================================================
module ffn_seq #(
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
    output wire            node_rx,
    input  wire            node_tx,
    output reg  [W*D-1:0]  result,
    output reg  signed [7:0] result_sh,
    output reg             done
);
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    uart_tx_8n1 #(.DIV(DIV)) u_htx (.clk(clk), .rst(~rst_n), .data(tx_data),
                                    .send(tx_send), .tx(node_rx), .busy(tx_busy));
    wire [7:0] rx_data; wire rx_valid;
    uart_rx_8n1 #(.DIV(DIV)) u_hrx (.clk(clk), .rst(~rst_n), .rx(node_tx),
                                    .data(rx_data), .valid(rx_valid));

    // glue ALU (multiply on DSP + residual add)
    reg            alu_start, alu_op;
    reg  [W*D-1:0] alu_a, alu_b;
    reg  signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out;
    wire signed [7:0] alu_out_sh;
    wire           alu_done;
    vec_alu #(W, D) u_alu (
        .clk(clk), .rst_n(rst_n), .start(alu_start), .op(alu_op),
        .a(alu_a), .sa(alu_sa), .b(alu_b), .sb(alu_sb),
        .out(alu_out), .out_sh(alu_out_sh), .done(alu_done));

    reg [7:0] pkt [0:79];
    reg [7:0] resp [0:79];
    reg [9:0] pkt_len, resp_len;

    reg [W*D-1:0]    xb, xnb, h1b, h3b, sgb, hgb, ob, yb;
    reg signed [7:0] sx0, sxn, sh1, sh3, ssg, shg, sho, sy;

    integer j;

    localparam FN=0, W1=1, W3=2, SS=3, MUL=4, W2=5, ADD=6, FINI=7;
    reg [2:0] step;

    localparam IDLE=0, BWAIT=1, BUILD=2, E_SET=3, E_BUSY=4, E_DONE=5,
               RECV=6, EXTRACT=7, ALU_GO=8, ALU_WAIT=9, DONE_ST=10;
    reg [3:0] st;
    reg [9:0] idx, rcnt;
    reg [15:0] bcnt;

    function [7:0] a0; input [22:0] a; a0 = a[7:0];   endfunction
    function [7:0] a1; input [22:0] a; a1 = a[15:8];  endfunction
    function [7:0] a2; input [22:0] a; a2 = a[22:16]; endfunction
    function is_alu; input [2:0] s; is_alu = (s==MUL) || (s==ADD); endfunction

    task next_step;                       // advance step, choose next state
        input [2:0] s;
        begin
            if (s == ADD) st <= DONE_ST;
            else begin
                step <= s + 1;
                st   <= is_alu(s+1) ? ALU_GO : BUILD;
            end
        end
    endtask

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin st<=IDLE; done<=0; tx_send<=0; alu_start<=0; idx<=0; rcnt<=0; bcnt<=0; step<=FN; end
        else begin
            tx_send<=0; done<=0; alu_start<=0;
            if (st==RECV && rx_valid) begin resp[rcnt] <= rx_data; rcnt <= rcnt + 1; end

            case (st)
                IDLE: if (start) begin
                    xb<=x_in; sx0<=sx_in; sxn<=sx_in; step<=FN; bcnt<=0; st<=BWAIT;
                end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (step)
                        FN: begin
                            pkt[0]<="F"; pkt[1]<="N"; pkt[2]<=sx0; pkt[3]<=sw_rms;
                            for (j=0;j<D;j=j+1) pkt[4+j] <= xb[j*W +: W];
                            pkt[68]<=a0(A_RMS); pkt[69]<=a1(A_RMS); pkt[70]<=a2(A_RMS);
                            pkt_len<=71; resp_len<=75;
                        end
                        W1: begin
                            pkt[0]<="F"; pkt[1]<="Q"; pkt[2]<=D[7:0]; pkt[3]<=sxn; pkt[4]<=sw1;
                            for (j=0;j<D;j=j+1) pkt[5+j] <= xnb[j*W +: W];
                            pkt[69]<=a0(A_W1); pkt[70]<=a1(A_W1); pkt[71]<=a2(A_W1);
                            pkt_len<=72; resp_len<=3+D;
                        end
                        W3: begin
                            pkt[0]<="F"; pkt[1]<="Q"; pkt[2]<=D[7:0]; pkt[3]<=sxn; pkt[4]<=sw3;
                            for (j=0;j<D;j=j+1) pkt[5+j] <= xnb[j*W +: W];
                            pkt[69]<=a0(A_W3); pkt[70]<=a1(A_W3); pkt[71]<=a2(A_W3);
                            pkt_len<=72; resp_len<=3+D;
                        end
                        SS: begin
                            pkt[0]<="S"; pkt[1]<="S"; pkt[2]<=sh1;
                            for (j=0;j<D;j=j+1) pkt[3+j] <= h1b[j*W +: W];
                            pkt_len<=67; resp_len<=70;
                        end
                        default: begin // W2
                            pkt[0]<="F"; pkt[1]<="Q"; pkt[2]<=D[7:0]; pkt[3]<=shg; pkt[4]<=sw2;
                            for (j=0;j<D;j=j+1) pkt[5+j] <= hgb[j*W +: W];
                            pkt[69]<=a0(A_W2); pkt[70]<=a1(A_W2); pkt[71]<=a2(A_W2);
                            pkt_len<=72; resp_len<=3+D;
                        end
                    endcase
                    idx<=0; st<=E_SET;
                end

                E_SET:  begin tx_data <= pkt[idx]; tx_send <= 1; st<=E_BUSY; end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx == pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin idx<=idx+1; st<=E_SET; end
                end
                RECV: if (rcnt == resp_len) st<=EXTRACT;

                EXTRACT: begin
                    case (step)
                        FN: begin for (j=0;j<D;j=j+1) xnb[j*W +: W]<=resp[11+j]; sxn<=resp[2]; end
                        W1: begin for (j=0;j<D;j=j+1) h1b[j*W +: W]<=resp[3+j];  sh1<=resp[2]; end
                        W3: begin for (j=0;j<D;j=j+1) h3b[j*W +: W]<=resp[3+j];  sh3<=resp[2]; end
                        SS: begin for (j=0;j<D;j=j+1) sgb[j*W +: W]<=resp[6+j];  ssg<=resp[2]; end
                        default: begin for (j=0;j<D;j=j+1) ob[j*W +: W]<=resp[3+j]; sho<=resp[2]; end // W2
                    endcase
                    next_step(step);
                end

                ALU_GO: begin
                    if (step==MUL) begin
                        alu_op<=1'b0; alu_a<=sgb; alu_sa<=ssg; alu_b<=h3b; alu_sb<=sh3;
                    end else begin // ADD residual
                        alu_op<=1'b1; alu_a<=xb;  alu_sa<=sx0; alu_b<=ob;  alu_sb<=sho;
                    end
                    alu_start<=1'b1; st<=ALU_WAIT;
                end
                ALU_WAIT: if (alu_done) begin
                    if (step==MUL) begin hgb<=alu_out; shg<=alu_out_sh; end
                    else           begin yb <=alu_out; sy <=alu_out_sh; end
                    next_step(step);
                end

                DONE_ST: begin result<=yb; result_sh<=sy; done<=1; st<=IDLE; end
            endcase
        end
    end
endmodule
