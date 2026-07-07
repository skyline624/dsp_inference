`timescale 1ns/1ps
// =============================================================================
// tp_node - fully autonomous tensor-parallel matmul node (real mini-GG, inc.3.1)
//
// Combines the two validated pieces into ONE autonomous FSM, no PC:
//   MATMUL : y_slice[NR] = W[NR,K] @ x[K]   (real DSP mac18 + fq_ref requantize)
//   RING   : all-gather the slices over the ring (N-1 rounds)
//   => every node ends holding the full output y[NN*NR].
//
// x is broadcast to every node; weights stay local (preloaded, never cross the
// link). Ring = 1 in + 1 out link/node -> scales to any NN.
// =============================================================================
module tp_node #(
    parameter W  = 8,
    parameter K  = 64,    // input dim
    parameter NR = 16,    // output rows computed by THIS node (slice size)
    parameter NN = 4,     // number of nodes
    parameter ID = 0
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    input  wire [W*K-1:0]    my_x,           // broadcast input activation
    // ring outgoing (to node ID+1)
    output wire [W-1:0]      o_data,
    output wire              o_send,
    input  wire              o_full,
    // ring incoming (from node ID-1)
    output wire              i_rd,
    input  wire [W-1:0]      i_data,
    input  wire              i_empty,
    output reg  [W*NR*NN-1:0] full_y,        // assembled full output
    output reg               done
);
    localparam SB  = W*NR;
    localparam LAT = 4;

    reg signed [7:0] wmem [0:NR*K-1];        // local weight slice (backdoor-loaded)

    // real DSP
    reg  signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));
    reg signed [31:0] yi32 [0:NR-1];

    // ring links
    reg  [7:0]    cur_send;
    wire [SB-1:0] slice_tx = full_y[cur_send*SB +: SB];
    reg           send_start; wire send_busy;
    link_send #(W, NR) u_tx (
        .clk(clk), .rst_n(rst_n), .start(send_start), .vec(slice_tx),
        .busy(send_busy), .tx_data(o_data), .tx_send(o_send), .tx_full(o_full));
    wire [SB-1:0] rvec; wire rdone;
    link_recv #(W, NR) u_rx (
        .clk(clk), .rst_n(rst_n), .rx_rd(i_rd), .rx_data(i_data),
        .rx_empty(i_empty), .vec(rvec), .done(rdone));
    reg [SB-1:0] rhold; reg rgot;

    localparam IDLE=0, MAC=1, FLUSH=2, MAXF=3, REQ=4,
               RSEND=5, RSBUSY=6, RSWAIT=7, RRECV=8, RNEXT=9, FIN=10;
    reg [3:0]  st;
    reg [9:0]  r, k, fcnt;
    reg [7:0]  round, recv_slot;
    reg signed [31:0] max_abs;
    reg [5:0]  sh;
    integer i;
    reg signed [31:0] av, rounded, shifted;

    function [5:0] msb_idx;
        input [31:0] v; integer b;
        begin msb_idx = 0; for (b=0;b<32;b=b+1) if (v[b]) msb_idx = b[5:0]; end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; send_start<=0; rgot<=0;
            mac_a<=0; mac_b<=0; mac_load<=0; r<=0; k<=0; fcnt<=0; round<=0; cur_send<=0;
        end else begin
            send_start<=0; done<=0;
            if (rdone) begin rhold<=rvec; rgot<=1; end

            case (st)
                IDLE: if (start) begin r<=0; k<=0; st<=MAC; end

                // ---- matmul (real DSP) ----
                MAC: begin
                    mac_a    <= {{10{wmem[r*K+k][7]}}, wmem[r*K+k]};
                    mac_b    <= {{10{my_x[k*W+(W-1)]}}, my_x[k*W +: W]};
                    mac_load <= (k==0);
                    if (k==K-1) begin k<=0; fcnt<=0; st<=FLUSH; end else k<=k+1;
                end
                FLUSH: begin
                    mac_a<=0; mac_b<=0; mac_load<=0;
                    if (fcnt==LAT) begin
                        yi32[r] <= mac_res[31:0];
                        if (r==NR-1) st<=MAXF; else begin r<=r+1; st<=MAC; end
                    end else fcnt<=fcnt+1;
                end
                MAXF: begin
                    max_abs = 0;
                    for (i=0;i<NR;i=i+1) begin
                        av = yi32[i][31] ? -yi32[i] : yi32[i];
                        if (av > max_abs) max_abs = av;
                    end
                    sh <= (max_abs==0) ? 6'd0 :
                          (msb_idx(max_abs) > 6) ? (msb_idx(max_abs)-6) : 6'd0;
                    st <= REQ;
                end
                REQ: begin
                    for (i=0;i<NR;i=i+1) begin
                        rounded = yi32[i] + ((sh>0) ? (32'sd1 <<< (sh-1)) : 32'sd0);
                        shifted = rounded >>> sh;
                        full_y[(ID*NR+i)*W +: W] <= (shifted > 32'sd127)  ? 8'sd127  :
                                                    (shifted < -32'sd128) ? -8'sd128 : shifted[7:0];
                    end
                    cur_send <= ID[7:0]; round <= 0;
                    st <= (NN==1) ? FIN : RSEND;
                    if (NN==1) done <= 1;
                end

                // ---- ring all-gather ----
                RSEND: begin
                    send_start <= 1;
                    recv_slot  <= ((ID+NN-1-round) >= NN) ?
                                  (ID+NN-1-round-NN) : (ID+NN-1-round);
                    st <= RSBUSY;
                end
                RSBUSY: if (send_busy)  st <= RSWAIT;
                RSWAIT: if (!send_busy) st <= RRECV;
                RRECV:  if (rgot) begin
                    full_y[recv_slot*SB +: SB] <= rhold;
                    rgot <= 0; st <= RNEXT;
                end
                RNEXT: begin
                    cur_send <= recv_slot;
                    if (round == NN-2) begin done <= 1; st <= FIN; end
                    else begin round <= round + 1; st <= RSEND; end
                end
                FIN: st <= IDLE;
            endcase
        end
    end
endmodule
