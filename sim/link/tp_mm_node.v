`timescale 1ns/1ps
// =============================================================================
// tp_mm_node - autonomous tensor-parallel matmul node (real mini-GG, increment 1)
//
// One FSM, NO external per-op command: the node, link-driven, computes its slice
// y[N] = W[N,K] @ x[K] using the REAL DSP (mac18 / MULTALU18X18), requantizes
// exactly like the project (fq_ref), and sends the result back over the link.
//
//   RECV x  ->  MAC (N rows x K, via mac18)  ->  MAXFIND -> REQUANT  ->  SEND y
//
// Weights live in local memory (wmem), preloaded by backdoor (like the cluster
// sims) -> they NEVER cross the link; only activations do. This is the GG idea
// reduced to a node's slice: autonomous, real operator, small.
// =============================================================================
module tp_mm_node #(
    parameter W = 8,
    parameter K = 64,
    parameter N = 32
) (
    input  wire         clk,
    input  wire         rst_n,
    // link IN (x broadcast)
    output wire         rx_rd,
    input  wire [W-1:0] rx_data,
    input  wire         rx_empty,
    // link OUT (y slice)
    output wire [W-1:0] tx_data,
    output wire         tx_send,
    input  wire         tx_full,
    output reg          done
);
    localparam LAT = 4;   // mac18 pipeline drain (>= 3)

    // local weights (row-major), preloaded by backdoor
    reg signed [7:0] wmem [0:N*K-1];

    // received activation
    wire [W*K-1:0] xvec;
    wire           xdone;
    link_recv #(W, K) u_rx (
        .clk(clk), .rst_n(rst_n), .rx_rd(rx_rd), .rx_data(rx_data),
        .rx_empty(rx_empty), .vec(xvec), .done(xdone));

    // result to transmit
    reg  [W*N-1:0] yvec;
    reg            send_start;
    wire           send_busy;
    link_send #(W, N) u_tx (
        .clk(clk), .rst_n(rst_n), .start(send_start), .vec(yvec),
        .busy(send_busy), .tx_data(tx_data), .tx_send(tx_send), .tx_full(tx_full));

    // real DSP
    reg  signed [17:0] mac_a, mac_b;
    reg                mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));

    reg signed [31:0] yi32 [0:N-1];

    localparam IDLE=0, MAC=1, FLUSH=2, MAXF=3, REQ=4, SEND=5, WAITS=6, DONE=7;
    reg [2:0]  st;
    reg [9:0]  r, k, fcnt;
    reg signed [31:0] max_abs;
    reg [5:0]  sh;
    integer i;
    reg signed [31:0] av, rounded, shifted;

    function [5:0] msb_idx;
        input [31:0] v;
        integer b;
        begin
            msb_idx = 6'd0;
            for (b = 0; b < 32; b = b + 1) if (v[b]) msb_idx = b[5:0];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; done <= 0; send_start <= 0;
            mac_a <= 0; mac_b <= 0; mac_load <= 0; r <= 0; k <= 0; fcnt <= 0;
        end else begin
            done <= 0; send_start <= 0;
            case (st)
                IDLE: if (xdone) begin r <= 0; k <= 0; st <= MAC; end

                MAC: begin
                    mac_a    <= {{10{wmem[r*K + k][7]}}, wmem[r*K + k]};
                    mac_b    <= {{10{xvec[k*W + (W-1)]}}, xvec[k*W +: W]};
                    mac_load <= (k == 0);
                    if (k == K-1) begin k <= 0; fcnt <= 0; st <= FLUSH; end
                    else               k <= k + 1;
                end

                FLUSH: begin                       // drain pipeline (no new products)
                    mac_a <= 0; mac_b <= 0; mac_load <= 0;
                    if (fcnt == LAT) begin
                        yi32[r] <= mac_res[31:0];
                        if (r == N-1) st <= MAXF;
                        else begin r <= r + 1; st <= MAC; end
                    end else fcnt <= fcnt + 1;
                end

                MAXF: begin
                    max_abs = 0;
                    for (i = 0; i < N; i = i + 1) begin
                        av = yi32[i][31] ? -yi32[i] : yi32[i];
                        if (av > max_abs) max_abs = av;
                    end
                    sh <= (max_abs == 0) ? 6'd0 :
                          (msb_idx(max_abs) > 6) ? (msb_idx(max_abs) - 6) : 6'd0;
                    st <= REQ;
                end

                REQ: begin
                    for (i = 0; i < N; i = i + 1) begin
                        rounded = yi32[i] + ((sh > 0) ? (32'sd1 <<< (sh-1)) : 32'sd0);
                        shifted = rounded >>> sh;
                        yvec[i*W +: W] <= (shifted > 32'sd127)  ? 8'sd127  :
                                          (shifted < -32'sd128) ? -8'sd128 : shifted[7:0];
                    end
                    send_start <= 1; st <= SEND;
                end

                SEND:  if (send_busy)  st <= WAITS;
                WAITS: if (!send_busy) begin done <= 1; st <= DONE; end
                DONE:  st <= IDLE;
            endcase
        end
    end
endmodule
