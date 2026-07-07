`timescale 1ns/1ps
// =============================================================================
// tp_ar_node - autonomous COLUMN-parallel matmul + ring ALL-REDUCE (mini-GG 3.2)
//
// The other tensor-parallel exchange (3.1 did all-gather / row-parallel). Here
// W is split by COLUMNS: each node holds W[:, slice] and x[slice], computes a
// PARTIAL output[D] (raw int32, no requant), and the partials are SUMMED across
// all nodes so every node gets the full result.
//
// All-reduce = all-gather the int32 partials (reuse the validated ring_node) then
// sum locally + requantize once. Summing raw partials == the full dot product,
// so the result is bit-exact vs fq_ref over the FULL W,x. Autonomous, scalable N.
// =============================================================================
module tp_ar_node #(
    parameter W  = 8,
    parameter D  = 64,    // output dim (full)
    parameter KS = 32,    // input columns held by THIS node (K / NN)
    parameter NN = 2,
    parameter ID = 0
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [W*KS-1:0]  x_slice,        // this node's input columns
    // ring links (32-bit elements: int32 partials)
    output wire [31:0]      o_data,
    output wire             o_send,
    input  wire             o_full,
    output wire             i_rd,
    input  wire [31:0]      i_data,
    input  wire             i_empty,
    output reg  [W*D-1:0]   out,            // full reduced+requantized result
    output reg              done
);
    localparam LAT = 4;

    reg signed [7:0]  wmem [0:D*KS-1];       // W_cols[d][k], d outer (backdoor)

    // real DSP
    reg  signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));
    reg signed [31:0] partial [0:D-1];

    // all-gather of the int32 partials (reuse validated ring_node, 32-bit elems)
    reg  [32*D-1:0]    my_partial;
    reg                ring_start;
    wire               ring_done;
    wire [32*D*NN-1:0] gathered;
    ring_node #(32, D, NN, ID) u_ring (
        .clk(clk), .rst_n(rst_n), .start(ring_start), .my_slice(my_partial),
        .o_data(o_data), .o_send(o_send), .o_full(o_full),
        .i_rd(i_rd), .i_data(i_data), .i_empty(i_empty),
        .full_vec(gathered), .done(ring_done));

    localparam IDLE=0, MAC=1, FLUSH=2, PACK=3, RING=4, RWAIT=5, SUM=6, MAXF=7, REQ=8, FIN=9;
    reg [3:0]  st;
    reg [9:0]  r, k, fcnt;
    reg signed [39:0] total [0:D-1];
    reg signed [39:0] max_abs;
    reg [5:0]  sh;
    integer i, g;
    reg signed [39:0] av, rounded, shifted, acc;

    function [5:0] msb_idx40;
        input [39:0] v; integer b;
        begin msb_idx40 = 0; for (b=0;b<40;b=b+1) if (v[b]) msb_idx40 = b[5:0]; end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; ring_start<=0; mac_a<=0; mac_b<=0; mac_load<=0;
            r<=0; k<=0; fcnt<=0;
        end else begin
            ring_start<=0; done<=0;
            case (st)
                IDLE: if (start) begin r<=0; k<=0; st<=MAC; end

                // ---- column-parallel matmul -> raw int32 partials (no requant) ----
                MAC: begin
                    mac_a    <= {{10{wmem[r*KS+k][7]}}, wmem[r*KS+k]};
                    mac_b    <= {{10{x_slice[k*W+(W-1)]}}, x_slice[k*W +: W]};
                    mac_load <= (k==0);
                    if (k==KS-1) begin k<=0; fcnt<=0; st<=FLUSH; end else k<=k+1;
                end
                FLUSH: begin
                    mac_a<=0; mac_b<=0; mac_load<=0;
                    if (fcnt==LAT) begin
                        partial[r] <= mac_res[31:0];
                        if (r==D-1) st<=PACK; else begin r<=r+1; st<=MAC; end
                    end else fcnt<=fcnt+1;
                end
                PACK: begin
                    for (i=0;i<D;i=i+1) my_partial[i*32 +: 32] <= partial[i];
                    st <= RING;
                end

                // ---- all-gather partials over the ring ----
                RING:  begin ring_start <= 1; st <= RWAIT; end
                RWAIT: if (ring_done) st <= SUM;

                // ---- local reduce (sum) + requantize ----
                SUM: begin
                    for (i=0;i<D;i=i+1) begin
                        acc = 0;
                        for (g=0;g<NN;g=g+1)
                            acc = acc + $signed(gathered[(g*D+i)*32 +: 32]);
                        total[i] <= acc;
                    end
                    st <= MAXF;
                end
                MAXF: begin
                    max_abs = 0;
                    for (i=0;i<D;i=i+1) begin
                        av = total[i][39] ? -total[i] : total[i];
                        if (av > max_abs) max_abs = av;
                    end
                    sh <= (max_abs==0) ? 6'd0 :
                          (msb_idx40(max_abs) > 6) ? (msb_idx40(max_abs)-6) : 6'd0;
                    st <= REQ;
                end
                REQ: begin
                    for (i=0;i<D;i=i+1) begin
                        rounded = total[i] + ((sh>0) ? (40'sd1 <<< (sh-1)) : 40'sd0);
                        shifted = rounded >>> sh;
                        out[i*W +: W] <= (shifted > 40'sd127)  ? 8'sd127  :
                                         (shifted < -40'sd128) ? -8'sd128 : shifted[7:0];
                    end
                    done <= 1; st <= FIN;
                end
                FIN: st <= IDLE;
            endcase
        end
    end
endmodule
