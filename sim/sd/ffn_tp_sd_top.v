`timescale 1ns/1ps
// =============================================================================
// ffn_tp_sd_top - CLUSTER + SD boot : each node loads ITS slice from ITS OWN SD
//
// 2 nodes, each = top.v(SD_BOOT) + its own SD card model + SDRAM. Each SD holds
// only THAT node's weight slice. At power-on every node autonomously loads its
// slice SD->SDRAM (in parallel), then the on-chip FFN-tensor-parallel coordinator
// runs the FFN across both nodes (rmsnorm broadcast, W1/W3 row-parallel, W2
// col-parallel + all-reduce). Zero PC; the model is distributed across the 2 SDs.
//
// Compact SDRAM layout (per node) so the SD image is small:
//   rms @0x0000, W1 @0x1000, W3 @0x2000, W2 @0x3000  (32 blocks total)
// =============================================================================
module ffn_tp_sd_top #(parameter W = 8, parameter D = 64) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx, sw_rms, sw1, sw3, sw2,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire           done
);
    wire s_n0rx, s_n1rx, n0_tx, n1_tx;

    node_sd #(.NBLK(16'd32), .SDHALF(2)) u_n0 (
        .clk(clk), .uart_rx(s_n0rx), .uart_tx(n0_tx), .led());
    node_sd #(.NBLK(16'd32), .SDHALF(2)) u_n1 (
        .clk(clk), .uart_rx(s_n1rx), .uart_tx(n1_tx), .led());

    ffn_tp_seq #(.W(W), .D(D), .BOOT(700000),
                 .A_RMS(23'h0000), .A_W1(23'h1000), .A_W3(23'h2000), .A_W2(23'h3000)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .sw1(sw1), .sw3(sw3), .sw2(sw2), .base(23'd0),
        .n0_rx(s_n0rx), .n1_rx(s_n1rx), .n0_tx(n0_tx), .n1_tx(n1_tx),
        .result(result), .result_sh(result_sh), .done(done));
endmodule
