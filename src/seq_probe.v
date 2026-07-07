`timescale 1ns/1ps
// seq_probe - P&R probe : measure the LUT cost of the on-chip FFN-TP sequencer
// (ffn_tp_seq + vec_alu) + inter-FPGA link ALONE, without the brick node, so we
// can isolate what makes the integrated node_cluster overflow the GW2AR-18.
module seq_probe #(parameter W = 8, parameter D = 64) (
    input  wire        clk,
    output wire [5:0]  led
);
    reg [31:0] lfsr; wire [511:0] rnd;
    always @(posedge clk) lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
    assign rnd = {lfsr,lfsr,lfsr,lfsr,lfsr,lfsr,lfsr,lfsr, lfsr,lfsr,lfsr,lfsr,lfsr,lfsr,lfsr,lfsr};

    wire [W*D-1:0] seq_result; wire signed [7:0] seq_result_sh; wire seq_done;
    ffn_tp_seq2 #(.W(W), .D(D), .BOOT(20'd2000)) u_seq (
        .clk(clk), .rst_n(1'b1), .start(lfsr[0]), .x_in(rnd[W*D-1:0]),
        .sx_in(lfsr[7:0]), .sw_rms(lfsr[7:0]), .sw1(lfsr[7:0]), .sw3(lfsr[7:0]), .sw2(lfsr[7:0]),
        .base(lfsr[22:0]),
        .n0_rx(), .n1_rx(),
        .n0_tx(lfsr[2]), .n1_tx(lfsr[3]),
        .result(seq_result), .result_sh(seq_result_sh), .done(seq_done));

    // inter-FPGA link
    wire [W-1:0] tx_data; wire tx_send; wire tx_full; wire [W-1:0] rx_data; wire rx_empty; wire walmost_full;
    link_send #(.W(W), .L(D)) u_lsend (.clk(clk), .rst_n(1'b1), .start(lfsr[1]),
        .vec(rnd[W*D-1:0]), .busy(), .tx_data(tx_data), .tx_send(tx_send), .tx_full(tx_full));
    reg clk_div; always @(posedge clk) clk_div <= ~clk_div;
    async_fifo #(.DW(W), .AW(4)) u_fifo (.wclk(clk), .wrst_n(1'b1),
        .winc(tx_send & ~tx_full), .wdata(tx_data), .wfull(tx_full), .walmost_full(walmost_full),
        .rclk(clk_div), .rrst_n(1'b1), .rinc(~rx_empty), .rdata(rx_data), .rempty(rx_empty));

    assign led = { seq_done, ^seq_result[31:0], ^seq_result[511:480], seq_result_sh[3],
                   ^rx_data, walmost_full } ^ {lfsr[5:0]};
endmodule