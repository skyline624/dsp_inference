`timescale 1ns/1ps
// =============================================================================
// attn_seq_ss_top - NN brick nodes (top.v, LINK_SS) driven by the NN-param
// attention sequencer (attn_tp_seq, LINK_SS) over parallel ss_link fifos.
// NN=1 -> one autonomous card, NN>=2 -> head-parallel cluster, no UART.
// Mirror of ffn_seq2_ss_top : per node a command fifo (seq->node) and a
// response fifo (node->seq), single clock (the CDC is covered by test_uart_bridge).
// =============================================================================
module attn_seq_ss_top #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx, sw_rms, swq, swk, swv, swo,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire           done
);
    wire [8*NN-1:0] seq_cmd_data;  wire [NN-1:0] seq_cmd_wr;   wire [NN-1:0] seq_cmd_full;
    wire [8*NN-1:0] seq_resp_data; wire [NN-1:0] seq_resp_empty; wire [NN-1:0] seq_resp_rd;

    attn_tp_seq #(.W(W), .D(D), .NN(NN)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo), .base(23'd0),
        .lk_cmd_data(seq_cmd_data), .lk_cmd_wr(seq_cmd_wr), .lk_cmd_full(seq_cmd_full),
        .lk_resp_data(seq_resp_data), .lk_resp_empty(seq_resp_empty), .lk_resp_rd(seq_resp_rd),
        .result(result), .result_sh(result_sh), .done(done));

    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nodes
        wire [7:0] c_rdata; wire c_empty; wire c_rd;
        async_fifo #(.DW(8), .AW(4)) u_cmd (
            .wclk(clk), .wrst_n(rst_n), .winc(seq_cmd_wr[g]), .wdata(seq_cmd_data[8*g +: 8]),
            .wfull(seq_cmd_full[g]),
            .rclk(clk), .rrst_n(rst_n), .rinc(c_rd), .rdata(c_rdata), .rempty(c_empty));
        wire [7:0] r_wdata; wire r_wr; wire r_full;
        async_fifo #(.DW(8), .AW(4)) u_resp (
            .wclk(clk), .wrst_n(rst_n), .winc(r_wr), .wdata(r_wdata), .wfull(r_full),
            .rclk(clk), .rrst_n(rst_n), .rinc(seq_resp_rd[g]),
            .rdata(seq_resp_data[8*g +: 8]), .rempty(seq_resp_empty[g]));

        wire o_sdram_clk, o_sdram_cke, o_sdram_cs_n, o_sdram_cas_n, o_sdram_ras_n, o_sdram_wen_n;
        wire [10:0] o_sdram_addr; wire [1:0] o_sdram_ba; wire [3:0] o_sdram_dqm; wire [31:0] io_sdram_dq;

        top u_n (
            .clk(clk),
            .lk_rx_data(c_rdata), .lk_rx_empty(c_empty), .lk_rx_rd(c_rd),
            .lk_tx_data(r_wdata), .lk_tx_wr(r_wr), .lk_tx_full(r_full),
            .led(),
            .O_sdram_clk(o_sdram_clk), .O_sdram_cke(o_sdram_cke), .O_sdram_cs_n(o_sdram_cs_n),
            .O_sdram_cas_n(o_sdram_cas_n), .O_sdram_ras_n(o_sdram_ras_n), .O_sdram_wen_n(o_sdram_wen_n),
            .IO_sdram_dq(io_sdram_dq), .O_sdram_addr(o_sdram_addr), .O_sdram_ba(o_sdram_ba),
            .O_sdram_dqm(o_sdram_dqm));

        sdram_chip u_sdram (
            .SDRAM_CLK(o_sdram_clk), .SDRAM_CKE(o_sdram_cke), .SDRAM_nCS(o_sdram_cs_n),
            .SDRAM_nRAS(o_sdram_ras_n), .SDRAM_nCAS(o_sdram_cas_n), .SDRAM_nWE(o_sdram_wen_n),
            .SDRAM_BA(o_sdram_ba), .SDRAM_A(o_sdram_addr), .SDRAM_DQM(o_sdram_dqm),
            .SDRAM_DQ(io_sdram_dq));
    end endgenerate
endmodule
