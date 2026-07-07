`timescale 1ns/1ps
// gen_seq_top - one node (top.v LINK_SS) + the generation sequencer over ss_link.
module gen_seq_top #(
    parameter W = 8, parameter D = 64, parameter HS = 8, parameter NL = 5, parameter NPOS = 17
) (
    input  wire           clk, rst_n, start, gen_mode,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx_in,
    input  wire [NL*8-1:0] sw_rms, swq, swk, swv, swo,
    input  wire [NL*8-1:0] sw_rmsf, sw1, sw3, sw2,
    input  wire signed [7:0] sw_rmsfinal, sw_emb,
    input  wire [5:0]     pos,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire [9:0]     token,
    output wire           token_valid,
    output wire           done
);
    wire [7:0] s_cmd_data;  wire s_cmd_wr;  wire s_cmd_full;
    wire [7:0] s_resp_data; wire s_resp_empty; wire s_resp_rd;

    gen_seq #(.W(W), .D(D), .NL(NL), .NPOS(NPOS)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .gen_mode(gen_mode),
        .x_in(x_in), .sx_in(sx_in),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .sw_rmsf(sw_rmsf), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .sw_rmsfinal(sw_rmsfinal), .sw_emb(sw_emb),
        .pos(pos), .base(23'd0),
        .lk_cmd_data(s_cmd_data), .lk_cmd_wr(s_cmd_wr), .lk_cmd_full(s_cmd_full),
        .lk_resp_data(s_resp_data), .lk_resp_empty(s_resp_empty), .lk_resp_rd(s_resp_rd),
        .result(result), .result_sh(result_sh), .token(token),
        .token_valid(token_valid), .done(done));

    wire [7:0] c_rdata; wire c_empty; wire c_rd;
    async_fifo #(.DW(8), .AW(4)) u_cmd (
        .wclk(clk), .wrst_n(rst_n), .winc(s_cmd_wr), .wdata(s_cmd_data), .wfull(s_cmd_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(c_rd), .rdata(c_rdata), .rempty(c_empty));
    wire [7:0] r_wdata; wire r_wr; wire r_full;
    async_fifo #(.DW(8), .AW(4)) u_resp (
        .wclk(clk), .wrst_n(rst_n), .winc(r_wr), .wdata(r_wdata), .wfull(r_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(s_resp_rd), .rdata(s_resp_data), .rempty(s_resp_empty));

    wire o_sdram_clk, o_sdram_cke, o_sdram_cs_n, o_sdram_cas_n, o_sdram_ras_n, o_sdram_wen_n;
    wire [10:0] o_sdram_addr; wire [1:0] o_sdram_ba; wire [3:0] o_sdram_dqm; wire [31:0] io_sdram_dq;
    top u_n (
        .clk(clk),
        .lk_rx_data(c_rdata), .lk_rx_empty(c_empty), .lk_rx_rd(c_rd),
        .lk_tx_data(r_wdata), .lk_tx_wr(r_wr), .lk_tx_full(r_full), .led(),
        .O_sdram_clk(o_sdram_clk), .O_sdram_cke(o_sdram_cke), .O_sdram_cs_n(o_sdram_cs_n),
        .O_sdram_cas_n(o_sdram_cas_n), .O_sdram_ras_n(o_sdram_ras_n), .O_sdram_wen_n(o_sdram_wen_n),
        .IO_sdram_dq(io_sdram_dq), .O_sdram_addr(o_sdram_addr), .O_sdram_ba(o_sdram_ba),
        .O_sdram_dqm(o_sdram_dqm));
    sdram_chip u_sdram (
        .SDRAM_CLK(o_sdram_clk), .SDRAM_CKE(o_sdram_cke), .SDRAM_nCS(o_sdram_cs_n),
        .SDRAM_nRAS(o_sdram_ras_n), .SDRAM_nCAS(o_sdram_cas_n), .SDRAM_nWE(o_sdram_wen_n),
        .SDRAM_BA(o_sdram_ba), .SDRAM_A(o_sdram_addr), .SDRAM_DQM(o_sdram_dqm),
        .SDRAM_DQ(io_sdram_dq));
endmodule
