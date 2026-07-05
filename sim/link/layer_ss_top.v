`timescale 1ns/1ps
// =============================================================================
// layer_ss_top - COMPLETE transformer layer (attn + FFN) over NN nodes on the
// parallel ss_link. Both sequencers (attn_tp_seq, ffn_tp_seq2, LINK_SS) share the
// SAME per-node command/response fifos, muxed by `phase`. NN=1 mono-card.
// =============================================================================
module layer_ss_top #(
    parameter W = 8,
    parameter D = 64,
    parameter NN = 2
) (
    input  wire           clk, rst_n, start,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx,
    input  wire signed [7:0] sw_rms_a, swq, swk, swv, swo,
    input  wire signed [7:0] sw_rms_f, sw1, sw3, sw2,
    output reg  [W*D-1:0] result,
    output reg  signed [7:0] result_sh,
    output reg            done
);
    reg phase;   // 0 = attention drives the nodes, 1 = FFN

    // attention sequencer link ports
    wire [8*NN-1:0] a_cmd_data;  wire [NN-1:0] a_cmd_wr;   wire [NN-1:0] a_cmd_full;
    wire [8*NN-1:0] a_resp_data; wire [NN-1:0] a_resp_empty; wire [NN-1:0] a_resp_rd;
    // ffn sequencer link ports
    wire [8*NN-1:0] f_cmd_data;  wire [NN-1:0] f_cmd_wr;   wire [NN-1:0] f_cmd_full;
    wire [8*NN-1:0] f_resp_data; wire [NN-1:0] f_resp_empty; wire [NN-1:0] f_resp_rd;
    // muxed node-facing fifo ports
    wire [8*NN-1:0] n_cmd_data  = phase ? f_cmd_data : a_cmd_data;
    wire [NN-1:0]   n_cmd_wr    = phase ? f_cmd_wr   : a_cmd_wr;
    wire [8*NN-1:0] n_resp_data;   wire [NN-1:0] n_resp_empty; wire [NN-1:0] n_resp_rd;
    wire [NN-1:0]   n_cmd_full_i;   // fifo full, driven by the node-side fifos below
    // route fifo signals back to whichever sequencer owns the bus
    assign a_cmd_full   = phase ? {NN{1'b1}} : n_cmd_full_i;
    assign f_cmd_full   = phase ? n_cmd_full_i : {NN{1'b1}};
    assign a_resp_data  = n_resp_data;
    assign f_resp_data  = n_resp_data;
    assign a_resp_empty = phase ? {NN{1'b1}} : n_resp_empty;
    assign f_resp_empty = phase ? n_resp_empty : {NN{1'b1}};
    assign n_resp_rd    = phase ? f_resp_rd : a_resp_rd;

    reg               a_start; wire a_done;
    wire [W*D-1:0]    a_res; wire signed [7:0] a_sh;
    attn_tp_seq #(.W(W), .D(D), .NN(NN)) u_attn (
        .clk(clk), .rst_n(rst_n), .start(a_start), .x_in(x_in), .sx_in(sx),
        .sw_rms(sw_rms_a), .swq(swq), .swk(swk), .swv(swv), .swo(swo), .base(23'd0),
        .lk_cmd_data(a_cmd_data), .lk_cmd_wr(a_cmd_wr), .lk_cmd_full(a_cmd_full),
        .lk_resp_data(a_resp_data), .lk_resp_empty(a_resp_empty), .lk_resp_rd(a_resp_rd),
        .result(a_res), .result_sh(a_sh), .done(a_done));

    reg               f_start; wire f_done;
    reg  [W*D-1:0]    x1; reg signed [7:0] sx1;
    wire [W*D-1:0]    f_res; wire signed [7:0] f_sh;
    ffn_tp_seq2 #(.W(W), .D(D), .NN(NN), .BOOT(16),
                  .A_RMS(23'h110000), .A_W1(23'h111000), .A_W3(23'h112000), .A_W2(23'h113000)) u_ffn (
        .clk(clk), .rst_n(rst_n), .start(f_start), .x_in(x1), .sx_in(sx1),
        .sw_rms(sw_rms_f), .sw1(sw1), .sw3(sw3), .sw2(sw2), .base(23'd0),
        .lk_cmd_data(f_cmd_data), .lk_cmd_wr(f_cmd_wr), .lk_cmd_full(f_cmd_full),
        .lk_resp_data(f_resp_data), .lk_resp_empty(f_resp_empty), .lk_resp_rd(f_resp_rd),
        .result(f_res), .result_sh(f_sh), .done(f_done));

    genvar g;
    generate for (g=0; g<NN; g=g+1) begin: nodes
        wire [7:0] c_rdata; wire c_empty; wire c_rd;
        async_fifo #(.DW(8), .AW(4)) u_cmd (
            .wclk(clk), .wrst_n(rst_n), .winc(n_cmd_wr[g]), .wdata(n_cmd_data[8*g +: 8]),
            .wfull(n_cmd_full_i[g]),
            .rclk(clk), .rrst_n(rst_n), .rinc(c_rd), .rdata(c_rdata), .rempty(c_empty));
        wire [7:0] r_wdata; wire r_wr; wire r_full;
        async_fifo #(.DW(8), .AW(4)) u_resp (
            .wclk(clk), .wrst_n(rst_n), .winc(r_wr), .wdata(r_wdata), .wfull(r_full),
            .rclk(clk), .rrst_n(rst_n), .rinc(n_resp_rd[g]),
            .rdata(n_resp_data[8*g +: 8]), .rempty(n_resp_empty[g]));

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
    end endgenerate

    localparam IDLE=0, ATTN=1, HANDOFF=2, FFN=3, FIN=4;
    reg [2:0] cst;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin cst<=IDLE; a_start<=0; f_start<=0; phase<=0; done<=0; end
        else begin
            a_start<=0; f_start<=0; done<=0;
            case (cst)
                IDLE: if (start) begin phase<=0; a_start<=1; cst<=ATTN; end
                ATTN: if (a_done) begin x1<=a_res; sx1<=a_sh; cst<=HANDOFF; end
                HANDOFF: begin phase<=1; f_start<=1; cst<=FFN; end
                FFN: if (f_done) begin result<=f_res; result_sh<=f_sh; done<=1; cst<=FIN; end
                FIN: cst<=IDLE;
            endcase
        end
    end
endmodule
