`timescale 1ns/1ps
// =============================================================================
// node_gen - INTEGRATED autonomous-generation node for LUT budget measurement
// (P&R only), mono-carte.  Calque de node_cluster.v, mais la mini-GG UART est
// remplacee par le VRAI sequenceur de generation gen_seq, relie au noeud par le
// lien source-synchrone ss_link (deux async_fifo cmd/resp boucles ON-CHIP,
// exactement comme le top de sim gen_seq_top.v).
//
// Compose sur UNE Tang Nano 20K (GW2AR-18C) :
//   1. le datapath du noeud "brique"   = top.v (NODE_ONLY + SD_BOOT + LINK_SS)
//   2. le sequenceur de generation     = gen_seq (embed -> 5 couches -> lm_head
//                                        -> argmax -> boucle 17 tokens)
//   3. le lien inter-FPGA               = ss_link (async_fifo cmd + resp)
//
// C'est une SONDE PLACE&ROUTE, pas un test fonctionnel : un LFSR seme les entrees
// du sequenceur (x/shifts/rope/base) pour que le synthetiseur ne puisse PAS
// eliminer sa logique (KV-cache, logits, argmax...), et les sorties (token,
// result, done) sont XOR-reduites sur les LEDs pour qu'elles atteignent une
// broche. Le % LUT/BSRAM/DSP/CLS est le chiffre Gowin officiel d'un noeud
// autonome "tout compris". Le fonctionnel est deja prouve en sim (etape F).
// La lecture reelle du texte a l'UART est une etape materielle ULTERIEURE.
//
// LINK_SS boucle sur clk_sys : les fifos cmd/resp ont wclk==rclk==clk_sys, donc
// le sequenceur, ses fifos et le PHY link du noeud partagent un seul domaine
// d'horloge (le PLL du noeud, expose via clk_sys_out grace a CLUSTER_NODE).
// =============================================================================
`define NODE_ONLY
`define SD_BOOT
`define CLUSTER_NODE
`define LINK_SS
`include "top.v"

module node_gen #(parameter W = 8, parameter D = 64) (
    input  wire        clk,             // 27 MHz crystal (pin 4)
    output wire [5:0]  led,
    // SDRAM
    output wire        O_sdram_clk,
    output wire        O_sdram_cke,
    output wire        O_sdram_cs_n,
    output wire        O_sdram_cas_n,
    output wire        O_sdram_ras_n,
    output wire        O_sdram_wen_n,
    inout  wire [31:0] IO_sdram_dq,
    output wire [10:0] O_sdram_addr,
    output wire [1:0]  O_sdram_ba,
    output wire [3:0]  O_sdram_dqm,
    // SD card
    output wire        sd_clk,
    output wire        sd_cmd,
    input  wire        sd_dat0,
    output wire        sd_dat3
);
    localparam NL   = 5;
    localparam NPOS = 17;

    // ---- the brick node (own PLL -> clk_sys), LINK_SS transport ----
    wire        clk_sys;
    wire [5:0]  node_led;

    // gen_seq -> node : command fifo. node -> gen_seq : response fifo.
    wire [7:0]  s_cmd_data;  wire s_cmd_wr;  wire s_cmd_full;   // gen_seq write port
    wire [7:0]  c_rdata;     wire c_empty;   wire c_rd;         // node read port
    wire [7:0]  r_wdata;     wire r_wr;      wire r_full;       // node write port
    wire [7:0]  s_resp_data; wire s_resp_empty; wire s_resp_rd; // gen_seq read port

    top #(.SD_HALF(135), .SD_NBLK(16'd2)) u_node (
        .clk(clk),
        .lk_rx_data(c_rdata), .lk_rx_empty(c_empty), .lk_rx_rd(c_rd),
        .lk_tx_data(r_wdata), .lk_tx_wr(r_wr),       .lk_tx_full(r_full),
        .led(node_led),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke), .O_sdram_cs_n(O_sdram_cs_n),
        .O_sdram_cas_n(O_sdram_cas_n), .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .IO_sdram_dq(IO_sdram_dq), .O_sdram_addr(O_sdram_addr), .O_sdram_ba(O_sdram_ba),
        .O_sdram_dqm(O_sdram_dqm),
        .sd_clk(sd_clk), .sd_cmd(sd_cmd), .sd_dat0(sd_dat0), .sd_dat3(sd_dat3),
        .clk_sys_out(clk_sys));

    wire rst_n_sys = 1'b1;               // (sonde) garde le sequenceur hors reset

    // ---- LFSR : graines pseudo-aleatoires pour que la logique ne soit pas eliminee ----
    reg [31:0] lfsr;
    always @(posedge clk_sys) lfsr <= {lfsr[30:0], lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
    // vecteur pseudo-aleatoire pour les entrees restantes (x_in 512b, sw_* < 72b) ;
    // cos/sin sont maintenant en ROM BSRAM interne a gen_seq (plus de bus 1088b).
    wire [W*D-1:0] rnd  = {(W*D/32){lfsr}};   // 512 bits
    wire [W*D-1:0] rnd2 = {(W*D/32){~lfsr}};

    // ---- sequenceur de generation autonome (drive le noeud via ss_link) ----
    wire [W*D-1:0] seq_result; wire signed [7:0] seq_result_sh;
    wire [9:0]     seq_token;  wire seq_token_valid; wire seq_done;

    // TMAX=17 (= NPOS) : the KV cache only ever holds positions 0..NPOS-1, so sizing
    // it to 17 instead of 32 is output-identical for a 17-token generation (linear
    // (layer,pos) indexing, no power-of-2 masking) while shrinking ksh/vsh + the KV
    // BSRAM -> frees congested CLS. gen_mode tied to 1 : an autonomous node only ever
    // generates (embed-driven), so the single-shot LOADX path + external x_in are
    // pruned. Both are documented levers to relieve the CLS pressure (Session 6).
    gen_seq #(.W(W), .D(D), .NL(NL), .NPOS(NPOS), .TMAX(17)) u_seq (
        .clk(clk_sys), .rst_n(rst_n_sys),
        .start(lfsr[0]), .gen_mode(1'b1),
        .x_in(rnd[W*D-1:0]), .sx_in(lfsr[7:0]),
        .sw_rms(rnd[NL*8-1:0]),  .swq(rnd[NL*8+7:8]),   .swk(rnd[NL*8+15:16]),
        .swv(rnd[NL*8+23:24]),   .swo(rnd[NL*8+31:32]),
        .sw_rmsf(rnd2[NL*8-1:0]),.sw1(rnd2[NL*8+7:8]),  .sw3(rnd2[NL*8+15:16]),
        .sw2(rnd2[NL*8+23:24]),
        .sw_rmsfinal(lfsr[15:8]), .sw_emb(lfsr[23:16]),
        .pos(lfsr[5:0]),
        .base(lfsr[22:0]),
        // ss_link : cmd out / resp in
        .lk_cmd_data(s_cmd_data), .lk_cmd_wr(s_cmd_wr), .lk_cmd_full(s_cmd_full),
        .lk_resp_data(s_resp_data), .lk_resp_empty(s_resp_empty), .lk_resp_rd(s_resp_rd),
        .result(seq_result), .result_sh(seq_result_sh),
        .token(seq_token), .token_valid(seq_token_valid), .done(seq_done));

    // ---- lien inter-FPGA (source-synchrone, boucle ON-CHIP sur clk_sys) ----
    // cmd : gen_seq ecrit, le noeud lit (lk_rx). resp : le noeud ecrit (lk_tx),
    // gen_seq lit. Meme horloge des deux cotes -> fifo mono-domaine, cout LUT du
    // CDC identique que les fils sortent de la puce ou non.
    async_fifo #(.DW(8), .AW(4)) u_cmd (
        .wclk(clk_sys), .wrst_n(rst_n_sys), .winc(s_cmd_wr), .wdata(s_cmd_data), .wfull(s_cmd_full),
        .rclk(clk_sys), .rrst_n(rst_n_sys), .rinc(c_rd),     .rdata(c_rdata),    .rempty(c_empty));
    async_fifo #(.DW(8), .AW(4)) u_resp (
        .wclk(clk_sys), .wrst_n(rst_n_sys), .winc(r_wr),     .wdata(r_wdata),    .wfull(r_full),
        .rclk(clk_sys), .rrst_n(rst_n_sys), .rinc(s_resp_rd),.rdata(s_resp_data),.rempty(s_resp_empty));

    // ---- anti-elimination : chaque sortie du sequenceur atteint une broche ----
    wire [5:0] extra = { seq_done,
                         ^seq_result[31:0],
                         ^seq_result[511:480],
                         ^seq_token,
                         seq_result_sh[3],
                         seq_token_valid };
    assign led = node_led ^ extra;
endmodule
