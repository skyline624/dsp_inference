// =============================================================================
// top.v  --  Main FSM for the dsp_inference project.
// Note : in-line comments below are a mix of English and French (legacy of
//        the original session). Identifiers and structure are English-only.
//
// UART commands (1 Mbaud) :
//   NN : RMSNorm.   PC->FPGA: 'N''N' shift_x shift_w x[64] w[64]    (132 B)
//                   FPGA->PC: 'N''K' shift_out
//                            acc[3]+p[1]+shift_amt[1]+raw_inv[2]+apply_shift[1]
//                            out[64]                                  (75 B)
//
//   SS : SiLU.      PC->FPGA: 'S''S' shift_x x[64]                   (67 B)
//                   FPGA->PC: 'S''K' shift_out lut_idx[1] silu_int[2 LE] out[64]
//                                                                    (70 B)
//
//   RR : RoPE.      PC->FPGA: 'R''R' shift_x x[8] cos[4]i16LE sin[4]i16LE  (27 B)
//                   FPGA->PC: 'R''K' shift_out new_real[4 LE] new_imag[4 LE] out[8]
//                                                                    (19 B)
//
//   XX : Softmax.   PC->FPGA: 'X''X' shift_x x[32]                   (35 B)
//                   FPGA->PC: 'X''K' shift_out max[1] sum[3] p_sum[1] inv_sum[2] out[32]
//                                                                    (42 B)
//
//   AA : Single-head attention (HS=8, variable T).
//        PC->FPGA: 'A''A' sq sk sv T(1) Q[8] K[T*8] V[T*8]
//        FPGA->PC: 'A''K' shift_out score_last[2] max[1] sum[2] inv_sum[2] out[8]  (18 B)
//
//   MM : Multi-head attention (H=8, KH=4, HS=8, GQA n_rep=2).
//        PC->FPGA: 'M''M' sq sk sv T(1) Q[64] K[T*32] V[T*32]
//        FPGA->PC: 'M''K' shift_out out[64]  (67 B)
//
//   WW : SDRAM single-byte write. PC->FPGA: 'W''W' addr[3 LE] data(1)  (6 B)
//                                  FPGA->PC: 'W''K'                     (2 B)
//
//   BB : SDRAM single-byte read.  PC->FPGA: 'B''B' addr[3 LE]           (5 B)
//                                  FPGA->PC: 'B''K' data(1)             (3 B)
//
//   LL : SDRAM bulk load.         PC->FPGA: 'L''L' addr[3 LE] N[2 LE] data[N]   (7+N B)
//                                  FPGA->PC: 'L''K'                             (2 B)
//
//   CC : SDRAM bulk dump.         PC->FPGA: 'C''C' addr[3 LE] N[2 LE]           (7 B)
//                                  FPGA->PC: 'C''K' data[N]                     (2+N B)
//
//   FN : RMSNorm with w in SDRAM. PC->FPGA: 'F''N' sx sw x[64] addr[3]   (71 B)
//                                  FPGA->PC: same response as NN  (75 B)
//        The module DMAs the 64 bytes of w from SDRAM[addr] into wbuf,
//        then triggers rmsnorm_op. Validates the SDRAM->BSRAM path.
//
//   FM : Matmul y[N] = W[N][64] . x[64], W fetched from SDRAM (K=64 hardcoded).
//        PC->FPGA: 'F''M' N(1) sx sw x[64] addr[3]                    (72 B)
//        FPGA->PC: 'F''M' y[N*4 LE int32]                             (2+4N B)
//
//   FQ : Matmul + reQuantize to int8+shift (N <= 64). Chainable with other ops.
//        PC->FPGA: 'F''Q' N(1) sx sw x[64] addr[3]                    (72 B)
//        FPGA->PC: 'F''Q' shift_total(1) y_int8[N]                    (3+N B)
//
//   CN : Chain rmsnorm + matmul + requantize. Fully on FPGA, zero round-trips.
//        Steps : fetch rms_w -> rmsnorm(x, rms_w) -> copy obuf->xbuf
//                -> fetch W_row * N -> matmul -> requantize to int8+shift_total.
//        PC->FPGA: 'C''N' sx sw_rms sw_mm x[64] N(1) addr_rms(3) addr_W(3)
//                                                                    (76 B)
//        FPGA->PC: 'C''N' shift_total(1) y_int8[N]                   (3+N B)
//
//   CS : Like CN + SiLU(matmul_output), outputs int8 silu(y).
//        Steps : CN -> copy obuf->xbuf -> silu_op -> output silu(y)
//        Same RX protocol as CN (76 B).
//        FPGA->PC: 'C''S' shift_total(1) silu_y_int8[N]              (3+N B)
//
//   EE : Embedding lookup. PC->FPGA: 'E''E' tok_lo tok_hi          (4 B)
//        FPGA fetches 64 B from SDRAM[ADDR_TOK_EMB + tok*64] to obuf.
//        FPGA->PC: 'E''K' x[64]                                       (66 B)
//
//   GG : Generation FSM (incremental v0..v5g, see PLAN_GG_AUTONOMIE.md).
//        Eventually : 'G''G' start_tok N + per-model shifts -> N tokens.
// =============================================================================

module top (
    input  wire        clk,             // 27 MHz, pin 4
    input  wire        uart_rx,
    output wire        uart_tx,
    output wire [5:0]  led,

    // SDRAM (magic port names, auto-routed par Gowin)
    output wire        O_sdram_clk,
    output wire        O_sdram_cke,
    output wire        O_sdram_cs_n,
    output wire        O_sdram_cas_n,
    output wire        O_sdram_ras_n,
    output wire        O_sdram_wen_n,
    inout  wire [31:0] IO_sdram_dq,
    output wire [10:0] O_sdram_addr,
    output wire [1:0]  O_sdram_ba,
    output wire [3:0]  O_sdram_dqm
);

    localparam D = 64;
    localparam BSRAM_SZ = 1024;  // bumped a 1024 pour MM T_MAX=32 (32*32=1024 bytes K/V)

    // ---- PLL : clk_sys 27 MHz + clk_sdram phase-shifte ----
    wire clk_sys, clk_sdram, pll_lock;
    Gowin_rPLL u_pll (
        .clkout(clk_sys), .clkoutp(clk_sdram),
        .lock(pll_lock), .reset(1'b0), .clkin(clk)
    );

    // Reset : attend PLL lock + 32k cycles d'init
    reg [15:0] init_cnt = 16'd0;
    wire init_done = init_cnt[15];
    always @(posedge clk_sys) begin
        if (!pll_lock) init_cnt <= 16'd0;
        else if (!init_done) init_cnt <= init_cnt + 16'd1;
    end
    wire rst   = ~init_done;
    wire rst_n =  init_done;

    // ---- SDRAM controller (NESTang) ----
    reg  [22:0] sd_addr;
    reg         sd_rd, sd_wr;
    reg  [7:0]  sd_din;
    wire [7:0]  sd_dout;
    wire        sd_data_ready;
    wire        sd_busy;
    // refresh : COMPTEUR SEUL. Le PULSE sd_refresh vient EXCLUSIVEMENT de la FSM
    // principale (pattern v3). Counter reset uniquement quand sd_refresh fire reel.
    reg  [9:0]  refresh_cnt = 10'd0;
    wire        refresh_due = (refresh_cnt >= 10'd378);    // ~14 us, sous limite 15 us
    reg         sd_refresh;
    always @(posedge clk_sys) begin
        if (rst)                          refresh_cnt <= 10'd0;
        else if (sd_refresh)              refresh_cnt <= 10'd0;
        else if (refresh_cnt != 10'd1023) refresh_cnt <= refresh_cnt + 10'd1;
    end
    sdram #(.FREQ(27_000_000)) u_sdram (
        .clk(clk_sys), .clk_sdram(clk_sdram), .resetn(rst_n),
        .addr(sd_addr), .rd(sd_rd), .wr(sd_wr), .refresh(sd_refresh),
        .din(sd_din), .dout(sd_dout), .dout32(),
        .data_ready(sd_data_ready), .busy(sd_busy),
        .SDRAM_DQ(IO_sdram_dq), .SDRAM_A(O_sdram_addr), .SDRAM_BA(O_sdram_ba),
        .SDRAM_nCS(O_sdram_cs_n), .SDRAM_nWE(O_sdram_wen_n),
        .SDRAM_nRAS(O_sdram_ras_n), .SDRAM_nCAS(O_sdram_cas_n),
        .SDRAM_CLK(O_sdram_clk), .SDRAM_CKE(O_sdram_cke), .SDRAM_DQM(O_sdram_dqm)
    );

    wire [7:0] rx_data; wire rx_valid;
    uart_rx_8n1 #(.DIV(27)) u_rx (.clk(clk_sys), .rst(rst), .rx(uart_rx),
                                    .data(rx_data), .valid(rx_valid));
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    uart_tx_8n1 #(.DIV(27)) u_tx (.clk(clk_sys), .rst(rst),
                                    .data(tx_data), .send(tx_send),
                                    .tx(uart_tx), .busy(tx_busy));

    reg       rx_pending = 1'b0;
    reg [7:0] rx_byte    = 8'd0;
    wire      rx_consume;
    always @(posedge clk_sys) begin
        if (rst)             rx_pending <= 1'b0;
        else if (rx_valid)   begin rx_byte <= rx_data; rx_pending <= 1'b1; end
        else if (rx_consume) rx_pending <= 1'b0;
    end

    // ---- BSRAMs ----
    // xbuf/wbuf 128 oct : utilises par rmsnorm/silu/softmax (0..63) ET par attention K/V multi-head (0..127)
    reg signed [7:0] xbuf [0:BSRAM_SZ-1];
    reg signed [7:0] wbuf [0:BSRAM_SZ-1];
    reg signed [7:0] obuf [0:D-1];

    reg  [9:0]        xbuf_waddr; reg signed [7:0] xbuf_wdata; reg xbuf_we;
    reg  [9:0]        wbuf_waddr; reg signed [7:0] wbuf_wdata; reg wbuf_we;

    // Lectures pilotees par op active (mux)
    wire [9:0]        x_raddr_op, w_raddr_op, out_waddr_op;
    reg  signed [7:0] x_rdata_reg, w_rdata_reg;
    wire signed [7:0] out_wdata_op;
    wire              out_we_op;
    reg  [9:0]        obuf_raddr;
    reg  signed [7:0] obuf_rdata_reg;

    always @(posedge clk_sys) begin
        if (xbuf_we) xbuf[xbuf_waddr] <= xbuf_wdata;
        x_rdata_reg <= xbuf[x_raddr_op];
    end
    always @(posedge clk_sys) begin
        if (wbuf_we) wbuf[wbuf_waddr] <= wbuf_wdata;
        w_rdata_reg <= wbuf[w_raddr_op];
    end
    always @(posedge clk_sys) begin
        if (out_we_op) obuf[out_waddr_op[5:0]] <= out_wdata_op;
        obuf_rdata_reg <= obuf[obuf_raddr[5:0]];
    end

    // ---- Select operateur actif (4-bit pour faire de la place a CN) ----
    reg [3:0] op_sel;
    wire op_silu = (op_sel == 4'd1);
    wire op_rope = (op_sel == 4'd2);
    wire op_soft = (op_sel == 4'd3);
    wire op_attn = (op_sel == 4'd4);
    wire op_mh   = (op_sel == 4'd5);
    wire op_fn   = (op_sel == 4'd6);
    wire op_fm   = (op_sel == 4'd7);   // FM ou FQ (selon fq_mode)
    wire op_cn   = (op_sel == 4'd8);   // chain rmsnorm + matmul
    wire op_ee   = (op_sel == 4'd9);   // EE : embedding lookup (token -> x[64])
    wire op_gg   = (op_sel == 4'd10);  // GG : generation FSM (incremental v0..vN)
    reg  fq_mode;                       // 0 = FM (int32), 1 = FQ (requant)

    // ---- Adresses hardcodees des poids stories260K (modele specifique) ----
    // Layout identical a infer_fpga.py quantize_and_load_weights
    localparam [22:0] ADDR_TOK_EMB     = 23'h000000;
    // Layer 0 : base = 0x010000
    localparam [22:0] ADDR_RMS_ATT_L0  = 23'h010000;
    localparam [22:0] ADDR_WQ_L0       = 23'h010100;
    localparam [22:0] ADDR_WK_L0       = 23'h011100;
    localparam [22:0] ADDR_WV_L0       = 23'h011900;
    localparam [22:0] ADDR_WO_L0       = 23'h012100;
    localparam [22:0] ADDR_RMS_FFN_L0  = 23'h013100;
    localparam [22:0] ADDR_W1_L0       = 23'h013200;
    localparam [22:0] ADDR_W3_L0       = 23'h016200;
    localparam [22:0] ADDR_W2_L0       = 23'h019200;
    reg  cn_active;                     // 1 = on est in une commande CN
    reg  cs_active;                     // 1 = CS (CN + silu)
    reg signed [7:0] cn_sw_mm;          // saved shift_w pour matmul
    reg  [22:0] cn_addr_mm;             // saved addr matmul

    // GG (generation FSM) -- flag actif pendant operation GG
    reg          gg_active;
    reg signed [7:0] gg_sh_q;           // shift wq pour le matmul Q
    reg signed [7:0] gg_sh_k;           // shift wk
    reg signed [7:0] gg_sh_v;           // shift wv
    reg [3:0]        gg_qkv_phase;      // 0=Q 1=K 2=V 3=Wo 4=post-res 5=W1ch0 6=W1ch1 7=W1ch2
    reg signed [7:0] gg_sh_q_out;       // shift de Q after matmul (saved)
    reg signed [7:0] gg_sh_k_out;       // shift de K after matmul
    reg signed [7:0] gg_sh_v_out;       // shift de V after matmul
    reg [255:0]      gg_k_packed;       // K[32] storage (32 bytes packed)
    reg [255:0]      gg_v_packed;       // V[32] storage
    reg [511:0]      x_save_packed;     // x_orig pour le residual (64 bytes)
    reg signed [7:0] gg_x_shift;        // shift courant du x sauve (pour residual)
    reg signed [7:0] gg_sh_o;           // shift de Wo (recu de PC)
    reg signed [7:0] gg_sh_rf;          // shift de rms_ffn (recu de PC)
    reg signed [7:0] gg_sh_h1;          // shift de W1 (recu de PC)
    // h1[192] storage : BSRAM force pour economiser FF (1-cycle latency read)
    (* syn_ramstyle = "block_ram" *) reg signed [7:0] h1_packed [0:191];
    reg signed [7:0] h1_rdata_reg;
    reg [7:0]        h1_raddr;
    always @(posedge clk_sys) begin
        h1_rdata_reg <= h1_packed[h1_raddr];
    end
    reg signed [7:0] gg_sh_h1_ch0;      // shift de h1 chunk 0 after FQ
    reg signed [7:0] gg_sh_h1_ch1;      // shift de h1 chunk 1
    reg signed [7:0] gg_sh_h1_ch2;      // shift de h1 chunk 2
    reg signed [7:0] gg_sh_h3;          // shift de W3 (recu de PC)
    (* syn_ramstyle = "block_ram" *) reg signed [7:0] h3_packed [0:191];
    reg signed [7:0] h3_rdata_reg;
    reg [7:0]        h3_raddr;
    reg signed [7:0] gg_sh_h3_ch0, gg_sh_h3_ch1, gg_sh_h3_ch2;
    always @(posedge clk_sys) begin
        h3_rdata_reg <= h3_packed[h3_raddr];
    end
    // silu_packed : output silu sur 3 chunks de 64 (192 bytes)
    (* syn_ramstyle = "block_ram" *) reg signed [7:0] silu_packed [0:191];
    reg signed [7:0] silu_rdata_reg;
    reg [7:0]        silu_raddr;
    reg signed [7:0] gg_sh_silu_ch0, gg_sh_silu_ch1, gg_sh_silu_ch2;
    reg [1:0]        gg_silu_chunk;     // 0..2 quel chunk silu on traite
    always @(posedge clk_sys) begin
        silu_rdata_reg <= silu_packed[silu_raddr];
    end
    // tmp_prod : produits silu*h3 stockes en int32 (aligne au shift common)
    (* syn_ramstyle = "block_ram" *) reg signed [31:0] tmp_prod [0:191];
    reg signed [31:0] tmp_prod_rdata_reg;
    reg [7:0]         tmp_prod_raddr;
    always @(posedge clk_sys) begin
        tmp_prod_rdata_reg <= tmp_prod[tmp_prod_raddr];
    end
    // h_gated : output multiply, single shift
    (* syn_ramstyle = "block_ram" *) reg signed [7:0] h_gated_packed [0:191];
    reg signed [7:0] h_gated_rdata_reg;
    reg [7:0]        h_gated_raddr;
    reg signed [7:0] gg_sh_h_gated;
    always @(posedge clk_sys) begin
        h_gated_rdata_reg <= h_gated_packed[h_gated_raddr];
    end
    // W2 chunked : partials BSRAM
    (* syn_ramstyle = "block_ram" *) reg signed [7:0] partials_packed [0:191];
    reg signed [7:0] partials_rdata_reg;
    reg [7:0]        partials_raddr;
    reg signed [7:0] gg_sh_h_gated_x;     // shift de l'input W2 (= gg_sh_h_gated)
    reg signed [7:0] gg_sh_w2;            // shift Wo (recu de PC)
    reg signed [7:0] gg_sh_p0, gg_sh_p1, gg_sh_p2;
    reg [1:0]        gg_w2_chunk;
    reg signed [7:0] gg_p0_temp, gg_p1_temp;
    // v5g v2 : accumulate flag (FQ dot_acc starts from y_int32[fm_row] instead of 0)
    // skip_requant flag (FQ ends without MAX/REQ loops, goes to next chunk setup)
    reg              gg_accumulate;
    reg              gg_skip_requant;
    always @(posedge clk_sys) begin
        partials_rdata_reg <= partials_packed[partials_raddr];
    end
    // Pre-compute sh_prod per chunk + min + extras (combinational)
    wire signed [7:0] sh_prod_0 = gg_sh_silu_ch0 + gg_sh_h3_ch0;
    wire signed [7:0] sh_prod_1 = gg_sh_silu_ch1 + gg_sh_h3_ch1;
    wire signed [7:0] sh_prod_2 = gg_sh_silu_ch2 + gg_sh_h3_ch2;
    wire signed [7:0] sh_prod_min = (sh_prod_0 <= sh_prod_1) ?
                                    ((sh_prod_0 <= sh_prod_2) ? sh_prod_0 : sh_prod_2) :
                                    ((sh_prod_1 <= sh_prod_2) ? sh_prod_1 : sh_prod_2);
    wire signed [7:0] sh_extra_0 = sh_prod_0 - sh_prod_min;
    wire signed [7:0] sh_extra_1 = sh_prod_1 - sh_prod_min;
    wire signed [7:0] sh_extra_2 = sh_prod_2 - sh_prod_min;
    reg [7:0] gg_mult_i;            // index 0..191
    reg [1:0] gg_mult_chunk;        // i[7:6] = chunk index
    reg [31:0] gg_mult_max_abs;
    reg [5:0]  gg_mult_lead_bit;
    reg [5:0]  gg_mult_add_shift;
    integer gmi;
    always @(*) begin
        gg_mult_lead_bit = 6'd0;
        for (gmi = 0; gmi < 32; gmi = gmi + 1)
            if (gg_mult_max_abs[gmi]) gg_mult_lead_bit = gmi[5:0];
        gg_mult_add_shift = (gg_mult_lead_bit > 6'd6) ? (gg_mult_lead_bit - 6'd6) : 6'd0;
    end

    // ---- Cos/sin storage (pour RoPE) ----
    reg signed [15:0] cos_reg [0:3];
    reg signed [15:0] sin_reg [0:3];

    // ---- Q storage : 64 bytes packed (8 heads * 8 elements) ----
    reg [511:0] Q_packed;    // 64 bytes
    reg [511:0] Out_packed;  // 64 bytes
    reg [3:0]   mh_h;        // index head courant (0..7)
    // Q_flat fourni a attention_head_op : 8 bytes du head mh_h
    wire [63:0] Q_flat = Q_packed[mh_h*64 +: 64];

    // ---- RMSNorm ----
    reg               rms_start;
    wire              rms_done;
    reg  signed [7:0] rms_shift_x, rms_shift_w;
    wire signed [7:0] rms_shift_out;
    wire [9:0]        rms_x_raddr, rms_w_raddr, rms_out_waddr;
    wire signed [7:0] rms_out_wdata;
    wire              rms_out_we;
    wire [23:0]       dbg_acc;
    wire [4:0]        dbg_p;
    wire signed [5:0] dbg_shift_amt;
    wire [15:0]       dbg_raw_inv;
    wire signed [7:0] dbg_apply_shift;

    rmsnorm_op #(.D(D), .LOG2D(6)) u_rms (
        .clk(clk_sys), .rst(rst), .start(rms_start), .done(rms_done),
        .shift_x(rms_shift_x), .shift_w(rms_shift_w), .shift_out(rms_shift_out),
        .x_raddr(rms_x_raddr), .x_rdata(x_rdata_reg),
        .w_raddr(rms_w_raddr), .w_rdata(w_rdata_reg),
        .out_waddr(rms_out_waddr), .out_wdata(rms_out_wdata), .out_we(rms_out_we),
        .dbg_acc(dbg_acc), .dbg_p(dbg_p),
        .dbg_shift_amt(dbg_shift_amt), .dbg_raw_inv(dbg_raw_inv),
        .dbg_apply_shift(dbg_apply_shift)
    );

    // ---- SiLU ----
    reg               silu_start;
    wire              silu_done;
    reg  signed [7:0] silu_shift_x;
    wire signed [7:0] silu_shift_out;
    wire [9:0]        silu_x_raddr, silu_out_waddr;
    wire signed [7:0] silu_out_wdata;
    wire              silu_out_we;
    wire [7:0]        dbg_lut_idx;
    wire signed [15:0] dbg_silu_int;

    silu_op #(.D(D)) u_silu (
        .clk(clk_sys), .rst(rst), .start(silu_start), .done(silu_done),
        .shift_x(silu_shift_x), .shift_out(silu_shift_out),
        .x_raddr(silu_x_raddr), .x_rdata(x_rdata_reg),
        .out_waddr(silu_out_waddr), .out_wdata(silu_out_wdata), .out_we(silu_out_we),
        .dbg_lut_idx(dbg_lut_idx), .dbg_silu_int(dbg_silu_int)
    );

    // ---- RoPE ----
    reg               rope_start;
    wire              rope_done;
    reg  signed [7:0] rope_shift_x;
    wire signed [7:0] rope_shift_out;
    wire [9:0]        rope_x_raddr, rope_out_waddr;
    wire signed [7:0] rope_out_wdata;
    wire              rope_out_we;
    wire [3:0]        rope_pair_idx;
    wire signed [31:0] dbg_rope_real, dbg_rope_imag;

    rope_op #(.HS(8), .HALF(4)) u_rope (
        .clk(clk_sys), .rst(rst), .start(rope_start), .done(rope_done),
        .shift_x(rope_shift_x), .shift_out(rope_shift_out),
        .x_raddr(rope_x_raddr), .x_rdata(x_rdata_reg),
        .out_waddr(rope_out_waddr), .out_wdata(rope_out_wdata), .out_we(rope_out_we),
        .pair_idx(rope_pair_idx),
        .cos_in(cos_reg[rope_pair_idx[1:0]]),
        .sin_in(sin_reg[rope_pair_idx[1:0]]),
        .dbg_new_real(dbg_rope_real), .dbg_new_imag(dbg_rope_imag)
    );

    // ---- Softmax ----
    reg               soft_start;
    wire              soft_done;
    reg  signed [7:0] soft_shift_x;
    wire signed [7:0] soft_shift_out;
    wire [9:0]        soft_x_raddr, soft_out_waddr;
    wire signed [7:0] soft_out_wdata;
    wire              soft_out_we;
    wire signed [7:0] dbg_soft_max;
    wire [23:0]       dbg_soft_sum;
    wire [7:0]        dbg_soft_p_sum;
    wire [15:0]       dbg_soft_inv_sum;

    softmax_op #(.K_MAX(32)) u_soft (
        .clk(clk_sys), .rst(rst), .start(soft_start), .done(soft_done),
        .shift_x(soft_shift_x), .shift_out(soft_shift_out),
        .x_raddr(soft_x_raddr), .x_rdata(x_rdata_reg),
        .out_waddr(soft_out_waddr), .out_wdata(soft_out_wdata), .out_we(soft_out_we),
        .dbg_max(dbg_soft_max), .dbg_sum(dbg_soft_sum),
        .dbg_p_sum(dbg_soft_p_sum), .dbg_inv_sum(dbg_soft_inv_sum)
    );

    // ---- Attention ----
    reg               attn_start;
    wire              attn_done;
    reg  signed [7:0] attn_shift_q, attn_shift_k, attn_shift_v;
    wire signed [7:0] attn_shift_out;
    reg  [5:0]        attn_T;
    reg  [5:0]        attn_kv_stride, attn_kv_offset;
    wire [9:0]        attn_k_raddr, attn_v_raddr, attn_out_waddr;
    wire signed [7:0] attn_out_wdata;
    wire              attn_out_we;
    wire signed [15:0] dbg_attn_score_last;
    wire signed [7:0]  dbg_attn_max_score;
    wire [15:0]        dbg_attn_exp_sum, dbg_attn_inv_sum;

    attention_head_op #(.HS(8), .T_MAX(32)) u_attn (
        .clk(clk_sys), .rst(rst), .start(attn_start), .done(attn_done),
        .T(attn_T),
        .shift_q(attn_shift_q), .shift_k(attn_shift_k), .shift_v(attn_shift_v),
        .shift_out(attn_shift_out),
        .kv_stride(attn_kv_stride), .kv_offset(attn_kv_offset),
        .Q_flat(Q_flat),
        .k_raddr(attn_k_raddr), .k_rdata(x_rdata_reg),
        .v_raddr(attn_v_raddr), .v_rdata(w_rdata_reg),
        .out_waddr(attn_out_waddr), .out_wdata(attn_out_wdata), .out_we(attn_out_we),
        .dbg_score_last(dbg_attn_score_last),
        .dbg_max_score(dbg_attn_max_score),
        .dbg_exp_sum(dbg_attn_exp_sum),
        .dbg_inv_sum(dbg_attn_inv_sum)
    );

    // MUX BSRAM ports : op_mh partage with attn ; op_fm prend la main pendant dot product
    wire use_attn = op_attn | op_mh;
    assign x_raddr_op   = op_fm    ? x_raddr_fm     :
                          use_attn ? attn_k_raddr   :
                          op_soft  ? soft_x_raddr   :
                          op_rope  ? rope_x_raddr   :
                          op_silu  ? silu_x_raddr   : rms_x_raddr;
    assign w_raddr_op   = op_fm    ? w_raddr_fm     :
                          use_attn ? attn_v_raddr   : rms_w_raddr;
    // EE (embedding lookup) ecrit directement in obuf via ces regs
    reg [5:0]        ee_obuf_waddr;
    reg signed [7:0] ee_obuf_wdata;
    reg              ee_obuf_we;

    assign out_waddr_op = op_ee    ? {4'd0, ee_obuf_waddr} :
                          op_fm    ? out_waddr_fm   :
                          use_attn ? attn_out_waddr :
                          op_soft  ? soft_out_waddr :
                          op_rope  ? rope_out_waddr :
                          op_silu  ? silu_out_waddr : rms_out_waddr;
    assign out_wdata_op = op_ee    ? ee_obuf_wdata :
                          op_fm    ? out_wdata_fm   :
                          use_attn ? attn_out_wdata :
                          op_soft  ? soft_out_wdata :
                          op_rope  ? rope_out_wdata :
                          op_silu  ? silu_out_wdata : rms_out_wdata;
    assign out_we_op    = op_ee    ? ee_obuf_we     :
                          op_fm    ? out_we_fm      :
                          use_attn ? attn_out_we    :
                          op_soft  ? soft_out_we    :
                          op_rope  ? rope_out_we    :
                          op_silu  ? silu_out_we    : rms_out_we;

    // ---- FSM principale (8 bits from le refactor v3-style) ----
    localparam [8:0]
        S_IDLE      = 9'd0,
        S_M2_N      = 6'd1,
        S_M2_S      = 6'd2,
        S_M2_R      = 6'd3,
        S_M2_X      = 6'd4,
        S_M2_A      = 6'd5,
        S_M2_M      = 6'd6,
        S_NN_SX     = 6'd7,
        S_NN_SW     = 6'd8,
        S_NN_RX_X   = 6'd9,
        S_NN_RX_W   = 6'd10,
        S_SS_SX     = 6'd11,
        S_SS_RX_X   = 6'd12,
        S_RR_SX     = 6'd13,
        S_RR_RX_X   = 6'd14,
        S_RR_RX_COS = 6'd15,
        S_RR_RX_SIN = 6'd16,
        S_XX_SX     = 6'd17,
        S_XX_RX_X   = 6'd18,
        S_AA_SQ     = 6'd19,
        S_AA_SK     = 6'd20,
        S_AA_SV     = 6'd21,
        S_AA_T      = 6'd22,
        S_AA_RX_Q   = 6'd23,
        S_AA_RX_K   = 6'd24,
        S_AA_RX_V   = 6'd25,
        S_MM_SQ     = 6'd26,
        S_MM_SK     = 6'd27,
        S_MM_SV     = 6'd28,
        S_MM_T      = 6'd29,
        S_MM_RX_Q   = 6'd30,
        S_MM_RX_K   = 6'd31,
        S_MM_RX_V   = 6'd32,
        S_MM_HEAD   = 6'd33,
        S_MM_WAIT   = 6'd34,
        S_MM_COPY1  = 6'd35,
        S_MM_COPY2  = 6'd36,
        S_MM_COPY3  = 6'd37,
        S_MM_NEXT   = 6'd38,
        S_RUN_RMS   = 6'd39,
        S_RUN_SILU  = 6'd40,
        S_RUN_ROPE  = 6'd41,
        S_RUN_SOFT  = 6'd42,
        S_RUN_ATTN  = 6'd43,
        S_TX_M1     = 6'd44,
        S_TX_M2     = 6'd45,
        S_TX_SO     = 6'd46,
        S_TX_DBG    = 6'd47,
        S_TX_O_RD   = 6'd48,
        S_TX_O_W    = 6'd49,
        S_TX_MH_RD  = 6'd50,
        S_TX_MH_W   = 6'd51,
        // SDRAM commands
        S_M2_W      = 7'd52,
        S_WW_A0     = 7'd53,
        S_WW_A1     = 7'd54,
        S_WW_A2     = 7'd55,
        S_WW_DATA   = 7'd56,
        S_WW_PULSE  = 7'd57,
        S_WW_WAIT   = 7'd58,
        S_WW_TX_W   = 7'd59,
        S_WW_TX_K   = 7'd60,
        S_M2_B      = 7'd61,
        S_BB_A0     = 7'd62,
        S_BB_A1     = 7'd63,
        S_BB_A2     = 7'd64,
        S_BB_PULSE  = 7'd65,
        S_BB_WAIT   = 7'd66,
        S_BB_TX_B   = 7'd67,
        S_BB_TX_K   = 7'd68,
        S_BB_TX_D   = 7'd69,
        // LL bulk load
        S_M2_L      = 7'd70,
        S_LL_A0     = 7'd71,
        S_LL_A1     = 7'd72,
        S_LL_A2     = 7'd73,
        S_LL_N0     = 7'd74,
        S_LL_N1     = 7'd75,
        S_LL_DATA   = 7'd76,
        S_LL_WR_W   = 7'd77,
        S_LL_NEXT   = 7'd78,
        S_LL_TX_L   = 7'd79,
        S_LL_TX_K   = 7'd80,
        // CC bulk dump
        S_M2_C      = 7'd81,
        S_CC_A0     = 7'd82,
        S_CC_A1     = 7'd83,
        S_CC_A2     = 7'd84,
        S_CC_N0     = 7'd85,
        S_CC_N1     = 7'd86,
        S_CC_TX_C   = 7'd87,
        S_CC_TX_K   = 7'd88,
        S_CC_RD     = 7'd89,
        S_CC_RD_W   = 7'd90,
        S_CC_TX_D   = 7'd91,
        // FN (RMSNorm with SDRAM weight fetch)
        S_M2_F      = 7'd92,
        S_FN_SX     = 7'd93,
        S_FN_SW     = 7'd94,
        S_FN_RX_X   = 7'd95,
        S_FN_A0     = 7'd96,
        S_FN_A1     = 7'd97,
        S_FN_A2     = 7'd98,
        S_FN_RD     = 7'd99,
        S_FN_RDW    = 7'd100,
        S_FN_WB     = 7'd101,
        // FM (matmul y=W.x with W in SDRAM, K=64 fixe)
        S_FM_N      = 7'd102,    // receives N
        S_FM_SX     = 7'd103,
        S_FM_SW     = 7'd104,
        S_FM_RX_X   = 7'd105,    // receives x[64]
        S_FM_A0     = 7'd106,
        S_FM_A1     = 7'd107,
        S_FM_A2     = 7'd108,
        S_FM_ROW    = 7'd109,    // init row (fetch_idx, dot_acc)
        S_FM_RD     = 7'd110,
        S_FM_RDW    = 7'd111,
        S_FM_WB     = 7'd112,    // wbuf[fetch_idx] <= sd_dout
        S_FM_DOT_A  = 7'd113,    // addr x[k] et w[k]
        S_FM_DOT_W1 = 7'd114,
        S_FM_DOT_W2 = 7'd115,
        S_FM_DOT_M  = 7'd116,    // acc += x*w
        S_FM_STORE  = 7'd117,    // ecrit y[row] (4 oct LE) in obuf
        S_FM_TX_F   = 7'd118,
        S_FM_TX_M   = 7'd119,
        // FQ phases (after compute, finds max + requantize)
        S_FQ_MAX_INIT  = 7'd120,
        S_FQ_MAX_LOOP  = 7'd121,
        S_FQ_SHIFT     = 7'd122,
        S_FQ_REQ_INIT  = 7'd123,
        S_FQ_REQ_LOOP  = 7'd124,
        S_FM_SETTLE    = 7'd125,
        S_FM_WARMUP_RD = 7'd126,
        S_FM_WARMUP_W  = 7'd127,
        // ---- Pattern v3 : etats communs ----
        S_OP_RD_BUSY   = 8'd128,    // attend busy=1 after pulse rd
        S_OP_RD_DONE   = 8'd129,    // attend busy=0, dout valide, -> next_state
        S_OP_WR_BUSY   = 8'd130,
        S_OP_WR_DONE   = 8'd131,
        S_REF_BUSY     = 8'd132,
        S_REF_DONE     = 8'd133,    // -> ret_state
        // CN (chain rmsnorm + matmul) - reuse FN et FQ via flag cn_active
        S_M2_CN_2      = 8'd134,
        S_CN_SX        = 8'd135,
        S_CN_SW_RMS    = 8'd136,
        S_CN_SW_MM     = 8'd137,
        S_CN_RX_X      = 8'd138,
        S_CN_N         = 8'd139,
        S_CN_RA0       = 8'd140,
        S_CN_RA1       = 8'd141,
        S_CN_RA2       = 8'd142,
        S_CN_MA0       = 8'd143,
        S_CN_MA1       = 8'd144,
        S_CN_MA2       = 8'd145,
        S_CN_COPY_RD   = 8'd146,
        S_CN_COPY_W1   = 8'd147,
        S_CN_COPY_W2   = 8'd148,
        S_CN_COPY_WB   = 8'd149,
        S_CN_SETUP_MM  = 8'd150,
        S_CN_TX_C      = 8'd151,
        S_CN_TX_N      = 8'd152,
        S_CN_TX_SO     = 8'd153,
        // CS extension : copy obuf->xbuf then silu
        S_CS_COPY_RD   = 8'd154,
        S_CS_COPY_W1   = 8'd155,
        S_CS_COPY_W2   = 8'd156,
        S_CS_COPY_WB   = 8'd157,
        S_CS_RUN_SILU  = 8'd158,
        S_CS_TX_C      = 8'd159,
        S_CS_TX_S      = 8'd160,
        S_CS_TX_SO     = 8'd161,
        S_CS_ZERO_PAD  = 8'd162,
        // EE (embedding lookup) : SDRAM[ADDR_TOK_EMB + tok*64] -> obuf[0..63] -> TX
        S_M2_E         = 8'd163,
        S_EE_T0        = 8'd164,
        S_EE_T1        = 8'd165,
        S_EE_RD        = 8'd166,
        S_EE_WB        = 8'd167,
        S_EE_TX_E      = 8'd168,
        S_EE_TX_K      = 8'd169,
        // GG v0 : embed lookup + rmsnorm layer 0 -> x_norm[64]
        S_M2_G         = 8'd170,
        S_GG_T0        = 8'd171,
        S_GG_T1        = 8'd172,
        S_GG_SE        = 8'd173,
        S_GG_SR        = 8'd174,
        S_GG_EMB_RD    = 8'd175,
        S_GG_EMB_WB    = 8'd176,
        S_GG_FN_RD     = 8'd177,
        S_GG_FN_WB     = 8'd178,
        S_GG_RUN_RMS   = 8'd179,
        S_GG_TX_G      = 8'd180,
        S_GG_TX_K      = 8'd181,
        S_GG_TX_SH     = 8'd182,
        // GG v1 : + Q matmul (Wq)
        S_GG_SQ        = 8'd183,
        S_GG_COPY_RD   = 8'd184,
        S_GG_COPY_W1   = 8'd185,
        S_GG_COPY_W2   = 8'd186,
        S_GG_COPY_WB   = 8'd187,
        S_GG_SETUP_Q   = 8'd188,
        // GG v2 : + K, V matmuls
        S_GG_SK        = 8'd189,
        S_GG_SV        = 8'd190,
        S_GG_SAVE_Q_RD = 8'd191,
        S_GG_SAVE_Q_W1 = 8'd192,
        S_GG_SAVE_Q_W2 = 8'd193,
        S_GG_SAVE_Q_WB = 8'd194,
        S_GG_SETUP_K   = 8'd195,
        S_GG_SAVE_K_RD = 8'd196,
        S_GG_SAVE_K_W1 = 8'd197,
        S_GG_SAVE_K_W2 = 8'd198,
        S_GG_SAVE_K_WB = 8'd199,
        S_GG_SETUP_V   = 8'd200,
        S_GG_SAVE_V_RD = 8'd201,
        S_GG_SAVE_V_W1 = 8'd202,
        S_GG_SAVE_V_W2 = 8'd203,
        S_GG_SAVE_V_WB = 8'd204,
        S_GG_TX_SK     = 8'd205,
        S_GG_TX_SV     = 8'd206,
        S_GG_TX_Q      = 8'd207,
        S_GG_TX_KD     = 8'd208,
        S_GG_TX_V      = 8'd209,
        // GG v3 : + multi-head attention (T=1, pos=0, no rope)
        S_GG_LOAD_KV   = 8'd210,   // copie gg_k_packed -> xbuf, gg_v_packed -> wbuf
        S_GG_SETUP_MM  = 8'd211,
        S_GG_TX_AG     = 8'd212,   // 'G' magic
        S_GG_TX_AK     = 8'd213,   // 'K' magic
        S_GG_TX_ASH    = 8'd214,   // shift attn
        S_GG_TX_AD     = 8'd215,   // attn_out[64] from Out_packed
        // GG v4 : + Wo + residual
        S_GG_SO        = 8'd216,   // RX sh_o (shift Wo)
        S_GG_LOAD_ATTN = 8'd217,   // copy Out_packed -> xbuf pour Wo input
        S_GG_SETUP_WO  = 8'd218,
        S_GG_RES_INIT  = 8'd219,   // init residual : x_orig + obuf
        S_GG_RES_RD    = 8'd220,   // obuf_raddr <= idx (BSRAM read setup)
        S_GG_RES_W1    = 8'd221,
        S_GG_RES_W2    = 8'd222,
        S_GG_RES_STORE = 8'd223,   // compute aligned sum, y_int32[i] <= sum
        S_GG_RES_REQ   = 8'd224,   // jump to FQ requantize flow
        S_GG_SAVE_X_RD = 8'd225,   // save new x (post-residual) in x_save_packed
        S_GG_SAVE_X_W1 = 8'd226,
        S_GG_SAVE_X_W2 = 8'd227,
        S_GG_SAVE_X_WB = 8'd228,
        S_GG_TX_XG     = 8'd229,   // 'G'
        S_GG_TX_XK     = 8'd230,   // 'K'
        S_GG_TX_XSH    = 8'd231,   // shift x
        S_GG_TX_XD     = 8'd232,   // x[64] setup obuf_raddr
        S_GG_TX_XD_W   = 8'd233,   // x[64] send
        // GG v5a : + rmsnorm FFN
        S_GG_SRF       = 8'd234,   // RX sh_rms_ffn
        S_GG_FFN_COPY_RD = 8'd235, // copy x_save_packed -> xbuf pour input rmsnorm_ffn
        S_GG_FFN_COPY_WB = 8'd236,
        S_GG_FFN_FETCH_RD = 8'd237, // DMA fetch rms_ffn -> wbuf
        S_GG_FFN_FETCH_WB = 8'd238,
        S_GG_FFN_RUN_RMS  = 8'd239,
        // GG v5b : + W1 chunk 0
        S_GG_SH1         = 8'd240,    // RX sh_h1
        S_GG_W1_COPY_RD  = 8'd241,    // copy obuf (x_norm_ffn) -> xbuf
        S_GG_W1_COPY_W1  = 8'd242,
        S_GG_W1_COPY_W2  = 8'd243,
        S_GG_W1_COPY_WB  = 8'd244,
        S_GG_SETUP_W1    = 8'd245,
        // GG v5c : saved chunks W1 + chunks 1+2 W1
        S_GG_SAVE_H1_CH0_RD = 8'd246,
        S_GG_SAVE_H1_CH0_W1 = 8'd247,
        S_GG_SAVE_H1_CH0_W2 = 8'd248,
        S_GG_SAVE_H1_CH0_WB = 8'd249,
        S_GG_SETUP_W1_CH1   = 8'd250,
        S_GG_SAVE_H1_CH1_RD = 8'd251,
        S_GG_SAVE_H1_CH1_W1 = 8'd252,
        S_GG_SAVE_H1_CH1_W2 = 8'd253,
        S_GG_SAVE_H1_CH1_WB = 8'd254,
        S_GG_SETUP_W1_CH2   = 9'd255,
        S_GG_SAVE_H1_CH2_RD = 9'd256,
        S_GG_SAVE_H1_CH2_W1 = 9'd257,
        S_GG_SAVE_H1_CH2_W2 = 9'd258,
        S_GG_SAVE_H1_CH2_WB = 9'd259,
        // TX format v5c : 'G' 'K' sh_ch0 sh_ch1 sh_ch2 h1[192]
        S_GG_TX_H1G   = 9'd260,
        S_GG_TX_H1K   = 9'd261,
        S_GG_TX_H1S0  = 9'd262,
        S_GG_TX_H1S1  = 9'd263,
        S_GG_TX_H1S2  = 9'd264,
        S_GG_TX_H1D   = 9'd265,
        S_GG_TX_H1D_W = 9'd266,
        // GG v5d : + W3 chunked
        S_GG_SH3      = 9'd267,    // RX sh_h3
        S_GG_SETUP_W3_CH0 = 9'd268,
        S_GG_SAVE_H3_CH0_RD = 9'd269,
        S_GG_SAVE_H3_CH0_W1 = 9'd270,
        S_GG_SAVE_H3_CH0_W2 = 9'd271,
        S_GG_SAVE_H3_CH0_WB = 9'd272,
        S_GG_SETUP_W3_CH1 = 9'd273,
        S_GG_SAVE_H3_CH1_RD = 9'd274,
        S_GG_SAVE_H3_CH1_W1 = 9'd275,
        S_GG_SAVE_H3_CH1_W2 = 9'd276,
        S_GG_SAVE_H3_CH1_WB = 9'd277,
        S_GG_SETUP_W3_CH2 = 9'd278,
        S_GG_SAVE_H3_CH2_RD = 9'd279,
        S_GG_SAVE_H3_CH2_W1 = 9'd280,
        S_GG_SAVE_H3_CH2_W2 = 9'd281,
        S_GG_SAVE_H3_CH2_WB = 9'd282,
        // TX h3 (similar to h1)
        S_GG_TX_H3G   = 9'd283,
        S_GG_TX_H3K   = 9'd284,
        S_GG_TX_H3S0  = 9'd285,
        S_GG_TX_H3S1  = 9'd286,
        S_GG_TX_H3S2  = 9'd287,
        S_GG_TX_H3D   = 9'd288,
        S_GG_TX_H3D_W = 9'd289,
        // GG v5e : silu chunked sur h1
        S_GG_SILU_LOAD_RD  = 9'd290,   // h1_packed[chunk*64 + idx] -> xbuf[idx]
        S_GG_SILU_LOAD_W1  = 9'd291,
        S_GG_SILU_LOAD_WB  = 9'd292,
        S_GG_SILU_START    = 9'd293,
        S_GG_SILU_WAIT     = 9'd294,
        S_GG_SILU_SAVE_RD  = 9'd295,
        S_GG_SILU_SAVE_W1  = 9'd296,
        S_GG_SILU_SAVE_W2  = 9'd297,
        S_GG_SILU_SAVE_WB  = 9'd298,
        S_GG_SILU_NEXT     = 9'd299,
        // TX silu_packed (test)
        S_GG_TX_SILUG  = 9'd300,
        S_GG_TX_SILUK  = 9'd301,
        S_GG_TX_SILUS0 = 9'd302,
        S_GG_TX_SILUS1 = 9'd303,
        S_GG_TX_SILUS2 = 9'd304,
        S_GG_TX_SILUD  = 9'd305,
        S_GG_TX_SILUDW = 9'd306,
        // GG v5f : multiply elementwise silu * h3 -> h_gated
        S_GG_MULT_INIT  = 9'd307,
        S_GG_MULT_RD    = 9'd308,
        S_GG_MULT_W1    = 9'd309,
        S_GG_MULT_W2    = 9'd310,
        S_GG_MULT_COMP  = 9'd311,
        S_GG_MULT2_INIT = 9'd312,
        S_GG_MULT2_RD   = 9'd313,
        S_GG_MULT2_W1   = 9'd314,
        S_GG_MULT2_W2   = 9'd315,
        S_GG_MULT2_STORE = 9'd316,
        // TX h_gated
        S_GG_TX_HGG  = 9'd317,
        S_GG_TX_HGK  = 9'd318,
        S_GG_TX_HGSH = 9'd319,
        S_GG_TX_HGD  = 9'd320,
        S_GG_TX_HGDW = 9'd321,
        // GG v5g : W2 chunked K + residual final
        S_GG_SW2          = 9'd322,   // RX sh_w2
        S_GG_W2_LOAD_RD   = 9'd323,   // copy h_gated[chunk*64+idx] -> xbuf
        S_GG_W2_LOAD_W1   = 9'd324,
        S_GG_W2_LOAD_WB   = 9'd325,
        S_GG_SETUP_W2     = 9'd326,
        S_GG_SAVE_P_RD    = 9'd327,   // save obuf -> partials_packed[chunk*64+idx]
        S_GG_SAVE_P_W1    = 9'd328,
        S_GG_SAVE_P_W2    = 9'd329,
        S_GG_SAVE_P_WB    = 9'd330,
        S_GG_W2_NEXT      = 9'd331,
        // Sum 3-way des partials + residual with x_save (= x_after_attn)
        S_GG_FFN_RES_INIT = 9'd332,
        S_GG_FFN_RES_RD   = 9'd333,
        S_GG_FFN_RES_W1   = 9'd334,
        S_GG_FFN_RES_W2   = 9'd335,
        S_GG_FFN_RES_STORE = 9'd336,
        // Sous-etats pour lectures sequentielles BSRAM partials
        S_GG_FRR_P0_W1    = 9'd337,
        S_GG_FRR_P0_R    = 9'd338,
        S_GG_FRR_P1_W1   = 9'd339,
        S_GG_FRR_P1_R    = 9'd340,
        S_GG_FRR_P2_W1   = 9'd341,
        S_GG_FRR_P2_R    = 9'd342;

    // compteurs LL/CC/FN/FM
    reg [15:0] bulk_n;
    reg [3:0]  bulk_delay;
    reg [6:0]  fetch_idx;
    // Pour FM / FQ (N etendu a 64 pour realistic matmul wq/wo)
    reg [6:0]  fm_row;             // 0..63
    reg [6:0]  fm_N;               // 1..64
    reg [6:0]  dot_k;
    reg signed [31:0] dot_acc;
    // Stockage int32 pour requantize (FQ uniquement, N <= 64)
    (* syn_ramstyle = "registers" *) reg signed [31:0] y_int32 [0:63];
    reg [31:0] max_abs;
    reg [5:0]  fq_lead_bit;
    reg [5:0]  fq_shift_out;
    reg signed [7:0] fq_shift_total;
    reg [6:0]  fq_idx;             // 0..63
    // Mult signed locaux (Gowin gotcha)
    reg signed [15:0] fm_xs16, fm_ws16;
    integer fq_i;
    always @(*) begin
        fm_xs16 = {{8{x_rdata_reg[7]}}, x_rdata_reg};
        fm_ws16 = {{8{w_rdata_reg[7]}}, w_rdata_reg};
        // leading bit position of max_abs (0..31)
        fq_lead_bit = 6'd0;
        for (fq_i = 0; fq_i < 32; fq_i = fq_i + 1)
            if (max_abs[fq_i]) fq_lead_bit = fq_i[5:0];
    end
    // Ports FM pour driver les BSRAMs durant dot product et store
    reg [9:0]        x_raddr_fm, w_raddr_fm;
    reg [9:0]        out_waddr_fm;
    reg signed [7:0] out_wdata_fm;
    reg              out_we_fm;

    // Tableaux debug (mux selon op)
    // RMSNorm dbg = 8 oct ; SiLU dbg = 3 oct ; RoPE dbg = 8 oct (new_real[4] + new_imag[4])
    wire [7:0] rms_dbg [0:7];
    wire [7:0] silu_dbg [0:2];
    wire [7:0] rope_dbg [0:7];
    assign rms_dbg[0] = dbg_acc[7:0];
    assign rms_dbg[1] = dbg_acc[15:8];
    assign rms_dbg[2] = dbg_acc[23:16];
    assign rms_dbg[3] = {3'd0, dbg_p};
    assign rms_dbg[4] = {2'd0, dbg_shift_amt};
    assign rms_dbg[5] = dbg_raw_inv[7:0];
    assign rms_dbg[6] = dbg_raw_inv[15:8];
    assign rms_dbg[7] = dbg_apply_shift;
    assign silu_dbg[0] = dbg_lut_idx;
    assign silu_dbg[1] = dbg_silu_int[7:0];
    assign silu_dbg[2] = dbg_silu_int[15:8];
    assign rope_dbg[0] = dbg_rope_real[7:0];
    assign rope_dbg[1] = dbg_rope_real[15:8];
    assign rope_dbg[2] = dbg_rope_real[23:16];
    assign rope_dbg[3] = dbg_rope_real[31:24];
    assign rope_dbg[4] = dbg_rope_imag[7:0];
    assign rope_dbg[5] = dbg_rope_imag[15:8];
    assign rope_dbg[6] = dbg_rope_imag[23:16];
    assign rope_dbg[7] = dbg_rope_imag[31:24];

    wire [7:0] soft_dbg [0:6];
    assign soft_dbg[0] = dbg_soft_max;
    assign soft_dbg[1] = dbg_soft_sum[7:0];
    assign soft_dbg[2] = dbg_soft_sum[15:8];
    assign soft_dbg[3] = dbg_soft_sum[23:16];
    assign soft_dbg[4] = dbg_soft_p_sum;
    assign soft_dbg[5] = dbg_soft_inv_sum[7:0];
    assign soft_dbg[6] = dbg_soft_inv_sum[15:8];

    // Attention dbg = 7 oct : score_last[2] max[1] exp_sum[2] inv_sum[2]
    wire [7:0] attn_dbg [0:6];
    assign attn_dbg[0] = dbg_attn_score_last[7:0];
    assign attn_dbg[1] = dbg_attn_score_last[15:8];
    assign attn_dbg[2] = dbg_attn_max_score;
    assign attn_dbg[3] = dbg_attn_exp_sum[7:0];
    assign attn_dbg[4] = dbg_attn_exp_sum[15:8];
    assign attn_dbg[5] = dbg_attn_inv_sum[7:0];
    assign attn_dbg[6] = dbg_attn_inv_sum[15:8];

    wire [9:0] dbg_n = op_attn ? 10'd7 :
                       op_soft ? 10'd7 :
                       op_rope ? 10'd8 :
                       op_silu ? 10'd3 : 10'd8;
    wire [9:0] out_n = (op_fm && fq_mode) ? {3'd0, fm_N} :         // FQ : N int8
                       op_rope ? 10'd8 :
                       op_soft ? 10'd32 :
                       op_attn ? 10'd8 :
                       op_fm   ? {1'd0, fm_N, 2'd0} :              // FM : N*4 bytes
                                 10'd64;

    reg [8:0] state;            // 9-bit (256+ pour GG v5 full)
    reg [8:0] ret_state;
    reg [8:0] next_state;
    reg [9:0] rx_idx, tx_idx;

    assign rx_consume = rx_pending && (
        state == S_IDLE     || state == S_M2_N    || state == S_M2_S   || state == S_M2_R ||
        state == S_M2_X     || state == S_M2_A    || state == S_M2_M   ||
        state == S_M2_W     || state == S_M2_B    ||
        state == S_NN_SX    || state == S_NN_SW   ||
        state == S_NN_RX_X  || state == S_NN_RX_W ||
        state == S_SS_SX    || state == S_SS_RX_X ||
        state == S_RR_SX    || state == S_RR_RX_X || state == S_RR_RX_COS || state == S_RR_RX_SIN ||
        state == S_XX_SX    || state == S_XX_RX_X ||
        state == S_AA_SQ    || state == S_AA_SK   || state == S_AA_SV || state == S_AA_T ||
        state == S_AA_RX_Q  || state == S_AA_RX_K || state == S_AA_RX_V ||
        state == S_MM_SQ    || state == S_MM_SK   || state == S_MM_SV || state == S_MM_T ||
        state == S_MM_RX_Q  || state == S_MM_RX_K || state == S_MM_RX_V ||
        state == S_WW_A0    || state == S_WW_A1   || state == S_WW_A2 || state == S_WW_DATA ||
        state == S_BB_A0    || state == S_BB_A1   || state == S_BB_A2 ||
        state == S_M2_L     || state == S_LL_A0   || state == S_LL_A1 || state == S_LL_A2 ||
        state == S_LL_N0    || state == S_LL_N1   || state == S_LL_DATA ||
        state == S_M2_C     || state == S_CC_A0   || state == S_CC_A1 || state == S_CC_A2 ||
        state == S_CC_N0    || state == S_CC_N1   ||
        state == S_M2_F     || state == S_FN_SX   || state == S_FN_SW || state == S_FN_RX_X ||
        state == S_FN_A0    || state == S_FN_A1   || state == S_FN_A2 ||
        state == S_FM_N     || state == S_FM_SX   || state == S_FM_SW || state == S_FM_RX_X ||
        state == S_FM_A0    || state == S_FM_A1   || state == S_FM_A2 ||
        state == S_CN_SX    || state == S_CN_SW_RMS || state == S_CN_SW_MM ||
        state == S_CN_N     || state == S_CN_RX_X ||
        state == S_CN_RA0   || state == S_CN_RA1  || state == S_CN_RA2 ||
        state == S_CN_MA0   || state == S_CN_MA1  || state == S_CN_MA2 ||
        state == S_CS_ZERO_PAD ||    // pas really RX mais we consume rien ici
        state == S_M2_E     || state == S_EE_T0   || state == S_EE_T1 ||
        state == S_M2_G     || state == S_GG_T0   || state == S_GG_T1  ||
        state == S_GG_SE    || state == S_GG_SR   || state == S_GG_SQ  ||
        state == S_GG_SK    || state == S_GG_SV   || state == S_GG_SO  ||
        state == S_GG_SRF   || state == S_GG_SH1  || state == S_GG_SH3 || state == S_GG_SW2
    );

    wire signed [7:0] cur_shift_out = (op_fm && fq_mode)  ? fq_shift_total :
                                      (op_attn || op_mh)  ? attn_shift_out :
                                      op_soft             ? soft_shift_out :
                                      op_rope             ? rope_shift_out :
                                      op_silu             ? silu_shift_out : rms_shift_out;
    // Debug TX bytes : envoyait acc/lut_idx/etc. Apres optimisation LUT, on send 0.
    // L'API Python reste identique (meme nombre de bytes), juste les valeurs sont zero.
    wire [7:0]        cur_dbg_byte  = 8'd0;

    always @(posedge clk_sys) begin
        if (rst) begin
            state      <= S_IDLE;
            tx_send    <= 1'b0;
            xbuf_we    <= 1'b0;
            wbuf_we    <= 1'b0;
            rms_start  <= 1'b0;
            silu_start <= 1'b0;
            rope_start <= 1'b0;
            soft_start <= 1'b0;
            attn_start <= 1'b0;
            sd_rd      <= 1'b0;
            sd_wr      <= 1'b0;
            sd_refresh <= 1'b0;
            out_we_fm  <= 1'b0;
            dot_acc    <= 32'sd0;
            op_sel     <= 4'd0;
            cn_active  <= 1'b0;
            cs_active  <= 1'b0;
            gg_active  <= 1'b0;
            ee_obuf_we <= 1'b0;
            gg_accumulate   <= 1'b0;
            gg_skip_requant <= 1'b0;
        end else begin
            tx_send    <= 1'b0;
            xbuf_we    <= 1'b0;
            wbuf_we    <= 1'b0;
            ee_obuf_we <= 1'b0;
            rms_start  <= 1'b0;
            silu_start <= 1'b0;
            rope_start <= 1'b0;
            soft_start <= 1'b0;
            attn_start <= 1'b0;
            sd_rd      <= 1'b0;
            sd_wr      <= 1'b0;
            sd_refresh <= 1'b0;
            out_we_fm  <= 1'b0;

            case (state)
                S_IDLE: if (rx_pending) begin
                    if      (rx_byte == "N") state <= S_M2_N;
                    else if (rx_byte == "S") state <= S_M2_S;
                    else if (rx_byte == "R") state <= S_M2_R;
                    else if (rx_byte == "X") state <= S_M2_X;
                    else if (rx_byte == "A") state <= S_M2_A;
                    else if (rx_byte == "M") state <= S_M2_M;
                    else if (rx_byte == "W") state <= S_M2_W;
                    else if (rx_byte == "B") state <= S_M2_B;
                    else if (rx_byte == "L") state <= S_M2_L;
                    else if (rx_byte == "C") state <= S_M2_C;
                    else if (rx_byte == "F") state <= S_M2_F;
                    else if (rx_byte == "E") state <= S_M2_E;
                    else if (rx_byte == "G") state <= S_M2_G;
                end
                S_M2_N: if (rx_pending) begin
                    if (rx_byte == "N") begin op_sel <= 4'd0; state <= S_NN_SX; end
                    else                state <= S_IDLE;
                end
                S_M2_S: if (rx_pending) begin
                    if (rx_byte == "S") begin op_sel <= 4'd1; state <= S_SS_SX; end
                    else                state <= S_IDLE;
                end
                S_M2_R: if (rx_pending) begin
                    if (rx_byte == "R") begin op_sel <= 4'd2; state <= S_RR_SX; end
                    else                state <= S_IDLE;
                end
                S_M2_X: if (rx_pending) begin
                    if (rx_byte == "X") begin op_sel <= 4'd3; state <= S_XX_SX; end
                    else                state <= S_IDLE;
                end
                S_M2_A: if (rx_pending) begin
                    if (rx_byte == "A") begin op_sel <= 4'd4; state <= S_AA_SQ; end
                    else                state <= S_IDLE;
                end

                // ---- NN (RMSNorm) ----
                S_NN_SX: if (rx_pending) begin
                    rms_shift_x <= $signed(rx_byte); state <= S_NN_SW;
                end
                S_NN_SW: if (rx_pending) begin
                    rms_shift_w <= $signed(rx_byte); rx_idx <= 10'd0; state <= S_NN_RX_X;
                end
                S_NN_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == D - 1) begin rx_idx <= 10'd0; state <= S_NN_RX_W; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_NN_RX_W: if (rx_pending) begin
                    wbuf_waddr <= rx_idx; wbuf_wdata <= $signed(rx_byte); wbuf_we <= 1'b1;
                    if (rx_idx == D - 1) begin rms_start <= 1'b1; state <= S_RUN_RMS; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_RUN_RMS: if (rms_done) begin
                    if (cn_active) begin rx_idx <= 10'd0; state <= S_CN_COPY_RD; end
                    else                                  state <= S_TX_M1;
                end

                // ---- SS (SiLU) ----
                S_SS_SX: if (rx_pending) begin
                    silu_shift_x <= $signed(rx_byte); rx_idx <= 10'd0; state <= S_SS_RX_X;
                end
                S_SS_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == D - 1) begin silu_start <= 1'b1; state <= S_RUN_SILU; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_RUN_SILU: if (silu_done) state <= S_TX_M1;

                // ---- RR (RoPE) : sx, x[8], cos[4]i16, sin[4]i16 = 25 oct payload ----
                S_RR_SX: if (rx_pending) begin
                    rope_shift_x <= $signed(rx_byte); rx_idx <= 10'd0; state <= S_RR_RX_X;
                end
                S_RR_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == 10'd7) begin rx_idx <= 10'd0; state <= S_RR_RX_COS; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_RR_RX_COS: if (rx_pending) begin
                    // 8 bytes = 4 int16 LE -> cos_reg[0..3]
                    if (rx_idx[0] == 1'b0)
                        cos_reg[rx_idx[2:1]][7:0]  <= rx_byte;
                    else
                        cos_reg[rx_idx[2:1]][15:8] <= rx_byte;
                    if (rx_idx == 10'd7) begin rx_idx <= 10'd0; state <= S_RR_RX_SIN; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_RR_RX_SIN: if (rx_pending) begin
                    if (rx_idx[0] == 1'b0)
                        sin_reg[rx_idx[2:1]][7:0]  <= rx_byte;
                    else
                        sin_reg[rx_idx[2:1]][15:8] <= rx_byte;
                    if (rx_idx == 10'd7) begin rope_start <= 1'b1; state <= S_RUN_ROPE; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_RUN_ROPE: if (rope_done) state <= S_TX_M1;

                // ---- XX (Softmax) : sx, x[32] = 33 oct payload ----
                S_XX_SX: if (rx_pending) begin
                    soft_shift_x <= $signed(rx_byte); rx_idx <= 10'd0; state <= S_XX_RX_X;
                end
                S_XX_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == 10'd31) begin soft_start <= 1'b1; state <= S_RUN_SOFT; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_RUN_SOFT: if (soft_done) state <= S_TX_M1;

                // ---- AA (Attention) : sq, sk, sv, T, Q[8], K[T*8], V[T*8] ----
                S_AA_SQ: if (rx_pending) begin
                    attn_shift_q <= $signed(rx_byte); state <= S_AA_SK;
                end
                S_AA_SK: if (rx_pending) begin
                    attn_shift_k <= $signed(rx_byte); state <= S_AA_SV;
                end
                S_AA_SV: if (rx_pending) begin
                    attn_shift_v <= $signed(rx_byte); state <= S_AA_T;
                end
                S_AA_T: if (rx_pending) begin
                    attn_T <= rx_byte[5:0];
                    attn_kv_stride <= 6'd8;     // single-head : K layout = [t][hs]
                    attn_kv_offset <= 6'd0;
                    mh_h <= 4'd0;
                    rx_idx <= 10'd0;
                    state <= S_AA_RX_Q;
                end
                S_AA_RX_Q: if (rx_pending) begin
                    // packe in Q_packed[0..7] (mh_h=0 alimentera Q_flat)
                    Q_packed[rx_idx[2:0]*8 +: 8] <= rx_byte;
                    if (rx_idx == 10'd7) begin rx_idx <= 10'd0; state <= S_AA_RX_K; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_AA_RX_K: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == ({1'd0, attn_T, 3'd0} - 10'd1)) begin    // T*8-1
                        rx_idx <= 10'd0; state <= S_AA_RX_V;
                    end else rx_idx <= rx_idx + 10'd1;
                end
                S_AA_RX_V: if (rx_pending) begin
                    wbuf_waddr <= rx_idx; wbuf_wdata <= $signed(rx_byte); wbuf_we <= 1'b1;
                    if (rx_idx == ({1'd0, attn_T, 3'd0} - 10'd1)) begin    // T*8-1
                        attn_start <= 1'b1; state <= S_RUN_ATTN;
                    end else rx_idx <= rx_idx + 10'd1;
                end
                S_RUN_ATTN: if (attn_done) state <= S_TX_M1;

                // ---- MM (Multi-head Attention) : sq sk sv T(1) Q[64] K[T*32] V[T*32] ----
                S_M2_M: if (rx_pending) begin
                    if (rx_byte == "M") begin op_sel <= 4'd5; state <= S_MM_SQ; end
                    else                state <= S_IDLE;
                end
                S_MM_SQ: if (rx_pending) begin
                    attn_shift_q <= $signed(rx_byte); state <= S_MM_SK;
                end
                S_MM_SK: if (rx_pending) begin
                    attn_shift_k <= $signed(rx_byte); state <= S_MM_SV;
                end
                S_MM_SV: if (rx_pending) begin
                    attn_shift_v <= $signed(rx_byte); state <= S_MM_T;
                end
                S_MM_T: if (rx_pending) begin
                    attn_T <= rx_byte[5:0];
                    attn_kv_stride <= 6'd32;   // multi-head GQA : K layout = [t][kvh][hs]
                    rx_idx <= 10'd0;
                    state  <= S_MM_RX_Q;
                end
                S_MM_RX_Q: if (rx_pending) begin
                    Q_packed[rx_idx[5:0]*8 +: 8] <= rx_byte;
                    if (rx_idx == 10'd63) begin rx_idx <= 10'd0; state <= S_MM_RX_K; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_MM_RX_K: if (rx_pending) begin
                    // K stocke in xbuf[0..T*32-1]
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == ({1'd0, attn_T, 5'd0} - 12'd1)) begin   // T*32-1, max 1023
                        rx_idx <= 10'd0; state <= S_MM_RX_V;
                    end else rx_idx <= rx_idx + 10'd1;
                end
                S_MM_RX_V: if (rx_pending) begin
                    wbuf_waddr <= rx_idx; wbuf_wdata <= $signed(rx_byte); wbuf_we <= 1'b1;
                    if (rx_idx == ({1'd0, attn_T, 5'd0} - 12'd1)) begin
                        mh_h <= 4'd0;
                        state <= S_MM_HEAD;
                    end else rx_idx <= rx_idx + 10'd1;
                end
                // loop multi-head : pour h=0..H-1, configure kv_offset, run attn, copie output
                S_MM_HEAD: begin
                    // kv_offset = kvh * HS = (h >> 1) * 8 = {h[3:1], 3'd0}
                    attn_kv_offset <= {1'b0, mh_h[3:1], 3'd0};
                    attn_start     <= 1'b1;
                    state          <= S_MM_WAIT;
                end
                S_MM_WAIT: if (attn_done) begin
                    rx_idx <= 10'd0;   // reuse comme copy index 0..7
                    state  <= S_MM_COPY1;
                end
                // Copie obuf[0..7] -> Out_packed[mh_h*64..mh_h*64+63]
                S_MM_COPY1: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_MM_COPY2;
                end
                S_MM_COPY2: state <= S_MM_COPY3;
                S_MM_COPY3: begin
                    // obuf_rdata_reg valide
                    Out_packed[(mh_h*64) + (rx_idx[2:0]*8) +: 8] <= obuf_rdata_reg;
                    if (rx_idx == 10'd7) state <= S_MM_NEXT;
                    else begin rx_idx <= rx_idx + 10'd1; state <= S_MM_COPY1; end
                end
                S_MM_NEXT: begin
                    if (mh_h == 4'd7) begin
                        if (gg_active) begin
                            op_sel <= 4'd10;         // restore op_gg
                            rx_idx <= 10'd0;
                            state  <= S_GG_LOAD_ATTN;
                        end else state <= S_TX_M1;
                    end
                    else begin mh_h <= mh_h + 4'd1; state <= S_MM_HEAD; end
                end

                // ---- WW (SDRAM write byte) ----
                S_M2_W: if (rx_pending) begin
                    if (rx_byte == "W") state <= S_WW_A0;
                    else                state <= S_IDLE;
                end
                S_WW_A0: if (rx_pending) begin
                    sd_addr[7:0]   <= rx_byte; state <= S_WW_A1;
                end
                S_WW_A1: if (rx_pending) begin
                    sd_addr[15:8]  <= rx_byte; state <= S_WW_A2;
                end
                S_WW_A2: if (rx_pending) begin
                    sd_addr[22:16] <= rx_byte[6:0]; state <= S_WW_DATA;
                end
                S_WW_DATA: if (rx_pending) begin
                    sd_din <= rx_byte; state <= S_WW_PULSE;
                end
                // v3 pattern : check refresh d'abord, sinon pulse wr then sync busy
                S_WW_PULSE: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_WW_PULSE; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_wr      <= 1'b1;
                        next_state <= S_WW_TX_W;
                        state      <= S_OP_WR_BUSY;
                    end
                end
                S_WW_TX_W: if (!tx_busy && !tx_send) begin
                    tx_data <= "W"; tx_send <= 1'b1; state <= S_WW_TX_K;
                end
                S_WW_TX_K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_IDLE;
                end

                // ---- BB (SDRAM read byte) ----
                S_M2_B: if (rx_pending) begin
                    if (rx_byte == "B") state <= S_BB_A0;
                    else                state <= S_IDLE;
                end
                S_BB_A0: if (rx_pending) begin
                    sd_addr[7:0]   <= rx_byte; state <= S_BB_A1;
                end
                S_BB_A1: if (rx_pending) begin
                    sd_addr[15:8]  <= rx_byte; state <= S_BB_A2;
                end
                S_BB_A2: if (rx_pending) begin
                    sd_addr[22:16] <= rx_byte[6:0]; state <= S_BB_PULSE;
                end
                S_BB_PULSE: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_BB_PULSE; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_BB_TX_B;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                S_BB_TX_B: if (!tx_busy && !tx_send) begin
                    tx_data <= "B"; tx_send <= 1'b1; state <= S_BB_TX_K;
                end
                S_BB_TX_K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_BB_TX_D;
                end
                S_BB_TX_D: if (!tx_busy && !tx_send) begin
                    tx_data <= sd_dout; tx_send <= 1'b1; state <= S_IDLE;
                end

                // ---- LL (bulk load) : addr[3] N[2] data[N] ----
                S_M2_L: if (rx_pending) begin
                    if (rx_byte == "L") state <= S_LL_A0;
                    else                state <= S_IDLE;
                end
                S_LL_A0: if (rx_pending) begin sd_addr[7:0]   <= rx_byte;       state <= S_LL_A1; end
                S_LL_A1: if (rx_pending) begin sd_addr[15:8]  <= rx_byte;       state <= S_LL_A2; end
                S_LL_A2: if (rx_pending) begin sd_addr[22:16] <= rx_byte[6:0];  state <= S_LL_N0; end
                S_LL_N0: if (rx_pending) begin bulk_n[7:0]    <= rx_byte;       state <= S_LL_N1; end
                S_LL_N1: if (rx_pending) begin bulk_n[15:8]   <= rx_byte;       state <= S_LL_DATA; end
                S_LL_DATA: if (rx_pending) begin
                    sd_din    <= rx_byte;
                    state     <= S_LL_WR_W;
                end
                // v3 pattern : check refresh, sinon pulse wr then sync busy
                S_LL_WR_W: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_LL_WR_W; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_wr      <= 1'b1;
                        next_state <= S_LL_NEXT;
                        state      <= S_OP_WR_BUSY;
                    end
                end
                S_LL_NEXT: begin
                    sd_addr <= sd_addr + 23'd1;
                    bulk_n  <= bulk_n - 16'd1;
                    if (bulk_n == 16'd1) state <= S_LL_TX_L;
                    else                 state <= S_LL_DATA;
                end
                S_LL_TX_L: if (!tx_busy && !tx_send) begin
                    tx_data <= "L"; tx_send <= 1'b1; state <= S_LL_TX_K;
                end
                S_LL_TX_K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_IDLE;
                end

                // ---- CC (bulk dump) : addr[3] N[2] -> data[N] ----
                S_M2_C: if (rx_pending) begin
                    if      (rx_byte == "C") state <= S_CC_A0;
                    else if (rx_byte == "N") begin
                        op_sel    <= 4'd8;
                        cn_active <= 1'b1;
                        state     <= S_CN_SX;
                    end
                    else if (rx_byte == "S") begin
                        // CS = CN + silu : reuse all CN RX states via cs_active
                        op_sel    <= 4'd8;
                        cn_active <= 1'b1;     // declenche CN flow
                        cs_active <= 1'b1;     // ET silu en plus after
                        state     <= S_CN_SX;
                    end
                    else state <= S_IDLE;
                end
                S_CC_A0: if (rx_pending) begin sd_addr[7:0]   <= rx_byte;       state <= S_CC_A1; end
                S_CC_A1: if (rx_pending) begin sd_addr[15:8]  <= rx_byte;       state <= S_CC_A2; end
                S_CC_A2: if (rx_pending) begin sd_addr[22:16] <= rx_byte[6:0];  state <= S_CC_N0; end
                S_CC_N0: if (rx_pending) begin bulk_n[7:0]    <= rx_byte;       state <= S_CC_N1; end
                S_CC_N1: if (rx_pending) begin bulk_n[15:8]   <= rx_byte;       state <= S_CC_TX_C; end
                S_CC_TX_C: if (!tx_busy && !tx_send) begin
                    tx_data <= "C"; tx_send <= 1'b1; state <= S_CC_TX_K;
                end
                S_CC_TX_K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_CC_RD;
                end
                S_CC_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_CC_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_CC_TX_D;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                // S_CC_RD_W obsolete (remplace par S_OP_RD_BUSY/DONE)
                S_CC_TX_D: if (!tx_busy && !tx_send) begin
                    tx_data <= sd_dout; tx_send <= 1'b1;
                    sd_addr <= sd_addr + 23'd1;
                    bulk_n  <= bulk_n - 16'd1;
                    if (bulk_n == 16'd1) state <= S_IDLE;
                    else                 state <= S_CC_RD;
                end

                // ---- FN (RMSNorm with w from SDRAM) : sx sw x[64] addr[3] ----
                S_M2_F: if (rx_pending) begin
                    if      (rx_byte == "N") begin op_sel <= 4'd6; fq_mode <= 1'b0; state <= S_FN_SX; end
                    else if (rx_byte == "M") begin op_sel <= 4'd7; fq_mode <= 1'b0; state <= S_FM_N; end
                    else if (rx_byte == "Q") begin op_sel <= 4'd7; fq_mode <= 1'b1; state <= S_FM_N; end
                    else                state <= S_IDLE;
                end
                S_FN_SX: if (rx_pending) begin
                    rms_shift_x <= $signed(rx_byte); state <= S_FN_SW;
                end
                S_FN_SW: if (rx_pending) begin
                    rms_shift_w <= $signed(rx_byte); rx_idx <= 10'd0; state <= S_FN_RX_X;
                end
                S_FN_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == 10'd63) begin rx_idx <= 10'd0; state <= S_FN_A0; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_FN_A0: if (rx_pending) begin sd_addr[7:0]   <= rx_byte;       state <= S_FN_A1; end
                S_FN_A1: if (rx_pending) begin sd_addr[15:8]  <= rx_byte;       state <= S_FN_A2; end
                S_FN_A2: if (rx_pending) begin
                    sd_addr[22:16] <= rx_byte[6:0];
                    fetch_idx <= 7'd0;
                    state     <= S_FN_RD;
                end
                // DMA fetch : SDRAM[sd_addr] -> wbuf[fetch_idx], 64 fois
                S_FN_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_FN_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_FN_WB;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                // S_FN_RDW obsolete
                S_FN_WB: begin
                    // ecrit in wbuf et avance
                    wbuf_waddr <= {3'd0, fetch_idx};
                    wbuf_wdata <= sd_dout;
                    wbuf_we    <= 1'b1;
                    sd_addr    <= sd_addr + 23'd1;
                    if (fetch_idx == 7'd63) begin
                        rms_start <= 1'b1;          // lance rmsnorm
                        state     <= S_RUN_RMS;
                    end else begin
                        fetch_idx <= fetch_idx + 7'd1;
                        state     <= S_FN_RD;
                    end
                end

                // ---- FM (matmul y[N]=W.x with W in SDRAM) ----
                S_FM_N:  if (rx_pending) begin fm_N <= rx_byte[6:0]; state <= S_FM_SX; end
                S_FM_SX: if (rx_pending) begin rms_shift_x <= $signed(rx_byte); state <= S_FM_SW; end
                S_FM_SW: if (rx_pending) begin rms_shift_w <= $signed(rx_byte); rx_idx <= 10'd0; state <= S_FM_RX_X; end
                S_FM_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == 10'd63) begin rx_idx <= 10'd0; state <= S_FM_A0; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_FM_A0: if (rx_pending) begin sd_addr[7:0]   <= rx_byte;       state <= S_FM_A1; end
                S_FM_A1: if (rx_pending) begin sd_addr[15:8]  <= rx_byte;       state <= S_FM_A2; end
                S_FM_A2: if (rx_pending) begin
                    sd_addr[22:16] <= rx_byte[6:0];
                    fm_row    <= 7'd0;
                    fetch_idx <= 7'd0;
                    bulk_delay<= 4'd0;
                    state     <= S_FM_WARMUP_RD;   // dummy read pour stabiliser SDRAM
                end
                // Warm-up (peut etre supprime with v3 pattern mais on garde par precaution)
                S_FM_WARMUP_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_FM_WARMUP_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_FM_RD;     // direct to la vraie loop
                        state      <= S_OP_RD_BUSY;
                    end
                end
                // S_FM_WARMUP_W et S_FM_SETTLE obsoletes

                // Phase 1: fetch 64 bytes de la line fm_row (v3 pattern)
                S_FM_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_FM_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_FM_WB;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                // S_FM_RDW obsolete
                S_FM_WB: begin
                    wbuf_waddr <= {3'd0, fetch_idx};
                    wbuf_wdata <= sd_dout;
                    wbuf_we    <= 1'b1;
                    sd_addr    <= sd_addr + 23'd1;
                    if (fetch_idx == 7'd63) begin
                        // line fetched. Init dot product.
                        // gg_accumulate=1 : start from y_int32[fm_row] (for W2 chunked sum)
                        fetch_idx <= 7'd0;
                        dot_k     <= 7'd0;
                        dot_acc   <= gg_accumulate ? y_int32[fm_row] : 32'sd0;
                        state     <= S_FM_DOT_A;
                    end else begin
                        fetch_idx <= fetch_idx + 7'd1;
                        state     <= S_FM_RD;
                    end
                end

                // Phase 2: dot product x[k]*w[k] sur K=64
                S_FM_DOT_A: begin
                    x_raddr_fm <= {3'd0, dot_k};
                    w_raddr_fm <= {3'd0, dot_k};
                    state      <= S_FM_DOT_W1;
                end
                S_FM_DOT_W1: state <= S_FM_DOT_W2;
                S_FM_DOT_W2: state <= S_FM_DOT_M;
                S_FM_DOT_M: begin
                    dot_acc <= dot_acc + (fm_xs16 * fm_ws16);
                    if (dot_k == 7'd63) begin
                        fetch_idx <= 7'd0;     // reuse comme byte index pour store
                        state <= S_FM_STORE;
                    end else begin
                        dot_k <= dot_k + 7'd1;
                        state <= S_FM_DOT_A;
                    end
                end

                // Phase 3: store y[row]
                S_FM_STORE: begin
                    if (fq_mode) begin
                        // FQ : stocke in y_int32 reg array
                        y_int32[fm_row] <= dot_acc;
                        if (fm_row == fm_N - 7'd1) begin
                            // gg_skip_requant : end FQ here, dispatch to next chunk via phase
                            if (gg_skip_requant) begin
                                op_sel <= 4'd10;
                                case (gg_qkv_phase)
                                    4'd11: state <= S_GG_W2_NEXT;  // post-W2 chunk N (accumulator filled, go next chunk)
                                    default: state <= S_FQ_MAX_INIT;
                                endcase
                            end else state <= S_FQ_MAX_INIT;
                        end
                        else begin
                            fm_row <= fm_row + 7'd1;
                            fetch_idx <= 7'd0;
                            state <= S_FM_RD;
                        end
                    end else begin
                        // FM : stocke int32 LE in obuf
                        case (fetch_idx[1:0])
                            2'd0: begin out_waddr_fm <= {fm_row[3:0], 2'd0}; out_wdata_fm <= dot_acc[7:0];   end
                            2'd1: begin out_waddr_fm <= {fm_row[3:0], 2'd1}; out_wdata_fm <= dot_acc[15:8];  end
                            2'd2: begin out_waddr_fm <= {fm_row[3:0], 2'd2}; out_wdata_fm <= dot_acc[23:16]; end
                            2'd3: begin out_waddr_fm <= {fm_row[3:0], 2'd3}; out_wdata_fm <= dot_acc[31:24]; end
                        endcase
                        out_we_fm <= 1'b1;
                        if (fetch_idx[1:0] == 2'd3) begin
                            if (fm_row == fm_N - 7'd1) state <= S_TX_M1;
                            else begin
                                fm_row    <= fm_row + 7'd1;
                                fetch_idx <= 7'd0;
                                state     <= S_FM_RD;
                            end
                        end else begin
                            fetch_idx <= fetch_idx + 7'd1;
                        end
                    end
                end

                // ---- FQ requantize phases ----
                // Pass 1 : find max(|y[i]|)
                S_FQ_MAX_INIT: begin
                    max_abs <= 32'd0;
                    fq_idx  <= 7'd0;
                    state   <= S_FQ_MAX_LOOP;
                end
                S_FQ_MAX_LOOP: begin
                    begin: abs_blk
                        reg signed [31:0] yv;
                        reg [31:0]        absy;
                        yv   = y_int32[fq_idx];
                        absy = yv[31] ? (~yv + 32'd1) : yv;   // abs
                        if (absy > max_abs) max_abs <= absy;
                    end
                    if (fq_idx == fm_N - 7'd1) state <= S_FQ_SHIFT;
                    else fq_idx <= fq_idx + 7'd1;
                end
                // computes shift_out = max(0, leading_bit(max_abs) - 6)
                S_FQ_SHIFT: begin
                    // fq_lead_bit computes combinationnel
                    fq_shift_out   <= (fq_lead_bit > 6'd6) ? (fq_lead_bit - 6'd6) : 6'd0;
                    fq_shift_total <= rms_shift_x + rms_shift_w
                                      + $signed({2'd0, ((fq_lead_bit > 6'd6) ? (fq_lead_bit - 6'd6) : 6'd0)});
                    fq_idx <= 7'd0;
                    state  <= S_FQ_REQ_INIT;
                end
                S_FQ_REQ_INIT: state <= S_FQ_REQ_LOOP;
                // Pass 2 : ecrit y_int8 in obuf[fq_idx]
                S_FQ_REQ_LOOP: begin
                    begin: req_blk
                        reg signed [31:0] yv;
                        reg signed [31:0] shifted;
                        reg signed [31:0] rounding;
                        reg signed [7:0]  clipped;
                        yv = y_int32[fq_idx];
                        rounding = (fq_shift_out > 6'd0) ? (32'sd1 <<< (fq_shift_out - 6'd1)) : 32'sd0;
                        shifted  = (yv + rounding) >>> fq_shift_out;
                        clipped  = (shifted > 32'sd127)  ? 8'sd127  :
                                   (shifted < -32'sd128) ? -8'sd128 :
                                                            shifted[7:0];
                        out_waddr_fm <= {3'd0, fq_idx};
                        out_wdata_fm <= clipped;
                        out_we_fm    <= 1'b1;
                    end
                    if (fq_idx == fm_N - 7'd1) begin
                        if (cs_active)      begin rx_idx <= 10'd0; state <= S_CS_COPY_RD; end
                        else if (cn_active) state <= S_CN_TX_C;
                        else if (gg_active) begin
                            op_sel <= 4'd10;
                            rx_idx <= 10'd0;
                            case (gg_qkv_phase)
                                4'd0: state <= S_GG_SAVE_Q_RD;
                                4'd1: state <= S_GG_SAVE_K_RD;
                                4'd2: state <= S_GG_SAVE_V_RD;
                                4'd3: state <= S_GG_RES_INIT;
                                4'd4: state <= S_GG_SAVE_X_RD;
                                4'd5: state <= S_GG_SAVE_H1_CH0_RD;
                                4'd6: state <= S_GG_SAVE_H1_CH1_RD;
                                4'd7: state <= S_GG_SAVE_H1_CH2_RD;
                                4'd8: state <= S_GG_SAVE_H3_CH0_RD;
                                4'd9: state <= S_GG_SAVE_H3_CH1_RD;
                                4'd10: state <= S_GG_SAVE_H3_CH2_RD;
                                4'd11: state <= S_GG_RES_INIT;            // post-W2 last chunk : lance residual v4 (2-way avec x_save)
                                default: state <= S_GG_SAVE_Q_RD;
                            endcase
                            rx_idx <= 10'd0;
                        end
                        else                state <= S_TX_M1;
                    end
                    else fq_idx <= fq_idx + 7'd1;
                end

                // ---- CS : copie obuf->xbuf then silu ----
                S_CS_COPY_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_CS_COPY_W1;
                end
                S_CS_COPY_W1: state <= S_CS_COPY_W2;
                S_CS_COPY_W2: state <= S_CS_COPY_WB;
                S_CS_COPY_WB: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= obuf_rdata_reg;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == ({3'd0, fm_N} - 10'd1)) begin
                        rx_idx <= {3'd0, fm_N};
                        state  <= S_CS_ZERO_PAD;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_CS_COPY_RD;
                    end
                end
                S_CS_ZERO_PAD: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= 8'd0;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        silu_shift_x <= fq_shift_total;
                        silu_start   <= 1'b1;
                        op_sel       <= 4'd1;
                        state        <= S_CS_RUN_SILU;
                    end else rx_idx <= rx_idx + 10'd1;
                end
                S_CS_RUN_SILU: if (silu_done) state <= S_CS_TX_C;

                S_CS_TX_C: if (!tx_busy && !tx_send) begin
                    tx_data <= "C"; tx_send <= 1'b1; state <= S_CS_TX_S;
                end
                S_CS_TX_S: if (!tx_busy && !tx_send) begin
                    tx_data <= "S"; tx_send <= 1'b1; state <= S_CS_TX_SO;
                end
                S_CS_TX_SO: if (!tx_busy && !tx_send) begin
                    tx_data <= silu_shift_out; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_TX_O_RD;     // reuse TX flow pour envoyer obuf
                end

                // ---- EE : embedding lookup ----
                // RX : 'E' 'E' tok_lo tok_hi  -> 4 bytes total
                // TX : 'E' 'K' x[64]          -> 66 bytes
                S_M2_E: if (rx_pending) begin
                    if (rx_byte == "E") begin
                        op_sel    <= 4'd9;
                        fetch_idx <= 7'd0;
                        state     <= S_EE_T0;
                    end else state <= S_IDLE;
                end
                S_EE_T0: if (rx_pending) begin
                    // Capture token bits [7:0] in sd_addr[13:6] (= tok_lo * 64)
                    sd_addr[13:6] <= rx_byte;
                    sd_addr[5:0]  <= 6'd0;
                    state         <= S_EE_T1;
                end
                S_EE_T1: if (rx_pending) begin
                    // Capture token bits [15:8] in sd_addr[21:14] (= tok_hi * 64 * 256)
                    sd_addr[21:14] <= rx_byte;
                    sd_addr[22]    <= 1'b0;
                    state          <= S_EE_RD;
                end
                // DMA fetch : SDRAM[sd_addr] -> obuf[fetch_idx], 64 fois
                S_EE_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_EE_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_EE_WB;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                S_EE_WB: begin
                    // ecrit in obuf et avance (via ee_obuf_we mux)
                    ee_obuf_waddr <= fetch_idx[5:0];
                    ee_obuf_wdata <= sd_dout;
                    ee_obuf_we    <= 1'b1;
                    sd_addr       <= sd_addr + 23'd1;
                    if (fetch_idx == 7'd63) begin
                        state <= S_EE_TX_E;
                    end else begin
                        fetch_idx <= fetch_idx + 7'd1;
                        state     <= S_EE_RD;
                    end
                end
                S_EE_TX_E: if (!tx_busy && !tx_send) begin
                    tx_data <= "E"; tx_send <= 1'b1; state <= S_EE_TX_K;
                end
                S_EE_TX_K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_TX_O_RD;     // reuse TX flow pour envoyer obuf[0..63]
                end

                // ---- GG v0 : embed + rmsnorm L0 -> x_norm[64] ----
                // RX : 'G' 'G' tok_lo tok_hi sh_emb sh_rms_att  (6 bytes)
                // TX : 'G' 'K' shift_out x_norm[64]              (67 bytes)
                S_M2_G: if (rx_pending) begin
                    if (rx_byte == "G") begin
                        op_sel          <= 4'd10;
                        gg_active       <= 1'b1;
                        gg_qkv_phase    <= 4'd0;
                        gg_accumulate   <= 1'b0;     // default off for all non-W2 matmuls
                        gg_skip_requant <= 1'b0;
                        state           <= S_GG_T0;
                    end else state <= S_IDLE;
                end
                S_GG_T0: if (rx_pending) begin
                    sd_addr[13:6] <= rx_byte;   // tok_lo * 64 (offset low)
                    sd_addr[5:0]  <= 6'd0;
                    state         <= S_GG_T1;
                end
                S_GG_T1: if (rx_pending) begin
                    sd_addr[21:14] <= rx_byte;  // tok_hi * 64 * 256
                    sd_addr[22]    <= 1'b0;
                    state          <= S_GG_SE;
                end
                S_GG_SE: if (rx_pending) begin
                    rms_shift_x <= $signed(rx_byte);
                    gg_x_shift  <= $signed(rx_byte);   // sauve sh_emb pour residual
                    state       <= S_GG_SR;
                end
                S_GG_SR: if (rx_pending) begin
                    // sh_rms_att = shift_w pour rmsnorm
                    rms_shift_w <= $signed(rx_byte);
                    state       <= S_GG_SQ;
                end
                S_GG_SQ: if (rx_pending) begin
                    gg_sh_q <= $signed(rx_byte);
                    state   <= S_GG_SK;
                end
                S_GG_SK: if (rx_pending) begin
                    gg_sh_k <= $signed(rx_byte);
                    state   <= S_GG_SV;
                end
                S_GG_SV: if (rx_pending) begin
                    gg_sh_v <= $signed(rx_byte);
                    state   <= S_GG_SO;
                end
                S_GG_SO: if (rx_pending) begin
                    gg_sh_o <= $signed(rx_byte);
                    state   <= S_GG_SRF;
                end
                S_GG_SRF: if (rx_pending) begin
                    gg_sh_rf <= $signed(rx_byte);
                    state    <= S_GG_SH1;
                end
                S_GG_SH1: if (rx_pending) begin
                    gg_sh_h1 <= $signed(rx_byte);
                    state    <= S_GG_SH3;
                end
                S_GG_SH3: if (rx_pending) begin
                    gg_sh_h3 <= $signed(rx_byte);
                    state    <= S_GG_SW2;
                end
                S_GG_SW2: if (rx_pending) begin
                    gg_sh_w2   <= $signed(rx_byte);
                    fetch_idx  <= 7'd0;
                    state      <= S_GG_EMB_RD;
                end
                // Phase 1 : DMA fetch embed (64 bytes) from SDRAM[tok*64] to xbuf
                S_GG_EMB_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_GG_EMB_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_GG_EMB_WB;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                S_GG_EMB_WB: begin
                    xbuf_waddr <= {3'd0, fetch_idx};
                    xbuf_wdata <= sd_dout;
                    xbuf_we    <= 1'b1;
                    // in parallel : save embed in x_save_packed pour residual
                    x_save_packed[fetch_idx[5:0]*8 +: 8] <= sd_dout;
                    sd_addr    <= sd_addr + 23'd1;
                    if (fetch_idx == 7'd63) begin
                        // Embed load in xbuf. Maintenant fetch rms_w[L0] in wbuf.
                        sd_addr   <= ADDR_RMS_ATT_L0;
                        fetch_idx <= 7'd0;
                        state     <= S_GG_FN_RD;
                    end else begin
                        fetch_idx <= fetch_idx + 7'd1;
                        state     <= S_GG_EMB_RD;
                    end
                end
                // Phase 2 : DMA fetch rms_w (64 bytes) from ADDR_RMS_ATT_L0 to wbuf
                S_GG_FN_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_GG_FN_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_GG_FN_WB;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                S_GG_FN_WB: begin
                    wbuf_waddr <= {3'd0, fetch_idx};
                    wbuf_wdata <= sd_dout;
                    wbuf_we    <= 1'b1;
                    sd_addr    <= sd_addr + 23'd1;
                    if (fetch_idx == 7'd63) begin
                        rms_start <= 1'b1;
                        state     <= S_GG_RUN_RMS;
                    end else begin
                        fetch_idx <= fetch_idx + 7'd1;
                        state     <= S_GG_FN_RD;
                    end
                end
                // Phase 3 : run rmsnorm (resultat in obuf via rms_out_we mux)
                S_GG_RUN_RMS: if (rms_done) begin
                    rx_idx <= 10'd0;        // reuse comme copy index
                    state  <= S_GG_COPY_RD;
                end
                // Phase 4 : copie obuf -> xbuf (pour input du matmul Q)
                S_GG_COPY_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_COPY_W1;
                end
                S_GG_COPY_W1: state <= S_GG_COPY_W2;
                S_GG_COPY_W2: state <= S_GG_COPY_WB;
                S_GG_COPY_WB: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= obuf_rdata_reg;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        rx_idx <= 10'd0;
                        state  <= S_GG_SETUP_Q;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_COPY_RD;
                    end
                end
                // Phase 5 : setup matmul Q (Wq @ x_norm), reuse flow FQ
                S_GG_SETUP_Q: begin
                    sd_addr     <= ADDR_WQ_L0;
                    rms_shift_x <= rms_shift_out;   // input shift = rmsnorm output shift
                    rms_shift_w <= gg_sh_q;         // weight shift = sh_q
                    op_sel      <= 4'd7;            // bascule sur op_fm pour FQ
                    fq_mode     <= 1'b1;
                    fm_N        <= 7'd64;           // H * HS = 64
                    fm_row      <= 7'd0;
                    fetch_idx   <= 7'd0;
                    state       <= S_FM_WARMUP_RD;
                end
                // ---- Save Q : copy obuf[0..63] -> Q_packed ----
                S_GG_SAVE_Q_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_Q_W1;
                end
                S_GG_SAVE_Q_W1: state <= S_GG_SAVE_Q_W2;
                S_GG_SAVE_Q_W2: state <= S_GG_SAVE_Q_WB;
                S_GG_SAVE_Q_WB: begin
                    Q_packed[rx_idx[5:0]*8 +: 8] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_q_out  <= fq_shift_total;
                        gg_qkv_phase <= 4'd1;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_K;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_Q_RD;
                    end
                end
                // ---- Setup matmul K (Wk) ----
                S_GG_SETUP_K: begin
                    sd_addr     <= ADDR_WK_L0;
                    rms_shift_x <= rms_shift_out;   // input = x_norm shift
                    rms_shift_w <= gg_sh_k;
                    op_sel      <= 4'd7;
                    fq_mode     <= 1'b1;
                    fm_N        <= 7'd32;            // KH * HS
                    fm_row      <= 7'd0;
                    fetch_idx   <= 7'd0;
                    state       <= S_FM_WARMUP_RD;
                end
                // ---- Save K : copy obuf[0..31] -> gg_k_packed ----
                S_GG_SAVE_K_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_K_W1;
                end
                S_GG_SAVE_K_W1: state <= S_GG_SAVE_K_W2;
                S_GG_SAVE_K_W2: state <= S_GG_SAVE_K_WB;
                S_GG_SAVE_K_WB: begin
                    gg_k_packed[rx_idx[4:0]*8 +: 8] <= obuf_rdata_reg;
                    if (rx_idx == 10'd31) begin
                        gg_sh_k_out  <= fq_shift_total;
                        gg_qkv_phase <= 4'd2;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_V;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_K_RD;
                    end
                end
                // ---- Setup matmul V (Wv) ----
                S_GG_SETUP_V: begin
                    sd_addr     <= ADDR_WV_L0;
                    rms_shift_x <= rms_shift_out;
                    rms_shift_w <= gg_sh_v;
                    op_sel      <= 4'd7;
                    fq_mode     <= 1'b1;
                    fm_N        <= 7'd32;
                    fm_row      <= 7'd0;
                    fetch_idx   <= 7'd0;
                    state       <= S_FM_WARMUP_RD;
                end
                // ---- Save V : copy obuf[0..31] -> gg_v_packed ----
                S_GG_SAVE_V_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_V_W1;
                end
                S_GG_SAVE_V_W1: state <= S_GG_SAVE_V_W2;
                S_GG_SAVE_V_W2: state <= S_GG_SAVE_V_WB;
                S_GG_SAVE_V_WB: begin
                    gg_v_packed[rx_idx[4:0]*8 +: 8] <= obuf_rdata_reg;
                    if (rx_idx == 10'd31) begin
                        gg_sh_v_out <= fq_shift_total;
                        rx_idx      <= 10'd0;
                        state       <= S_GG_LOAD_KV;     // v3 : passe a attention
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_V_RD;
                    end
                end
                // ---- v3 : load K et V from packed regs to xbuf et wbuf ----
                S_GG_LOAD_KV: begin
                    // T=1 : K[0..31] -> xbuf[0..31], V[0..31] -> wbuf[0..31]
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= gg_k_packed[rx_idx[4:0]*8 +: 8];
                    xbuf_we    <= 1'b1;
                    wbuf_waddr <= rx_idx;
                    wbuf_wdata <= gg_v_packed[rx_idx[4:0]*8 +: 8];
                    wbuf_we    <= 1'b1;
                    if (rx_idx == 10'd31) state <= S_GG_SETUP_MM;
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_GG_SETUP_MM: begin
                    // Q already in Q_packed (= Q_flat in MM)
                    attn_shift_q   <= gg_sh_q_out;
                    attn_shift_k   <= gg_sh_k_out;
                    attn_shift_v   <= gg_sh_v_out;
                    attn_T         <= 6'd1;
                    attn_kv_stride <= 6'd32;       // multi-head : [t][kvh][hs]
                    mh_h           <= 4'd0;
                    op_sel         <= 4'd5;        // op_mh pour loop MM
                    state          <= S_MM_HEAD;   // reuse flow MM existant
                end
                // ---- TX response v3 : 'G' 'K' shift_attn attn_out[64] ----
                S_GG_TX_AG: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_AK;
                end
                S_GG_TX_AK: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_ASH;
                end
                S_GG_TX_ASH: if (!tx_busy && !tx_send) begin
                    tx_data <= attn_shift_out;     // = sh_v after fix v4.5t
                    tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_AD;
                end
                S_GG_TX_AD: if (!tx_busy && !tx_send) begin
                    tx_data <= Out_packed[tx_idx[5:0]*8 +: 8];
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd63) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else tx_idx <= tx_idx + 10'd1;
                end

                // ---- v4 : load attn_out from Out_packed to xbuf ----
                S_GG_LOAD_ATTN: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= Out_packed[rx_idx[5:0]*8 +: 8];
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) state <= S_GG_SETUP_WO;
                    else rx_idx <= rx_idx + 10'd1;
                end
                // ---- v4 : setup matmul Wo ----
                S_GG_SETUP_WO: begin
                    sd_addr      <= ADDR_WO_L0;
                    rms_shift_x  <= attn_shift_out;     // input = output MM
                    rms_shift_w  <= gg_sh_o;
                    op_sel       <= 4'd7;               // op_fm pour FQ
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;              // output D=64
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd3;               // phase 3 = post-Wo
                    state        <= S_FM_WARMUP_RD;
                end
                // ---- v4 : residual init (after Wo, obuf contient Wo_out) ----
                S_GG_RES_INIT: begin
                    max_abs <= 32'd0;
                    fq_idx  <= 7'd0;
                    state   <= S_GG_RES_RD;
                end
                S_GG_RES_RD: begin
                    obuf_raddr <= {3'd0, fq_idx};
                    state      <= S_GG_RES_W1;
                end
                S_GG_RES_W1: state <= S_GG_RES_W2;
                S_GG_RES_W2: state <= S_GG_RES_STORE;
                S_GG_RES_STORE: begin
                    // residual : x_orig + Wo_out, aligne sur le plus petit shift
                    // gg_x_shift reste constant pendant tout le residual (= sh du x_save)
                    begin: residual_blk
                        reg signed [7:0]  x_orig_i8;
                        reg signed [7:0]  out_i8;
                        reg signed [31:0] x_i32, out_i32, sum_i32;
                        reg [31:0]        absy;
                        reg signed [7:0]  sh_x_diff, sh_o_diff;
                        x_orig_i8 = $signed(x_save_packed[{fq_idx[5:0], 3'b0} +: 8]);
                        out_i8    = obuf_rdata_reg;
                        // sh_min = min(gg_x_shift, fq_shift_total)
                        if (gg_x_shift < fq_shift_total) begin
                            sh_x_diff = 8'sd0;
                            sh_o_diff = fq_shift_total - gg_x_shift;
                        end else begin
                            sh_x_diff = gg_x_shift - fq_shift_total;
                            sh_o_diff = 8'sd0;
                        end
                        x_i32   = $signed({{24{x_orig_i8[7]}}, x_orig_i8}) <<< sh_x_diff;
                        out_i32 = $signed({{24{out_i8[7]}},    out_i8})    <<< sh_o_diff;
                        sum_i32 = x_i32 + out_i32;
                        y_int32[fq_idx] <= sum_i32;
                        absy = sum_i32[31] ? (~sum_i32 + 32'd1) : sum_i32;
                        if (absy > max_abs) max_abs <= absy;
                    end
                    if (fq_idx == 7'd63) begin
                        // Setup pour FQ_SHIFT : rms_shift_x = sh_min, rms_shift_w = 0
                        // fq_shift_total = sh_min + add_shift after requantize
                        op_sel       <= 4'd7;       // FQ pour ecrire obuf via out_we_fm
                        fq_mode      <= 1'b1;
                        fm_N         <= 7'd64;
                        rms_shift_x  <= (gg_x_shift < fq_shift_total) ? gg_x_shift : fq_shift_total;
                        rms_shift_w  <= 8'sd0;
                        gg_qkv_phase <= 4'd4;
                        fq_idx <= 7'd0;
                        state  <= S_FQ_SHIFT;
                    end else begin
                        fq_idx <= fq_idx + 7'd1;
                        state  <= S_GG_RES_RD;
                    end
                end
                // after FQ requantize (obuf contient x_after_residual int8)
                // -> save x in x_save_packed pour le prochain layer
                S_GG_SAVE_X_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_X_W1;
                end
                S_GG_SAVE_X_W1: state <= S_GG_SAVE_X_W2;
                S_GG_SAVE_X_W2: state <= S_GG_SAVE_X_WB;
                S_GG_SAVE_X_WB: begin
                    x_save_packed[rx_idx[5:0]*8 +: 8] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        // Update gg_x_shift = fq_shift_total (le shift final after residual+requant)
                        gg_x_shift <= fq_shift_total;
                        op_sel     <= 4'd10;
                        rx_idx     <= 10'd0;
                        state      <= S_GG_FFN_COPY_RD;    // chain to FFN rmsnorm
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_X_RD;
                    end
                end
                // ---- v5a : FFN RMSNorm ----
                // Copy x_save_packed -> xbuf (input du rmsnorm_ffn)
                S_GG_FFN_COPY_RD: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= x_save_packed[rx_idx[5:0]*8 +: 8];
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        sd_addr   <= ADDR_RMS_FFN_L0;
                        fetch_idx <= 7'd0;
                        state     <= S_GG_FFN_FETCH_RD;
                    end else rx_idx <= rx_idx + 10'd1;
                end
                // DMA fetch rms_ffn -> wbuf
                S_GG_FFN_FETCH_RD: begin
                    if (refresh_due && !sd_busy) begin
                        sd_refresh <= 1'b1; ret_state <= S_GG_FFN_FETCH_RD; state <= S_REF_BUSY;
                    end else if (!sd_busy) begin
                        sd_rd      <= 1'b1;
                        next_state <= S_GG_FFN_FETCH_WB;
                        state      <= S_OP_RD_BUSY;
                    end
                end
                S_GG_FFN_FETCH_WB: begin
                    wbuf_waddr <= {3'd0, fetch_idx};
                    wbuf_wdata <= sd_dout;
                    wbuf_we    <= 1'b1;
                    sd_addr    <= sd_addr + 23'd1;
                    if (fetch_idx == 7'd63) begin
                        // Setup rmsnorm pour FFN : shift_x = gg_x_shift (= sh after residual), shift_w = sh_rms_ffn
                        rms_shift_x <= gg_x_shift;
                        rms_shift_w <= gg_sh_rf;
                        rms_start   <= 1'b1;
                        state       <= S_GG_FFN_RUN_RMS;
                    end else begin
                        fetch_idx <= fetch_idx + 7'd1;
                        state     <= S_GG_FFN_FETCH_RD;
                    end
                end
                S_GG_FFN_RUN_RMS: if (rms_done) begin
                    rx_idx <= 10'd0;
                    state  <= S_GG_W1_COPY_RD;     // chain to W1 matmul
                end
                // ---- v5b : copy x_norm_ffn (obuf) -> xbuf, then setup W1 ----
                S_GG_W1_COPY_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_W1_COPY_W1;
                end
                S_GG_W1_COPY_W1: state <= S_GG_W1_COPY_W2;
                S_GG_W1_COPY_W2: state <= S_GG_W1_COPY_WB;
                S_GG_W1_COPY_WB: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= obuf_rdata_reg;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        rx_idx <= 10'd0;
                        state  <= S_GG_SETUP_W1;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_W1_COPY_RD;
                    end
                end
                // Setup W1 chunk 0 (N=64)
                S_GG_SETUP_W1: begin
                    sd_addr      <= ADDR_W1_L0;
                    rms_shift_x  <= rms_shift_out;
                    rms_shift_w  <= gg_sh_h1;
                    op_sel       <= 4'd7;
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd5;
                    state        <= S_FM_WARMUP_RD;
                end
                // Save h1 chunk 0 : obuf -> h1_packed[0..63]
                S_GG_SAVE_H1_CH0_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_H1_CH0_W1;
                end
                S_GG_SAVE_H1_CH0_W1: state <= S_GG_SAVE_H1_CH0_W2;
                S_GG_SAVE_H1_CH0_W2: state <= S_GG_SAVE_H1_CH0_WB;
                S_GG_SAVE_H1_CH0_WB: begin
                    h1_packed[rx_idx[7:0]] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_h1_ch0 <= fq_shift_total;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_W1_CH1;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_H1_CH0_RD;
                    end
                end
                // Setup W1 chunk 1 (N=64, addr +4096)
                S_GG_SETUP_W1_CH1: begin
                    sd_addr      <= ADDR_W1_L0 + 23'd4096;
                    rms_shift_x  <= rms_shift_out;
                    rms_shift_w  <= gg_sh_h1;
                    op_sel       <= 4'd7;
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd6;
                    state        <= S_FM_WARMUP_RD;
                end
                S_GG_SAVE_H1_CH1_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_H1_CH1_W1;
                end
                S_GG_SAVE_H1_CH1_W1: state <= S_GG_SAVE_H1_CH1_W2;
                S_GG_SAVE_H1_CH1_W2: state <= S_GG_SAVE_H1_CH1_WB;
                S_GG_SAVE_H1_CH1_WB: begin
                    h1_packed[8'd64 + rx_idx[7:0]] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_h1_ch1 <= fq_shift_total;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_W1_CH2;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_H1_CH1_RD;
                    end
                end
                // Setup W1 chunk 2 (N=64, addr +8192)
                S_GG_SETUP_W1_CH2: begin
                    sd_addr      <= ADDR_W1_L0 + 23'd8192;
                    rms_shift_x  <= rms_shift_out;
                    rms_shift_w  <= gg_sh_h1;
                    op_sel       <= 4'd7;
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd7;
                    state        <= S_FM_WARMUP_RD;
                end
                S_GG_SAVE_H1_CH2_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_H1_CH2_W1;
                end
                S_GG_SAVE_H1_CH2_W1: state <= S_GG_SAVE_H1_CH2_W2;
                S_GG_SAVE_H1_CH2_W2: state <= S_GG_SAVE_H1_CH2_WB;
                S_GG_SAVE_H1_CH2_WB: begin
                    h1_packed[8'd128 + rx_idx[7:0]] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_h1_ch2 <= fq_shift_total;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_W3_CH0;     // chain to W3
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_H1_CH2_RD;
                    end
                end
                // ---- v5d : W3 chunked (3 chunks de N=64) ----
                S_GG_SETUP_W3_CH0: begin
                    sd_addr      <= ADDR_W3_L0;
                    rms_shift_x  <= rms_shift_out;      // input = x_norm_ffn shift
                    rms_shift_w  <= gg_sh_h3;
                    op_sel       <= 4'd7;
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd8;
                    state        <= S_FM_WARMUP_RD;
                end
                S_GG_SAVE_H3_CH0_RD: begin obuf_raddr <= rx_idx; state <= S_GG_SAVE_H3_CH0_W1; end
                S_GG_SAVE_H3_CH0_W1: state <= S_GG_SAVE_H3_CH0_W2;
                S_GG_SAVE_H3_CH0_W2: state <= S_GG_SAVE_H3_CH0_WB;
                S_GG_SAVE_H3_CH0_WB: begin
                    h3_packed[rx_idx[7:0]] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_h3_ch0 <= fq_shift_total;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_W3_CH1;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_H3_CH0_RD;
                    end
                end
                S_GG_SETUP_W3_CH1: begin
                    sd_addr      <= ADDR_W3_L0 + 23'd4096;
                    rms_shift_x  <= rms_shift_out;
                    rms_shift_w  <= gg_sh_h3;
                    op_sel       <= 4'd7;
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd9;
                    state        <= S_FM_WARMUP_RD;
                end
                S_GG_SAVE_H3_CH1_RD: begin obuf_raddr <= rx_idx; state <= S_GG_SAVE_H3_CH1_W1; end
                S_GG_SAVE_H3_CH1_W1: state <= S_GG_SAVE_H3_CH1_W2;
                S_GG_SAVE_H3_CH1_W2: state <= S_GG_SAVE_H3_CH1_WB;
                S_GG_SAVE_H3_CH1_WB: begin
                    h3_packed[8'd64 + rx_idx[7:0]] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_h3_ch1 <= fq_shift_total;
                        rx_idx       <= 10'd0;
                        state        <= S_GG_SETUP_W3_CH2;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_H3_CH1_RD;
                    end
                end
                S_GG_SETUP_W3_CH2: begin
                    sd_addr      <= ADDR_W3_L0 + 23'd8192;
                    rms_shift_x  <= rms_shift_out;
                    rms_shift_w  <= gg_sh_h3;
                    op_sel       <= 4'd7;
                    fq_mode      <= 1'b1;
                    fm_N         <= 7'd64;
                    fm_row       <= 7'd0;
                    fetch_idx    <= 7'd0;
                    gg_qkv_phase <= 4'd10;
                    state        <= S_FM_WARMUP_RD;
                end
                S_GG_SAVE_H3_CH2_RD: begin obuf_raddr <= rx_idx; state <= S_GG_SAVE_H3_CH2_W1; end
                S_GG_SAVE_H3_CH2_W1: state <= S_GG_SAVE_H3_CH2_W2;
                S_GG_SAVE_H3_CH2_W2: state <= S_GG_SAVE_H3_CH2_WB;
                S_GG_SAVE_H3_CH2_WB: begin
                    h3_packed[8'd128 + rx_idx[7:0]] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        gg_sh_h3_ch2  <= fq_shift_total;
                        rx_idx        <= 10'd0;
                        gg_silu_chunk <= 2'd0;
                        op_sel        <= 4'd10;
                        state         <= S_GG_SILU_LOAD_RD;    // chain to silu
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_H3_CH2_RD;
                    end
                end
                // ---- v5e : silu chunked sur h1 (3 chunks de 64) ----
                // Load h1_packed[chunk*64+idx] -> xbuf[idx]
                S_GG_SILU_LOAD_RD: begin
                    h1_raddr <= {gg_silu_chunk, rx_idx[5:0]};   // chunk*64 + idx
                    state    <= S_GG_SILU_LOAD_W1;
                end
                S_GG_SILU_LOAD_W1: state <= S_GG_SILU_LOAD_WB;
                S_GG_SILU_LOAD_WB: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= h1_rdata_reg;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        rx_idx <= 10'd0;
                        state  <= S_GG_SILU_START;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SILU_LOAD_RD;
                    end
                end
                S_GG_SILU_START: begin
                    op_sel       <= 4'd1;        // op_silu
                    silu_shift_x <= (gg_silu_chunk == 2'd0) ? gg_sh_h1_ch0 :
                                    (gg_silu_chunk == 2'd1) ? gg_sh_h1_ch1 :
                                                              gg_sh_h1_ch2;
                    silu_start   <= 1'b1;
                    state        <= S_GG_SILU_WAIT;
                end
                S_GG_SILU_WAIT: if (silu_done) begin
                    // Save silu shift for this chunk
                    case (gg_silu_chunk)
                        2'd0: gg_sh_silu_ch0 <= silu_shift_out;
                        2'd1: gg_sh_silu_ch1 <= silu_shift_out;
                        2'd2: gg_sh_silu_ch2 <= silu_shift_out;
                    endcase
                    op_sel <= 4'd10;             // restore op_gg
                    rx_idx <= 10'd0;
                    state  <= S_GG_SILU_SAVE_RD;
                end
                // Save obuf[0..63] -> silu_packed[chunk*64 .. chunk*64+63]
                S_GG_SILU_SAVE_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SILU_SAVE_W1;
                end
                S_GG_SILU_SAVE_W1: state <= S_GG_SILU_SAVE_W2;
                S_GG_SILU_SAVE_W2: state <= S_GG_SILU_SAVE_WB;
                S_GG_SILU_SAVE_WB: begin
                    silu_packed[{gg_silu_chunk, rx_idx[5:0]}] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        state <= S_GG_SILU_NEXT;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SILU_SAVE_RD;
                    end
                end
                S_GG_SILU_NEXT: begin
                    if (gg_silu_chunk == 2'd2) begin
                        // All 3 chunks silu done : start multiply phase
                        state <= S_GG_MULT_INIT;
                    end else begin
                        gg_silu_chunk <= gg_silu_chunk + 2'd1;
                        rx_idx        <= 10'd0;
                        state         <= S_GG_SILU_LOAD_RD;
                    end
                end
                // TX silu_packed (test only)
                S_GG_TX_SILUG: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_SILUK;
                end
                S_GG_TX_SILUK: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_SILUS0;
                end
                S_GG_TX_SILUS0: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_silu_ch0; tx_send <= 1'b1; state <= S_GG_TX_SILUS1;
                end
                S_GG_TX_SILUS1: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_silu_ch1; tx_send <= 1'b1; state <= S_GG_TX_SILUS2;
                end
                S_GG_TX_SILUS2: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_silu_ch2; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_SILUD;
                end
                S_GG_TX_SILUD: begin
                    silu_raddr <= tx_idx[7:0];
                    state      <= S_GG_TX_SILUDW;
                end
                S_GG_TX_SILUDW: if (!tx_busy && !tx_send) begin
                    tx_data <= silu_rdata_reg;
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd191) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        tx_idx <= tx_idx + 10'd1;
                        state  <= S_GG_TX_SILUD;
                    end
                end
                // ---- v5f : multiply elementwise silu * h3 ----
                // Pass 1 : compute prod_i32 et track max_abs (i=0..191)
                S_GG_MULT_INIT: begin
                    gg_mult_i       <= 8'd0;
                    gg_mult_max_abs <= 32'd0;
                    state           <= S_GG_MULT_RD;
                end
                S_GG_MULT_RD: begin
                    silu_raddr <= gg_mult_i;
                    h3_raddr   <= gg_mult_i;
                    state      <= S_GG_MULT_W1;
                end
                S_GG_MULT_W1: state <= S_GG_MULT_W2;
                S_GG_MULT_W2: state <= S_GG_MULT_COMP;
                S_GG_MULT_COMP: begin
                    begin: mult_blk
                        reg signed [15:0] silu_v, h3_v;
                        reg signed [31:0] prod_i32;
                        reg signed [7:0]  sh_extra;
                        reg [31:0]        absp;
                        // sign extend silu et h3 a 16 bits
                        silu_v = {{8{silu_rdata_reg[7]}}, silu_rdata_reg};
                        h3_v   = {{8{h3_rdata_reg[7]}},   h3_rdata_reg};
                        prod_i32 = silu_v * h3_v;        // i16 * i16 = i32 (mais holds in 16 typically)
                        // Aligner au shift common via shift_extra du chunk
                        case (gg_mult_i[7:6])
                            2'd0: sh_extra = sh_extra_0;
                            2'd1: sh_extra = sh_extra_1;
                            default: sh_extra = sh_extra_2;
                        endcase
                        prod_i32 = prod_i32 <<< sh_extra;
                        tmp_prod[gg_mult_i] <= prod_i32;
                        absp = prod_i32[31] ? (~prod_i32 + 32'd1) : prod_i32;
                        if (absp > gg_mult_max_abs) gg_mult_max_abs <= absp;
                    end
                    if (gg_mult_i == 8'd191) begin
                        // Pass 1 done. Lance Pass 2.
                        state <= S_GG_MULT2_INIT;
                    end else begin
                        gg_mult_i <= gg_mult_i + 8'd1;
                        state     <= S_GG_MULT_RD;
                    end
                end
                // Pass 2 : compute h_gated[i] = clip(tmp_prod[i] >> add_shift)
                S_GG_MULT2_INIT: begin
                    gg_mult_i     <= 8'd0;
                    gg_sh_h_gated <= sh_prod_min + $signed({2'd0, gg_mult_add_shift});
                    state         <= S_GG_MULT2_RD;
                end
                S_GG_MULT2_RD: begin
                    tmp_prod_raddr <= gg_mult_i;
                    state          <= S_GG_MULT2_W1;
                end
                S_GG_MULT2_W1: state <= S_GG_MULT2_W2;
                S_GG_MULT2_W2: state <= S_GG_MULT2_STORE;
                S_GG_MULT2_STORE: begin
                    begin: store_blk
                        reg signed [31:0] shifted, rounding;
                        reg signed [7:0]  clipped;
                        rounding = (gg_mult_add_shift > 6'd0) ? (32'sd1 <<< (gg_mult_add_shift - 6'd1)) : 32'sd0;
                        shifted  = (tmp_prod_rdata_reg + rounding) >>> gg_mult_add_shift;
                        clipped  = (shifted > 32'sd127)  ? 8'sd127 :
                                   (shifted < -32'sd128) ? -8'sd128 :
                                                            shifted[7:0];
                        h_gated_packed[gg_mult_i] <= clipped;
                    end
                    if (gg_mult_i == 8'd191) begin
                        gg_w2_chunk <= 2'd0;
                        rx_idx      <= 10'd0;
                        state       <= S_GG_W2_LOAD_RD;     // lance W2 chunked
                    end else begin
                        gg_mult_i <= gg_mult_i + 8'd1;
                        state     <= S_GG_MULT2_RD;
                    end
                end
                // TX h_gated[192] : 'G' 'K' shift h_gated
                S_GG_TX_HGG: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_HGK;
                end
                S_GG_TX_HGK: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_HGSH;
                end
                S_GG_TX_HGSH: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h_gated; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_HGD;
                end
                S_GG_TX_HGD: begin
                    h_gated_raddr <= tx_idx[7:0];
                    state         <= S_GG_TX_HGDW;
                end
                S_GG_TX_HGDW: if (!tx_busy && !tx_send) begin
                    tx_data <= h_gated_rdata_reg;
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd191) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        tx_idx <= tx_idx + 10'd1;
                        state  <= S_GG_TX_HGD;
                    end
                end
                // ---- v5g : W2 chunked K + residual ----
                // Load h_gated[chunk*64+i] -> xbuf[i]
                S_GG_W2_LOAD_RD: begin
                    h_gated_raddr <= {gg_w2_chunk, rx_idx[5:0]};
                    state         <= S_GG_W2_LOAD_W1;
                end
                S_GG_W2_LOAD_W1: state <= S_GG_W2_LOAD_WB;
                S_GG_W2_LOAD_WB: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= h_gated_rdata_reg;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        rx_idx <= 10'd0;
                        state  <= S_GG_SETUP_W2;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_W2_LOAD_RD;
                    end
                end
                // Setup matmul W2 chunk : addr = ADDR_W2_L0 + chunk*4096
                // gg_accumulate = (chunk > 0)  : chunks 1,2 accumulate dans y_int32
                // gg_skip_requant = (chunk < 2) : chunks 0,1 skip MAX/REQ_LOOP (intermediate)
                S_GG_SETUP_W2: begin
                    sd_addr         <= ADDR_W2_L0 + ({21'd0, gg_w2_chunk} <<< 12);
                    rms_shift_x     <= gg_sh_h_gated;
                    rms_shift_w     <= gg_sh_w2;
                    op_sel          <= 4'd7;
                    fq_mode         <= 1'b1;
                    fm_N            <= 7'd64;
                    fm_row          <= 7'd0;
                    fetch_idx       <= 7'd0;
                    gg_accumulate   <= (gg_w2_chunk != 2'd0);   // first chunk : start at 0
                    gg_skip_requant <= (gg_w2_chunk != 2'd2);   // last chunk : do requant
                    gg_qkv_phase    <= 4'd11;
                    state           <= S_FM_WARMUP_RD;
                end
                // Save obuf[0..63] -> partials_packed[chunk*64..chunk*64+63]
                S_GG_SAVE_P_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_GG_SAVE_P_W1;
                end
                S_GG_SAVE_P_W1: state <= S_GG_SAVE_P_W2;
                S_GG_SAVE_P_W2: state <= S_GG_SAVE_P_WB;
                S_GG_SAVE_P_WB: begin
                    partials_packed[{gg_w2_chunk, rx_idx[5:0]}] <= obuf_rdata_reg;
                    if (rx_idx == 10'd63) begin
                        case (gg_w2_chunk)
                            2'd0: gg_sh_p0 <= fq_shift_total;
                            2'd1: gg_sh_p1 <= fq_shift_total;
                            2'd2: gg_sh_p2 <= fq_shift_total;
                        endcase
                        state <= S_GG_W2_NEXT;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_GG_SAVE_P_RD;
                    end
                end
                S_GG_W2_NEXT: begin
                    // With accumulate flag, no partials BSRAM needed.
                    // Chunks 0,1 arrive here via gg_skip_requant in S_FM_STORE.
                    // Chunk 2 goes through MAX/REQ_LOOP, then RES_INIT directly.
                    gg_w2_chunk <= gg_w2_chunk + 2'd1;
                    rx_idx      <= 10'd0;
                    state       <= S_GG_W2_LOAD_RD;
                end
                // Sum 3-way des partials + x_save (= x_after_attn) en int32 align
                // shifts impliques : gg_sh_p0, gg_sh_p1, gg_sh_p2, gg_x_shift (x_save)
                // Align tout au shift min, somme, requantize, store in obuf
                S_GG_FFN_RES_INIT: begin
                    max_abs <= 32'd0;
                    fq_idx  <= 7'd0;
                    state   <= S_GG_FFN_RES_RD;
                end
                // Lectures sequentielles BSRAM partials (p0, p1, p2)
                S_GG_FFN_RES_RD: begin
                    partials_raddr <= {2'd0, fq_idx[5:0]};
                    state          <= S_GG_FRR_P0_W1;
                end
                S_GG_FRR_P0_W1: state <= S_GG_FRR_P0_R;
                S_GG_FRR_P0_R: begin
                    gg_p0_temp     <= partials_rdata_reg;
                    partials_raddr <= {2'd1, fq_idx[5:0]};
                    state          <= S_GG_FRR_P1_W1;
                end
                S_GG_FRR_P1_W1: state <= S_GG_FRR_P1_R;
                S_GG_FRR_P1_R: begin
                    gg_p1_temp     <= partials_rdata_reg;
                    partials_raddr <= {2'd2, fq_idx[5:0]};
                    state          <= S_GG_FRR_P2_W1;
                end
                S_GG_FRR_P2_W1: state <= S_GG_FRR_P2_R;
                S_GG_FRR_P2_R: state  <= S_GG_FFN_RES_STORE;
                S_GG_FFN_RES_STORE: begin
                    begin: ffn_res_blk
                        reg signed [31:0] p0_i32, p1_i32, p2_i32, x_i32;
                        reg signed [7:0]  sh_min;
                        reg signed [7:0]  p0_extra, p1_extra, p2_extra, x_extra;
                        reg signed [31:0] sum_i32;
                        reg [31:0]        absy;
                        sh_min = gg_sh_p0;
                        if (gg_sh_p1 < sh_min) sh_min = gg_sh_p1;
                        if (gg_sh_p2 < sh_min) sh_min = gg_sh_p2;
                        if (gg_x_shift < sh_min) sh_min = gg_x_shift;
                        p0_extra = gg_sh_p0   - sh_min;
                        p1_extra = gg_sh_p1   - sh_min;
                        p2_extra = gg_sh_p2   - sh_min;
                        x_extra  = gg_x_shift - sh_min;
                        begin: xsave_extract
                            reg signed [7:0] x_byte;
                            x_byte = x_save_packed[fq_idx[5:0]*8 +: 8];
                            p0_i32 = $signed({{24{gg_p0_temp[7]}}, gg_p0_temp}) <<< p0_extra;
                            p1_i32 = $signed({{24{gg_p1_temp[7]}}, gg_p1_temp}) <<< p1_extra;
                            p2_i32 = $signed({{24{partials_rdata_reg[7]}}, partials_rdata_reg}) <<< p2_extra;
                            x_i32  = $signed({{24{x_byte[7]}}, x_byte}) <<< x_extra;
                        end
                        sum_i32 = p0_i32 + p1_i32 + p2_i32 + x_i32;
                        y_int32[fq_idx] <= sum_i32;
                        absy = sum_i32[31] ? (~sum_i32 + 32'd1) : sum_i32;
                        if (absy > max_abs) max_abs <= absy;
                    end
                    if (fq_idx == 7'd63) begin
                        op_sel       <= 4'd7;
                        fq_mode      <= 1'b1;
                        fm_N         <= 7'd64;
                        rms_shift_x  <= (gg_sh_p0 <= gg_sh_p1 && gg_sh_p0 <= gg_sh_p2 && gg_sh_p0 <= gg_x_shift) ? gg_sh_p0 :
                                        (gg_sh_p1 <= gg_sh_p2 && gg_sh_p1 <= gg_x_shift) ? gg_sh_p1 :
                                        (gg_sh_p2 <= gg_x_shift) ? gg_sh_p2 : gg_x_shift;
                        rms_shift_w  <= 8'sd0;
                        gg_qkv_phase <= 4'd4;     // post-residual -> SAVE_X (reuse v4 flow)
                        fq_idx       <= 7'd0;
                        state        <= S_FQ_SHIFT;
                    end else begin
                        fq_idx <= fq_idx + 7'd1;
                        state  <= S_GG_FFN_RES_RD;
                    end
                end
                // TX h3 (test only - similaire h1)
                S_GG_TX_H3G: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_H3K;
                end
                S_GG_TX_H3K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_H3S0;
                end
                S_GG_TX_H3S0: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h3_ch0; tx_send <= 1'b1; state <= S_GG_TX_H3S1;
                end
                S_GG_TX_H3S1: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h3_ch1; tx_send <= 1'b1; state <= S_GG_TX_H3S2;
                end
                S_GG_TX_H3S2: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h3_ch2; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_H3D;
                end
                S_GG_TX_H3D: begin
                    h3_raddr <= tx_idx[7:0];
                    state    <= S_GG_TX_H3D_W;
                end
                S_GG_TX_H3D_W: if (!tx_busy && !tx_send) begin
                    tx_data <= h3_rdata_reg;
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd191) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        tx_idx <= tx_idx + 10'd1;
                        state  <= S_GG_TX_H3D;
                    end
                end
                // TX h1[192] : 'G' 'K' sh_ch0 sh_ch1 sh_ch2 h1[192]
                S_GG_TX_H1G: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_H1K;
                end
                S_GG_TX_H1K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_H1S0;
                end
                S_GG_TX_H1S0: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h1_ch0; tx_send <= 1'b1; state <= S_GG_TX_H1S1;
                end
                S_GG_TX_H1S1: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h1_ch1; tx_send <= 1'b1; state <= S_GG_TX_H1S2;
                end
                S_GG_TX_H1S2: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_h1_ch2; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_H1D;
                end
                // Pattern BSRAM-style : set h1_raddr immediatement, attente UART = settle
                S_GG_TX_H1D: begin
                    h1_raddr <= tx_idx[7:0];
                    state    <= S_GG_TX_H1D_W;
                end
                S_GG_TX_H1D_W: if (!tx_busy && !tx_send) begin
                    tx_data <= h1_rdata_reg;
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd191) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        tx_idx <= tx_idx + 10'd1;
                        state  <= S_GG_TX_H1D;
                    end
                end
                S_GG_TX_XG: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_XK;
                end
                S_GG_TX_XK: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_XSH;
                end
                S_GG_TX_XSH: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_x_shift;
                    tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_XD;
                end
                // Pattern identical a S_TX_O_RD : set obuf_raddr immediatement, then
                // attente UART in le W state donne le temps au BSRAM (2 cycles).
                S_GG_TX_XD: begin
                    obuf_raddr <= tx_idx;
                    state      <= S_GG_TX_XD_W;
                end
                S_GG_TX_XD_W: if (!tx_busy && !tx_send) begin
                    tx_data <= obuf_rdata_reg;
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd63) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else begin
                        tx_idx <= tx_idx + 10'd1;
                        state  <= S_GG_TX_XD;
                    end
                end
                // ---- TX response v2 : 'G' 'K' sh_q sh_k sh_v Q[64] K[32] V[32] ----
                S_GG_TX_G: if (!tx_busy && !tx_send) begin
                    tx_data <= "G"; tx_send <= 1'b1; state <= S_GG_TX_K;
                end
                S_GG_TX_K: if (!tx_busy && !tx_send) begin
                    tx_data <= "K"; tx_send <= 1'b1; state <= S_GG_TX_SH;
                end
                S_GG_TX_SH: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_q_out; tx_send <= 1'b1; state <= S_GG_TX_SK;
                end
                S_GG_TX_SK: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_k_out; tx_send <= 1'b1; state <= S_GG_TX_SV;
                end
                S_GG_TX_SV: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_sh_v_out; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_GG_TX_Q;
                end
                S_GG_TX_Q: if (!tx_busy && !tx_send) begin
                    tx_data <= Q_packed[tx_idx[5:0]*8 +: 8];
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd63) begin tx_idx <= 10'd0; state <= S_GG_TX_KD; end
                    else tx_idx <= tx_idx + 10'd1;
                end
                S_GG_TX_KD: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_k_packed[tx_idx[4:0]*8 +: 8];
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd31) begin tx_idx <= 10'd0; state <= S_GG_TX_V; end
                    else tx_idx <= tx_idx + 10'd1;
                end
                S_GG_TX_V: if (!tx_busy && !tx_send) begin
                    tx_data <= gg_v_packed[tx_idx[4:0]*8 +: 8];
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd31) begin
                        gg_active <= 1'b0;
                        state     <= S_IDLE;
                    end else tx_idx <= tx_idx + 10'd1;
                end

                // ---- TX common ----
                S_TX_M1: if (!tx_busy && !tx_send) begin
                    tx_data <= (op_fn || op_fm) ? "F" :
                               op_mh   ? "M" :
                               op_attn ? "A" :
                               op_soft ? "X" :
                               op_rope ? "R" :
                               op_silu ? "S" : "N";
                    tx_send <= 1'b1; state <= S_TX_M2;
                end
                S_TX_M2: if (!tx_busy && !tx_send) begin
                    tx_data <= (op_fm && fq_mode) ? "Q" :
                                op_fm             ? "M" : "K";
                    tx_send <= 1'b1;
                    if (op_fm) tx_idx <= 10'd0;
                    // FQ : sends shift_total then obuf
                    state   <= (op_fm && fq_mode) ? S_TX_SO :       // SO sera shift_total
                               op_fm              ? S_TX_O_RD :
                               op_mh              ? S_TX_SO   : S_TX_SO;
                end
                S_TX_SO: if (!tx_busy && !tx_send) begin
                    tx_data <= cur_shift_out; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= (op_fm && fq_mode) ? S_TX_O_RD :    // FQ : N int8 in obuf
                               op_mh              ? S_TX_MH_W : S_TX_DBG;
                end
                S_TX_DBG: if (!tx_busy && !tx_send) begin
                    tx_data <= cur_dbg_byte; tx_send <= 1'b1;
                    if (tx_idx == dbg_n - 1) begin
                        tx_idx <= 10'd0; obuf_raddr <= 10'd0; state <= S_TX_O_RD;
                    end else tx_idx <= tx_idx + 10'd1;
                end
                S_TX_O_RD: begin
                    obuf_raddr <= tx_idx; state <= S_TX_O_W;
                end
                S_TX_O_W: if (!tx_busy && !tx_send) begin
                    tx_data <= obuf_rdata_reg; tx_send <= 1'b1;
                    if (tx_idx == out_n - 1) begin
                        cn_active <= 1'b0;
                        cs_active <= 1'b0;
                        gg_active <= 1'b0;
                        state <= S_IDLE;
                    end else begin
                        tx_idx <= tx_idx + 10'd1;
                        state  <= S_TX_O_RD;
                    end
                end
                // Multi-head TX : sends Out_packed (64 bytes) - source directe pas BSRAM
                S_TX_MH_W: if (!tx_busy && !tx_send) begin
                    tx_data <= Out_packed[tx_idx[5:0]*8 +: 8];
                    tx_send <= 1'b1;
                    if (tx_idx == 10'd63) state <= S_IDLE;
                    else tx_idx <= tx_idx + 10'd1;
                end
                // ---- CN : Chain rmsnorm + matmul ----
                S_CN_SX:     if (rx_pending) begin rms_shift_x <= $signed(rx_byte); state <= S_CN_SW_RMS; end
                S_CN_SW_RMS: if (rx_pending) begin rms_shift_w <= $signed(rx_byte); state <= S_CN_SW_MM;  end
                S_CN_SW_MM:  if (rx_pending) begin cn_sw_mm    <= $signed(rx_byte); state <= S_CN_RX_X;   end
                S_CN_RX_X: if (rx_pending) begin
                    xbuf_waddr <= rx_idx; xbuf_wdata <= $signed(rx_byte); xbuf_we <= 1'b1;
                    if (rx_idx == 10'd63) begin rx_idx <= 10'd0; state <= S_CN_N; end
                    else rx_idx <= rx_idx + 10'd1;
                end
                S_CN_N: if (rx_pending) begin fm_N <= rx_byte[6:0]; state <= S_CN_RA0; end
                // addr_rms (pour fetch rmsnorm w)
                S_CN_RA0: if (rx_pending) begin sd_addr[7:0]    <= rx_byte;       state <= S_CN_RA1; end
                S_CN_RA1: if (rx_pending) begin sd_addr[15:8]   <= rx_byte;       state <= S_CN_RA2; end
                S_CN_RA2: if (rx_pending) begin sd_addr[22:16]  <= rx_byte[6:0];  state <= S_CN_MA0; end
                // addr_mm (saved for matmul phase)
                S_CN_MA0: if (rx_pending) begin cn_addr_mm[7:0]   <= rx_byte;     state <= S_CN_MA1; end
                S_CN_MA1: if (rx_pending) begin cn_addr_mm[15:8]  <= rx_byte;     state <= S_CN_MA2; end
                S_CN_MA2: if (rx_pending) begin
                    cn_addr_mm[22:16] <= rx_byte[6:0];
                    // Demarre Phase 1 : fetch rmsnorm w (use FN flow)
                    fetch_idx <= 7'd0;
                    state     <= S_FN_RD;
                end

                // (S_FN_RD/WB se terminent en pulsant rms_start -> S_RUN_RMS)
                // S_RUN_RMS branche sur cn_active : si actif -> COPY_RD, sinon TX_M1

                // ---- CN Phase 2 : copie obuf -> xbuf (64 bytes) ----
                S_CN_COPY_RD: begin
                    obuf_raddr <= rx_idx;
                    state      <= S_CN_COPY_W1;
                end
                S_CN_COPY_W1: state <= S_CN_COPY_W2;
                S_CN_COPY_W2: state <= S_CN_COPY_WB;
                S_CN_COPY_WB: begin
                    xbuf_waddr <= rx_idx;
                    xbuf_wdata <= obuf_rdata_reg;
                    xbuf_we    <= 1'b1;
                    if (rx_idx == 10'd63) begin
                        rx_idx <= 10'd0;
                        state  <= S_CN_SETUP_MM;
                    end else begin
                        rx_idx <= rx_idx + 10'd1;
                        state  <= S_CN_COPY_RD;
                    end
                end

                // ---- CN Phase 3 : setup matmul then go en FQ flow ----
                S_CN_SETUP_MM: begin
                    sd_addr     <= cn_addr_mm;
                    rms_shift_x <= rms_shift_out;
                    rms_shift_w <= cn_sw_mm;
                    op_sel      <= 4'd7;
                    fq_mode     <= 1'b1;
                    fm_row      <= 7'd0;
                    fetch_idx   <= 7'd0;
                    state       <= S_FM_WARMUP_RD;
                end

                // ---- CN TX : 'C' 'N' shift_total y_int8[N] ----
                S_CN_TX_C: if (!tx_busy && !tx_send) begin
                    tx_data <= "C"; tx_send <= 1'b1; state <= S_CN_TX_N;
                end
                S_CN_TX_N: if (!tx_busy && !tx_send) begin
                    tx_data <= "N"; tx_send <= 1'b1; state <= S_CN_TX_SO;
                end
                S_CN_TX_SO: if (!tx_busy && !tx_send) begin
                    tx_data <= fq_shift_total; tx_send <= 1'b1;
                    tx_idx  <= 10'd0;
                    state   <= S_TX_O_RD;       // reuse pour envoyer obuf[0..N-1]
                end

                // ---- Pattern v3 : sync sd_rd/sd_wr/sd_refresh ----
                S_OP_RD_BUSY: if (sd_busy)  state <= S_OP_RD_DONE;
                S_OP_RD_DONE: if (!sd_busy) state <= next_state;
                S_OP_WR_BUSY: if (sd_busy)  state <= S_OP_WR_DONE;
                S_OP_WR_DONE: if (!sd_busy) state <= next_state;
                S_REF_BUSY:   if (sd_busy)  state <= S_REF_DONE;
                S_REF_DONE:   if (!sd_busy) state <= ret_state;

                default: state <= S_IDLE;
            endcase
        end
    end

    reg [23:0] hb = 24'd0;
    always @(posedge clk_sys) hb <= hb + 24'd1;
    assign led = ~{ state[3:0], op_silu, hb[23] };

endmodule
