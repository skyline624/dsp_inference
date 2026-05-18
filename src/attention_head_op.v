// =============================================================================
// attention_head_op.v  --  Attention pour UNE seule tete (single-head)
//
// V2 2026-05-18 : T_MAX bumpe a 32 (etait 8), shift_out corrige.
//
// Calcule : out[d] = sum_t attn[t] * V[t, d]
//   ou attn[t] = softmax_t( Q . K[t] )
//
// Params : HS = 8 (head_size), T_MAX = 32 (sequence longueur max)
//
// Inputs :
//   Q[HS]      : query vecteur, fourni via ports parall (regs externes)
//   K[T*HS]    : clefs en BSRAM kbuf, organise [t][d]
//   V[T*HS]    : valeurs en BSRAM vbuf, meme organisation
//   T          : longueur courante (1..T_MAX)
//   shift_q,k,v : scales d'entree
//
// Output :
//   out[HS]    : ecrit dans obuf[0..HS-1]
//   shift_out  : empiriquement = shift_v (la sortie attn_int est l'amplitude V_int)
//
// Refactor V2 :
//   - score/ev/att stockes dans des regs packed (16*T_MAX bits chacun)
//     evite l'inference BSRAM (lecture 1-cycle staled) tout en supportant T_MAX=32
//   - T widen a 6 bits, t_idx/d_idx widen
//   - shift_out fixe : shift_v (au lieu de shift_v - 7 qui etait empiriquement faux)
// =============================================================================

module attention_head_op #(
    parameter HS    = 8,
    parameter T_MAX = 32
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,

    input  wire [5:0]        T,                  // longueur seq courante (1..T_MAX)
    input  wire signed [7:0] shift_q, shift_k, shift_v,
    output reg  signed [7:0] shift_out,

    // GQA addressing : K/V buffer layout = [t][kv_head][hs]
    input  wire [5:0]        kv_stride,
    input  wire [5:0]        kv_offset,

    // Q en wires plats (8 elements * 8 bits = 64 bits)
    input  wire signed [HS*8-1:0] Q_flat,

    // K BSRAM
    output reg  [9:0]        k_raddr,
    input  wire signed [7:0] k_rdata,
    // V BSRAM
    output reg  [9:0]        v_raddr,
    input  wire signed [7:0] v_rdata,
    // Out BSRAM
    output reg  [9:0]        out_waddr,
    output reg  signed [7:0] out_wdata,
    output reg               out_we,

    // Debug
    output reg  signed [15:0] dbg_score_last,
    output reg  signed [7:0]  dbg_max_score,
    output reg  [15:0]        dbg_exp_sum,
    output reg  [15:0]        dbg_inv_sum
);

    // LUTs (memes que softmax_op)
    reg [15:0] exp_lut [0:255];
    reg [15:0] inv_lut [0:255];
    initial begin
        $readmemh("exp_lut.hex", exp_lut);
        $readmemh("inv_lut.hex", inv_lut);
    end

    // Storage interne PACKED : 1 reg plat de T_MAX*16 bits, part-select pour
    // indexer. Force Gowin a NE PAS inferer BSRAM (lecture 1-cycle staled).
    // Distributed LUT-RAM via part-select = lecture combinationnelle.
    reg signed [16*T_MAX-1:0] scores_packed;
    reg        [16*T_MAX-1:0] ev_packed;
    reg        [16*T_MAX-1:0] att_packed;

    // Q en regs (pour acces indexe)
    reg signed [7:0] Q_reg [0:HS-1];
    integer qi;
    always @(*) begin
        for (qi = 0; qi < HS; qi = qi + 1)
            Q_reg[qi] = Q_flat[qi*8 +: 8];
    end

    localparam S_IDLE         = 5'd0,
               S_SC_T_INIT    = 5'd1,
               S_SC_I_INIT    = 5'd2,
               S_SC_ADDR      = 5'd3,
               S_SC_W1        = 5'd4,
               S_SC_W2        = 5'd5,
               S_SC_MULT      = 5'd6,
               S_SC_NEXT_I    = 5'd7,
               S_SC_STORE     = 5'd8,
               S_MAX_INIT     = 5'd9,
               S_MAX_SCAN     = 5'd10,
               S_EXP_INIT     = 5'd11,
               S_EXP_LOOP     = 5'd12,
               S_INV_NORM     = 5'd13,
               S_INV_LUT      = 5'd14,
               S_NORM_INIT    = 5'd15,
               S_NORM_LOOP    = 5'd16,
               S_OUT_D_INIT   = 5'd17,
               S_OUT_T_INIT   = 5'd18,
               S_OUT_ADDR     = 5'd19,
               S_OUT_W1       = 5'd20,
               S_OUT_W2       = 5'd21,
               S_OUT_MULT     = 5'd22,
               S_OUT_NEXT_T   = 5'd23,
               S_OUT_STORE    = 5'd24,
               S_DONE         = 5'd25;

    reg [4:0]  state;
    reg [5:0]  t_idx;     // 0..T-1 (6 bits pour T_MAX=32)
    reg [3:0]  i_idx;     // 0..HS-1
    reg [3:0]  d_idx;     // 0..HS-1

    reg signed [31:0] dot_acc;
    reg signed [7:0]  max_score;

    // Acces indexes combinationnels via part-select des packed regs
    wire signed [15:0] score_at_idx  = $signed(scores_packed[t_idx*16 +: 16]);
    wire signed [15:0] score_at_next = $signed(scores_packed[(t_idx+6'd1)*16 +: 16]);
    wire        [15:0] ev_at_idx     = ev_packed   [t_idx*16 +: 16];
    wire        [15:0] att_at_idx    = att_packed  [t_idx*16 +: 16];

    reg signed [15:0] diff16;
    reg signed [31:0] diff_scaled;
    reg signed [31:0] exp_idx_s;
    reg signed [15:0] max_ext_lshift;
    reg signed [31:0] diff32_signed;
    reg signed [15:0] score_curr;
    reg [15:0]        exp_curr;
    reg [23:0]        sum_acc;

    always @(*) begin
        max_ext_lshift = {max_score, 8'd0};
        diff16         = score_curr - max_ext_lshift;
        diff32_signed  = $signed({{16{diff16[15]}}, diff16});
        diff_scaled    = diff32_signed >>> 10;
        exp_idx_s      = diff_scaled + 32'sd256;
    end
    wire [7:0] exp_idx =
        (exp_idx_s[31])          ? 8'd0   :
        (exp_idx_s > 32'sd255)   ? 8'd255 :
                                   exp_idx_s[7:0];

    reg [4:0] p_sum;
    integer ii;
    always @(*) begin
        p_sum = 5'd0;
        for (ii = 0; ii < 24; ii = ii + 1)
            if (sum_acc[ii]) p_sum = ii[4:0];
    end
    reg signed [5:0] shift_inv;
    reg [31:0]       sum_norm;
    always @(*) begin
        shift_inv = $signed({1'b0, 5'd8}) - $signed({1'b0, p_sum});
        if (shift_inv >= 0)
            sum_norm = {8'd0, sum_acc} << shift_inv;
        else
            sum_norm = {8'd0, sum_acc} >> (-shift_inv);
    end
    wire [7:0] inv_lut_idx = sum_norm[7:0];
    reg [15:0] inv_sum;

    reg signed [7:0] norm_shift_attn;
    reg signed [31:0] attn_prod;
    reg [15:0]        attn_curr_in;
    reg [15:0]        attn_norm;
    always @(*) begin
        norm_shift_attn = 8'sd15 - $signed({{2{shift_inv[5]}}, shift_inv});
        attn_prod = $signed({1'b0, attn_curr_in}) * $signed({1'b0, inv_sum});
        if (norm_shift_attn > 0)
            attn_norm = (attn_prod + (32'sd1 <<< (norm_shift_attn - 1))) >>> norm_shift_attn;
        else if (norm_shift_attn < 0)
            attn_norm = attn_prod <<< (-norm_shift_attn);
        else
            attn_norm = attn_prod;
    end

    reg signed [15:0] qi16, ki16;
    reg signed [31:0] qk_prod;
    always @(*) begin
        qi16 = {{8{Q_reg[i_idx[2:0]][7]}}, Q_reg[i_idx[2:0]]};
        ki16 = {{8{k_rdata[7]}}, k_rdata};
        qk_prod = qi16 * ki16;
    end

    reg signed [31:0] attn_v_prod;
    reg signed [15:0] attn_curr_out;
    reg signed [15:0] v_curr_ext;
    always @(*) begin
        attn_curr_out = $signed({1'b0, att_at_idx});
        v_curr_ext    = {{8{v_rdata[7]}}, v_rdata};
        attn_v_prod   = attn_curr_out * v_curr_ext;
    end

    // OUT STORE : shift et clip
    // Empiriquement : int8 = (out_acc + 128) >> 8 represente le bon out_real
    // au shift = shift_v. La formule "shift_out = shift_v - 7" du V1 etait fausse.
    reg signed [31:0] out_acc;
    reg signed [31:0] out_shifted;
    reg signed [7:0]  out_clip;
    always @(*) begin
        out_shifted = (out_acc + (32'sd1 <<< 7)) >>> 8;
        out_clip    = (out_shifted > 32'sd127)  ? 8'sd127  :
                      (out_shifted < -32'sd128) ? -8'sd128 :
                                                  out_shifted[7:0];
    end

    integer pi;
    always @(posedge clk) begin
        if (rst) begin
            state  <= S_IDLE;
            done   <= 1'b0;
            out_we <= 1'b0;
        end else begin
            out_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: if (start) begin
                    t_idx     <= 6'd0;
                    sum_acc   <= 24'd0;
                    max_score <= -8'sd128;
                    shift_out <= shift_v;       // fix : etait shift_v - 7 (faux)
                    state     <= S_SC_T_INIT;
                end

                // ---- PHASE SCORE ----
                S_SC_T_INIT: begin
                    dot_acc <= 32'sd0;
                    i_idx   <= 4'd0;
                    state   <= S_SC_ADDR;
                end

                S_SC_ADDR: begin
                    k_raddr <= (t_idx * kv_stride) + {4'd0, kv_offset} + {6'd0, i_idx[3:0]};
                    state   <= S_SC_W1;
                end
                S_SC_W1: state <= S_SC_W2;
                S_SC_W2: state <= S_SC_MULT;
                S_SC_MULT: begin
                    dot_acc <= dot_acc + qk_prod;
                    state   <= S_SC_NEXT_I;
                end
                S_SC_NEXT_I: begin
                    if (i_idx == HS - 1) state <= S_SC_STORE;
                    else begin
                        i_idx <= i_idx + 4'd1;
                        state <= S_SC_ADDR;
                    end
                end

                S_SC_STORE: begin
                    begin: store_score
                        reg signed [15:0] sval;
                        if      (dot_acc > 32'sd32767)  sval = 16'sd32767;
                        else if (dot_acc < -32'sd32768) sval = -16'sd32768;
                        else                            sval = dot_acc[15:0];
                        scores_packed[t_idx*16 +: 16] <= sval;
                    end
                    if (t_idx == T - 6'd1) begin
                        t_idx <= 6'd0;
                        state <= S_MAX_INIT;
                    end else begin
                        t_idx <= t_idx + 6'd1;
                        state <= S_SC_T_INIT;
                    end
                end

                // ---- PHASE MAX ----
                S_MAX_INIT: begin
                    max_score <= -8'sd128;
                    t_idx     <= 6'd0;
                    state     <= S_MAX_SCAN;
                end
                S_MAX_SCAN: begin
                    if ($signed(score_at_idx[15:8]) > $signed(max_score))
                        max_score <= score_at_idx[15:8];
                    if (t_idx == T - 6'd1) begin
                        t_idx   <= 6'd0;
                        sum_acc <= 24'd0;
                        state   <= S_EXP_INIT;
                    end else begin
                        t_idx <= t_idx + 6'd1;
                    end
                end

                // ---- PHASE EXP+SUM ----
                S_EXP_INIT: begin
                    score_curr <= score_at_idx;
                    state      <= S_EXP_LOOP;
                end
                S_EXP_LOOP: begin
                    ev_packed[t_idx*16 +: 16] <= exp_lut[exp_idx];
                    sum_acc <= sum_acc + {8'd0, exp_lut[exp_idx]};
                    if (t_idx == T - 6'd1) begin
                        t_idx <= 6'd0;
                        state <= S_INV_NORM;
                    end else begin
                        t_idx      <= t_idx + 6'd1;
                        score_curr <= score_at_next;
                        state      <= S_EXP_LOOP;
                    end
                end

                // ---- PHASE 1/sum ----
                S_INV_NORM: state <= S_INV_LUT;
                S_INV_LUT: begin
                    inv_sum       <= inv_lut[inv_lut_idx];
                    dbg_inv_sum   <= inv_lut[inv_lut_idx];
                    dbg_exp_sum   <= sum_acc[15:0];
                    dbg_max_score <= max_score;
                    t_idx         <= 6'd0;
                    state         <= S_NORM_INIT;
                end

                // ---- PHASE NORMALIZE attn ----
                S_NORM_INIT: begin
                    attn_curr_in <= ev_at_idx;
                    state        <= S_NORM_LOOP;
                end
                S_NORM_LOOP: begin
                    att_packed[t_idx*16 +: 16] <= attn_norm;
                    if (t_idx == T - 6'd1) begin
                        d_idx <= 4'd0;
                        state <= S_OUT_D_INIT;
                    end else begin
                        // pre-load next ev
                        attn_curr_in <= ev_packed[(t_idx+6'd1)*16 +: 16];
                        t_idx <= t_idx + 6'd1;
                        state <= S_NORM_LOOP;
                    end
                end

                // ---- PHASE OUT (sum_t attn[t] * V[t, d]) ----
                S_OUT_D_INIT: begin
                    t_idx   <= 6'd0;
                    out_acc <= 32'sd0;
                    state   <= S_OUT_T_INIT;
                end
                S_OUT_T_INIT: state <= S_OUT_ADDR;
                S_OUT_ADDR: begin
                    v_raddr <= (t_idx * kv_stride) + {4'd0, kv_offset} + {6'd0, d_idx[3:0]};
                    state   <= S_OUT_W1;
                end
                S_OUT_W1: state <= S_OUT_W2;
                S_OUT_W2: state <= S_OUT_MULT;
                S_OUT_MULT: begin
                    out_acc <= out_acc + attn_v_prod;
                    state   <= S_OUT_NEXT_T;
                end
                S_OUT_NEXT_T: begin
                    if (t_idx == T - 6'd1) state <= S_OUT_STORE;
                    else begin
                        t_idx <= t_idx + 6'd1;
                        state <= S_OUT_ADDR;
                    end
                end
                S_OUT_STORE: begin
                    out_waddr <= {7'd0, d_idx[2:0]};
                    out_wdata <= out_clip;
                    out_we    <= 1'b1;
                    if (d_idx == HS - 1) begin
                        dbg_score_last <= scores_packed[(T - 6'd1)*16 +: 16];
                        state <= S_DONE;
                    end else begin
                        d_idx <= d_idx + 4'd1;
                        state <= S_OUT_D_INIT;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
