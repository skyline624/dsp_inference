// =============================================================================
// softmax_op.v  --  Softmax en int8 + shift  (K fixe = 32)
//
//   softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
//
// 4 phases :
//   PH_MAX  : scan x[i], trouve max (signed 8-bit)
//   PH_EXP  : pour chaque i :
//               idx = (x[i] - max) * 2^(shift_x + 5) + 256, clamp [0, 255]
//               exp_val[i] = exp_lut[idx]   (Q15 -> int16)
//               sum += exp_val[i]
//   PH_INV  : compute 1/sum via LUT inv_lut + normalisation
//   PH_NORM : pour chaque i : out[i] = exp_val[i] * inv_sum, requantifie en int8
//
// Sortie : shift_out = -7 (proba dans [0, 1], donc 127 * 2^-7 ~= 0.99 OK)
//
// exp_val[K_MAX=32] stocke en interne (32 x 16-bit = 64 oct, trivial).
//
// Cf. [[gowin-signed-mult-gotcha]] pour les regs signed locaux.
// =============================================================================

module softmax_op #(
    parameter K_MAX = 32
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,

    input  wire signed [7:0] shift_x,
    output reg  signed [7:0] shift_out,

    // x buffer (read)
    output reg  [9:0]        x_raddr,
    input  wire signed [7:0] x_rdata,
    // out buffer (write)
    output reg  [9:0]        out_waddr,
    output reg  signed [7:0] out_wdata,
    output reg               out_we,

    // Debug : echantillonne sur dernier element
    output reg  signed [7:0] dbg_max,
    output reg  [23:0]       dbg_sum,
    output reg  [7:0]        dbg_p_sum,
    output reg  [15:0]       dbg_inv_sum
);

    // LUTs
    reg [15:0] exp_lut [0:255];
    reg [15:0] inv_lut [0:255];
    initial begin
        $readmemh("exp_lut.hex", exp_lut);
        $readmemh("inv_lut.hex", inv_lut);
    end

    // Storage exp_val (K_MAX entrees x 16-bit)
    reg [15:0] exp_val [0:K_MAX-1];

    localparam S_IDLE       = 5'd0,
               S_MAX_ADDR   = 5'd1,
               S_MAX_W1     = 5'd2,
               S_MAX_W2     = 5'd3,
               S_MAX_DATA   = 5'd4,
               S_EXP_ADDR   = 5'd5,
               S_EXP_W1     = 5'd6,
               S_EXP_W2     = 5'd7,
               S_EXP_LOOKUP = 5'd8,
               S_EXP_STORE  = 5'd9,
               S_INV_NORM   = 5'd10,
               S_INV_LUT    = 5'd11,
               S_NORM       = 5'd12,
               S_NORM_WB    = 5'd13,
               S_DONE       = 5'd14;

    reg [4:0]  state;
    reg [9:0]  idx;
    reg signed [7:0] max_val;

    // EXP phase
    reg signed [15:0] diff16;     // (x - max) sign-extended
    reg signed [31:0] diff_scaled;
    reg signed [31:0] exp_idx_s;
    wire signed [7:0] shx_p5 = shift_x + 8'sd5;
    always @(*) begin
        diff16      = {{8{x_rdata[7]}}, x_rdata} - {{8{max_val[7]}}, max_val};
        if (shx_p5 >= 0)
            diff_scaled = {{16{diff16[15]}}, diff16} <<< shx_p5;
        else
            diff_scaled = {{16{diff16[15]}}, diff16} >>> (-shx_p5);
        exp_idx_s   = diff_scaled + 32'sd256;
    end
    wire [7:0] exp_idx =
        (exp_idx_s[31])          ? 8'd0   :
        (exp_idx_s > 32'sd255)   ? 8'd255 :
                                   exp_idx_s[7:0];

    // SUM accumulator (24-bit unsigned, max 32 * 32768 = 1048576 = 2^20 fits)
    reg [23:0] sum_acc;
    reg [15:0] exp_curr;

    // Find leading bit of sum
    reg [4:0] p_sum;
    integer ii;
    always @(*) begin
        p_sum = 5'd0;
        for (ii = 0; ii < 24; ii = ii + 1)
            if (sum_acc[ii]) p_sum = ii[4:0];
    end

    // Normalize sum to [256, 512) for LUT
    reg signed [5:0] shift_inv;       // = 8 - p_sum
    reg [31:0]       sum_norm;
    always @(*) begin
        shift_inv = $signed({1'b0, 5'd8}) - $signed({1'b0, p_sum});
        if (shift_inv >= 0)
            sum_norm = {8'd0, sum_acc} << shift_inv;
        else
            sum_norm = {8'd0, sum_acc} >> (-shift_inv);
    end
    wire [7:0] inv_lut_idx = sum_norm[7:0];

    reg [15:0]       inv_sum;        // Q15
    reg signed [7:0] norm_shift;     // = -7 - (-shift_inv + 15) = shift_inv - 22

    // Normalize : prod = exp_val[i] * inv_sum (Q15 * Q15 = Q30)
    // Real result = (exp_real) * (inv_sum_real) = prob in [0, 1]
    // out_int8 = prob / 2^shift_out = prob / 2^-7 = prob * 128
    // exp_real = exp_val * 2^-15, inv_sum_real = inv_sum * 2^(shift_inv - 15 - extra_log_for_sum_value)
    //   Note: inv_sum_real = 1/sum_real. sum_real = sum_int * 2^? Hmm sum_int IS the sum of int values.
    //   Actually sum_real = sum_int (no shift because exp_val IS the int representation).
    //   So inv_sum_real = 1 / sum_int = lookup / 2^15 / sum_int / 256 = ...
    // Simpler: prob = exp_val[i] / sum_int (both are int representations)
    //   out_int8 = round(prob * 128) = round(exp_val[i] * 128 / sum_int)
    //           = round(exp_val[i] * inv_sum * 2^(shift_inv) * 128 / 2^15)
    //   where inv_sum is the LUT result (= 2^15 / (sum_int * 2^(-shift_inv) / 256) = 2^15 * 256 / (sum_int * 2^(-shift_inv)))
    //   Hmm let me redo cleanly :
    //
    //   sum_norm = sum_int * 2^shift_inv (in [256, 512))
    //   inv_sum_lut = 1 / (sum_norm / 256) * 2^15 = 256 * 2^15 / sum_norm  (LUT value)
    //   1/sum_int = inv_sum_lut * 2^(-15 - 8 + shift_inv) = inv_sum_lut * 2^(shift_inv - 23)
    //   out_int = round(exp_val[i] / sum_int * 128)
    //           = round(exp_val[i] * inv_sum_lut * 2^(shift_inv - 23) * 128)
    //           = round(exp_val[i] * inv_sum_lut * 2^(shift_inv - 16))
    //
    //   En pratique : prod = exp_val[i] * inv_sum_lut (Q30 ou environ), shifte par (16 - shift_inv).
    //   norm_shift = 16 - shift_inv
    reg signed [31:0] prob_prod;
    reg signed [31:0] prob_shifted;
    reg signed [7:0]  prob_clip;
    reg signed [7:0]  norm_shift_eff;
    always @(*) begin
        norm_shift_eff = 8'sd16 - $signed({{2{shift_inv[5]}}, shift_inv});
        prob_prod = $signed({1'b0, exp_curr}) * $signed({1'b0, inv_sum});
        if (norm_shift_eff > 0)
            prob_shifted = (prob_prod + (32'sd1 <<< (norm_shift_eff - 1))) >>> norm_shift_eff;
        else if (norm_shift_eff < 0)
            prob_shifted = prob_prod <<< (-norm_shift_eff);
        else
            prob_shifted = prob_prod;
        prob_clip = (prob_shifted > 32'sd127)  ? 8'sd127  :
                    (prob_shifted < -32'sd128) ? -8'sd128 :
                                                 prob_shifted[7:0];
    end

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
                    idx       <= 10'd0;
                    max_val   <= -8'sd128;
                    sum_acc   <= 24'd0;
                    shift_out <= -8'sd7;       // proba en Q-7
                    state     <= S_MAX_ADDR;
                end

                // --- PHASE 1: find max ---
                S_MAX_ADDR: begin
                    x_raddr <= idx;
                    state   <= S_MAX_W1;
                end
                S_MAX_W1: state <= S_MAX_W2;
                S_MAX_W2: state <= S_MAX_DATA;
                S_MAX_DATA: begin
                    if ($signed(x_rdata) > $signed(max_val))
                        max_val <= x_rdata;
                    if (idx == K_MAX - 1) begin
                        idx   <= 10'd0;
                        state <= S_EXP_ADDR;
                    end else begin
                        idx   <= idx + 10'd1;
                        state <= S_MAX_ADDR;
                    end
                end

                // --- PHASE 2: exp + sum ---
                S_EXP_ADDR: begin
                    x_raddr <= idx;
                    state   <= S_EXP_W1;
                end
                S_EXP_W1: state <= S_EXP_W2;
                S_EXP_W2: state <= S_EXP_LOOKUP;
                S_EXP_LOOKUP: begin
                    // exp_idx comb depuis x_rdata, max_val
                    exp_curr <= exp_lut[exp_idx];
                    state    <= S_EXP_STORE;
                end
                S_EXP_STORE: begin
                    exp_val[idx[4:0]] <= exp_curr;
                    sum_acc           <= sum_acc + {8'd0, exp_curr};
                    if (idx == K_MAX - 1) begin
                        idx   <= 10'd0;
                        state <= S_INV_NORM;
                    end else begin
                        idx   <= idx + 10'd1;
                        state <= S_EXP_ADDR;
                    end
                end

                // --- PHASE 3: 1/sum via LUT ---
                S_INV_NORM: begin
                    dbg_sum   <= sum_acc;
                    dbg_p_sum <= {3'd0, p_sum};
                    state     <= S_INV_LUT;
                end
                S_INV_LUT: begin
                    inv_sum     <= inv_lut[inv_lut_idx];
                    dbg_inv_sum <= inv_lut[inv_lut_idx];
                    dbg_max     <= max_val;
                    idx         <= 10'd0;
                    state       <= S_NORM;
                end

                // --- PHASE 4: normalize and write ---
                S_NORM: begin
                    exp_curr <= exp_val[idx[4:0]];
                    state    <= S_NORM_WB;
                end
                S_NORM_WB: begin
                    out_waddr <= idx;
                    out_wdata <= prob_clip;
                    out_we    <= 1'b1;
                    if (idx == K_MAX - 1) state <= S_DONE;
                    else begin
                        idx   <= idx + 10'd1;
                        state <= S_NORM;
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
