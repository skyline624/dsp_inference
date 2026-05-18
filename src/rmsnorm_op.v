// =============================================================================
// rmsnorm_op.v  --  RMSNorm en int8 + power-of-2 shift, version 2 (debugge)
//
// out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)
//
// Maths (D=64, LOG2D=6) :
//   acc = sum_{i} x_int[i]^2                       (24-bit unsigned)
//   p = position bit de tete de acc (0..23)
//   shift_amt = 8 - p (signed, range -15..8)
//   acc_norm = acc << shift_amt  (ou >> -shift_amt si negatif)
//   index = acc_norm[7:0]                          (LUT a 256 entrees)
//   raw_inv = LUT[index]                           (Q1.15)
//   if shift_amt[0]: raw_inv = (raw_inv * SQRT2_Q15) >> 15   (parite)
//   apply_shift = (shift_amt >>> 1) - 16
//   out_int[i] = clip( (x_int[i] * w_int[i] * raw_inv) >> (-apply_shift) )
//   shift_out = shift_w
//
// FSM en 4 phases pipelinees :
//   SUM : adresse + accumule (2 cycles par element)
//   COMPUTE : NORM -> LUT lookup -> PARITY -> APPLY_SHIFT
//   APPLY : adresse x et w + multiplie + clip + ecrit (3 cycles par element)
//
// Debug outputs (a echantillonner via top.v) :
//   dbg_acc, dbg_p, dbg_shift_amt, dbg_raw_inv, dbg_apply_shift
// =============================================================================

module rmsnorm_op #(
    parameter D     = 64,
    parameter LOG2D = 6
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,

    input  wire signed [7:0] shift_x,
    input  wire signed [7:0] shift_w,
    output reg  signed [7:0] shift_out,

    output reg  [9:0]        x_raddr,
    input  wire signed [7:0] x_rdata,
    output reg  [9:0]        w_raddr,
    input  wire signed [7:0] w_rdata,
    output reg  [9:0]        out_waddr,
    output reg  signed [7:0] out_wdata,
    output reg               out_we,

    // Debug
    output reg  [23:0]       dbg_acc,
    output reg  [4:0]        dbg_p,
    output reg  signed [5:0] dbg_shift_amt,
    output reg  [15:0]       dbg_raw_inv,
    output reg  signed [7:0] dbg_apply_shift
);

    // LUT 1/sqrt(x) pour x dans [1, 2), Q1.15
    reg [15:0] rsqrt_lut [0:255];
    initial $readmemh("rsqrt_lut.hex", rsqrt_lut);

    // sqrt(2) en Q1.15 = 1.4142 * 32768 = 46341 = 0xB505
    localparam [15:0] SQRT2_Q15 = 16'hB505;

    localparam S_IDLE        = 4'd0,
               S_SUM_ADDR    = 4'd1,
               S_SUM_WAIT1   = 4'd2,   // 2 cycles latence BSRAM (Gowin pipeline)
               S_SUM_WAIT2   = 4'd3,
               S_SUM_DATA    = 4'd4,
               S_NORM        = 4'd5,
               S_LUT         = 4'd6,
               S_PARITY      = 4'd7,
               S_APPLY_ADDR  = 4'd8,
               S_APPLY_WAIT1 = 4'd9,
               S_APPLY_WAIT2 = 4'd10,
               S_APPLY_MULT  = 4'd11,
               S_APPLY_WB    = 4'd12,
               S_DONE        = 4'd13;

    reg [3:0]   state;
    reg [9:0]   idx;
    reg [23:0]  acc;
    reg [4:0]   p_reg;
    reg signed [5:0] shift_amt;
    reg [15:0]  raw_inv;
    reg signed [7:0] apply_shift;

    // Apply pipeline
    reg signed [7:0]  xv_reg, wv_reg;
    reg signed [31:0] prod_reg;

    // Combinational: leading bit position of acc
    reg [4:0] p_comb;
    integer ii;
    always @(*) begin
        p_comb = 5'd0;
        for (ii = 0; ii < 24; ii = ii + 1)
            if (acc[ii]) p_comb = ii[4:0];
    end

    // Etendus signed 16-bit pour forcer la multiplication signee
    // (Gowin synth ne propage pas 'signed' a travers les ports).
    reg signed [15:0] xs16, ws16;
    reg signed [31:0] x_sq;     // resultat mult explicite signed
    always @(*) begin
        xs16 = {{8{x_rdata[7]}}, x_rdata};
        ws16 = {{8{w_rdata[7]}}, w_rdata};
        x_sq = xs16 * xs16;     // signed * signed -> signed 32
    end

    // Combinational: acc_normalized for LUT lookup
    // shift_amt = 8 - p_reg.  shift_amt > 0 -> shift left ; < 0 -> shift right
    reg [31:0] acc_norm_w;
    always @(*) begin
        if (shift_amt >= 0)
            acc_norm_w = {8'd0, acc} << shift_amt;
        else
            acc_norm_w = {8'd0, acc} >> (-shift_amt);
    end
    wire [7:0] lut_index = acc_norm_w[7:0];

    // Combinational: raw_inv apres correction parite
    wire [31:0] raw_inv_sqrt2 = {16'd0, raw_inv} * SQRT2_Q15;   // 32-bit
    wire [15:0] raw_inv_fixed = shift_amt[0] ? raw_inv_sqrt2[30:15]    // *sqrt(2)>>15
                                              : raw_inv;

    // Combinational: prod >> shift avec arrondi vers zero+demi, clipping int8
    reg signed [31:0] shifted;
    reg signed [31:0] rounding;
    always @(*) begin
        if (apply_shift >= 0) begin
            rounding = 32'sd0;
            shifted  = prod_reg <<< apply_shift;
        end else begin
            // rounding = 1 << (-apply_shift - 1), pour arrondir au plus proche
            rounding = (apply_shift == -8'sd1)
                       ? 32'sd0    // pas de rounding si shift = -1 (decalage de 1)
                       : (32'sd1 <<< (-apply_shift - 1));
            shifted  = (prod_reg + rounding) >>> (-apply_shift);
        end
    end
    wire signed [7:0] clipped =
        (shifted > 32'sd127)   ? 8'sd127 :
        (shifted < -32'sd128)  ? -8'sd128 :
                                 shifted[7:0];

    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            out_we          <= 1'b0;
            dbg_acc         <= 24'd0;
            dbg_p           <= 5'd0;
            dbg_shift_amt   <= 6'd0;
            dbg_raw_inv     <= 16'd0;
            dbg_apply_shift <= 8'd0;
        end else begin
            out_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: if (start) begin
                    acc   <= 24'd0;
                    idx   <= 10'd0;
                    state <= S_SUM_ADDR;
                end

                S_SUM_ADDR: begin
                    x_raddr <= idx;
                    state   <= S_SUM_WAIT1;
                end
                S_SUM_WAIT1: state <= S_SUM_WAIT2;
                S_SUM_WAIT2: state <= S_SUM_DATA;   // latence BSRAM 2 cycles

                S_SUM_DATA: begin
                    // x_sq pre-calcule par always @(*) (signed 32) -> bits[23:0]
                    acc <= acc + x_sq[23:0];
                    if (idx == D - 1) state <= S_NORM;
                    else begin
                        idx   <= idx + 10'd1;
                        state <= S_SUM_ADDR;
                    end
                end

                S_NORM: begin
                    p_reg         <= p_comb;
                    shift_amt     <= $signed({1'b0, 5'd8}) - $signed({1'b0, p_comb});
                    dbg_acc       <= acc;
                    dbg_p         <= p_comb;
                    state         <= S_LUT;
                end

                S_LUT: begin
                    dbg_shift_amt <= shift_amt;
                    raw_inv       <= rsqrt_lut[lut_index];
                    state         <= S_PARITY;
                end

                S_PARITY: begin
                    // Correction parite : si shift_amt impair, *= sqrt(2)
                    raw_inv         <= raw_inv_fixed;
                    // apply_shift = (shift_amt >>> 1) - 16, signe-correct sur 8 bits
                    apply_shift     <= $signed({{3{shift_amt[5]}}, shift_amt[5:1]}) - 8'sd16;
                    shift_out       <= shift_w;
                    dbg_raw_inv     <= raw_inv_fixed;
                    dbg_apply_shift <= $signed({{3{shift_amt[5]}}, shift_amt[5:1]}) - 8'sd16;
                    idx             <= 10'd0;
                    state           <= S_APPLY_ADDR;
                end

                S_APPLY_ADDR: begin
                    x_raddr <= idx;
                    w_raddr <= idx;
                    state   <= S_APPLY_WAIT1;
                end
                S_APPLY_WAIT1: state <= S_APPLY_WAIT2;
                S_APPLY_WAIT2: state <= S_APPLY_MULT;   // latence BSRAM 2 cycles

                S_APPLY_MULT: begin
                    xv_reg   <= x_rdata;
                    wv_reg   <= w_rdata;
                    // xs16 * ws16 = signed 32 ; * raw_inv (signed 17) = signed 49
                    prod_reg <= (xs16 * ws16) * $signed({1'b0, raw_inv});
                    state    <= S_APPLY_WB;
                end

                S_APPLY_WB: begin
                    out_waddr <= idx;
                    out_wdata <= clipped;
                    out_we    <= 1'b1;
                    if (idx == D - 1) state <= S_DONE;
                    else begin
                        idx   <= idx + 10'd1;
                        state <= S_APPLY_ADDR;
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
