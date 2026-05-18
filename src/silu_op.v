// =============================================================================
// silu_op.v  --  Operation SiLU (x * sigmoid(x)) en int8 + shift
//
//   out[i] = silu(x[i] * 2^shift_x)  retourne en int8 + shift_out (= shift_x)
//
// Maths :
//   x_real = x_int * 2^shift_x
//   silu(x) = x / (1 + exp(-x))   (calcule via LUT 256 entrees)
//   LUT[i] = round(silu((i-128)/16) * 2048)   en Q4.11 (scale 2^-11)
//   Index   = clamp((x_int * 2^(shift_x + 4)) + 128, 0, 255)
//   silu_int (Q4.11) -> int8 en shiftant a droite de (11 + shift_x).
//
// FSM par element : ADDR -> WAIT1 -> WAIT2 -> LOOKUP -> WB
//                   (5 cycles/element, similaire a rmsnorm_op).
// Pas de multiplication critique : juste LUT + shifts. Donc pas de
// gotcha Gowin signed-mult (cf. [[gowin-signed-mult-gotcha]]).
// =============================================================================

module silu_op #(
    parameter D = 64
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,

    input  wire signed [7:0] shift_x,
    output reg  signed [7:0] shift_out,

    output reg  [9:0]        x_raddr,
    input  wire signed [7:0] x_rdata,
    output reg  [9:0]        out_waddr,
    output reg  signed [7:0] out_wdata,
    output reg               out_we,

    // Debug : echantillonne sur le DERNIER element traite
    output reg  [7:0]        dbg_lut_idx,
    output reg  signed [15:0] dbg_silu_int
);

    // LUT silu, 256 entrees Q4.11
    reg signed [15:0] silu_lut [0:255];
    initial $readmemh("silu_lut.hex", silu_lut);

    localparam S_IDLE   = 3'd0,
               S_ADDR   = 3'd1,
               S_WAIT1  = 3'd2,
               S_WAIT2  = 3'd3,
               S_LOOKUP = 3'd4,
               S_WB     = 3'd5,
               S_DONE   = 3'd6;

    reg [2:0]  state;
    reg [9:0]  idx;
    reg [7:0]  lut_idx_reg;
    reg signed [15:0] silu_val;

    // ---- index combinational ----
    // x16 = x_rdata * 2^(shift_x + 4)
    // Si shift_x + 4 >= 0 : x16 = x_sext << (shift_x + 4)
    // Si shift_x + 4 < 0  : x16 = x_sext >> (-(shift_x + 4))  (arrondi vers -inf)
    reg signed [15:0]  xs16;
    reg signed [31:0]  x16;
    reg signed [16:0]  idx_signed;
    wire signed [7:0]  shx_p4 = shift_x + 8'sd4;

    always @(*) begin
        xs16 = {{8{x_rdata[7]}}, x_rdata};
        if (shx_p4 >= 0)
            x16 = {{16{xs16[15]}}, xs16} <<< shx_p4;
        else
            x16 = {{16{xs16[15]}}, xs16} >>> (-shx_p4);
        idx_signed = x16[16:0] + 17'sd128;
    end
    wire [7:0] lut_idx_w =
        (idx_signed[16])              ? 8'd0   :   // negatif -> clamp 0
        (idx_signed > 17'sd255)        ? 8'd255 :   // > 255   -> clamp 255
                                         idx_signed[7:0];

    // ---- shift de silu_int vers int8 ----
    // out_int = silu_int >> (11 + shift_x) avec arrondi + clip
    wire signed [7:0]  out_shift = 8'sd11 + shift_x;   // typiquement ~6-8
    reg signed [31:0]  silu_ext;
    reg signed [31:0]  silu_rounded;
    reg signed [31:0]  silu_shifted;
    always @(*) begin
        silu_ext = {{16{silu_val[15]}}, silu_val};
        if (out_shift > 0) begin
            silu_rounded = silu_ext + (32'sd1 <<< (out_shift - 1));
            silu_shifted = silu_rounded >>> out_shift;
        end else if (out_shift < 0) begin
            silu_rounded = silu_ext;
            silu_shifted = silu_ext <<< (-out_shift);
        end else begin
            silu_rounded = silu_ext;
            silu_shifted = silu_ext;
        end
    end
    wire signed [7:0] out_clip =
        (silu_shifted > 32'sd127)   ? 8'sd127  :
        (silu_shifted < -32'sd128)  ? -8'sd128 :
                                       silu_shifted[7:0];

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
                    shift_out <= shift_x;  // silu preserve la magnitude
                    state     <= S_ADDR;
                end

                S_ADDR: begin
                    x_raddr <= idx;
                    state   <= S_WAIT1;
                end
                S_WAIT1: state <= S_WAIT2;
                S_WAIT2: state <= S_LOOKUP;

                S_LOOKUP: begin
                    // x_rdata valide -> calcul comb de lut_idx_w
                    lut_idx_reg <= lut_idx_w;
                    silu_val    <= silu_lut[lut_idx_w];
                    state       <= S_WB;
                end

                S_WB: begin
                    out_waddr     <= idx;
                    out_wdata     <= out_clip;
                    out_we        <= 1'b1;
                    dbg_lut_idx   <= lut_idx_reg;
                    dbg_silu_int  <= silu_val;
                    if (idx == D - 1) state <= S_DONE;
                    else begin
                        idx   <= idx + 10'd1;
                        state <= S_ADDR;
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
