// =============================================================================
// rope_op.v  --  Rotary Positional Embedding (RoPE) en int8 + shift
//
// Pour chaque paire (x[2i], x[2i+1]) du vecteur x[HS] :
//   new_real = x[2i] * cos[i] - x[2i+1] * sin[i]
//   new_imag = x[2i] * sin[i] + x[2i+1] * cos[i]
//   out[2i]   = new_real
//   out[2i+1] = new_imag
//
// cos[i], sin[i] : signed Q15 (int16, scale 2^-15) fournis en entree (regs externes)
// Output : shift_out = shift_x (la rotation preserve la magnitude)
//
// FSM par paire i :
//   RDX1 : x_raddr <= 2i      | wait 2 cycles
//   RDX2 : x_raddr <= 2i+1    | wait 2 cycles, latch x[2i]
//   COMP : latch x[2i+1], pair_idx <= i (cos_in/sin_in deviennent valides)
//   WB1  : ecrit new_real a out[2i]
//   WB2  : ecrit new_imag a out[2i+1]
//
// Cf. [[gowin-signed-mult-gotcha]] pour les regs signed locaux.
// =============================================================================

module rope_op #(
    parameter HS   = 8,    // head_size (stories260K = 8)
    parameter HALF = 4     // HS / 2
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,

    input  wire signed [7:0] shift_x,
    output reg  signed [7:0] shift_out,

    // x buffer (read/write)
    output reg  [9:0]        x_raddr,
    input  wire signed [7:0] x_rdata,
    output reg  [9:0]        out_waddr,
    output reg  signed [7:0] out_wdata,
    output reg               out_we,

    // cos/sin externes (fournis combinationnel par top.v selon pair_idx)
    output reg  [3:0]        pair_idx,
    input  wire signed [15:0] cos_in,
    input  wire signed [15:0] sin_in,

    // Debug (dernier element traite)
    output reg  signed [31:0] dbg_new_real,
    output reg  signed [31:0] dbg_new_imag
);

    localparam S_IDLE   = 4'd0,
               S_RDX1_A = 4'd1,
               S_RDX1_W1= 4'd2,
               S_RDX1_W2= 4'd3,
               S_RDX1_D = 4'd4,
               S_RDX2_A = 4'd5,
               S_RDX2_W1= 4'd6,
               S_RDX2_W2= 4'd7,
               S_COMP   = 4'd8,
               S_WB1    = 4'd9,
               S_WB2    = 4'd10,
               S_DONE   = 4'd11;

    reg [3:0] state;
    reg [3:0] i;            // pair index
    reg signed [7:0] xr_lat, xi_lat;

    // Regs signed locaux pour forcer mult signee (gotcha Gowin)
    reg signed [15:0] xr16, xi16;
    reg signed [31:0] m_xr_cos, m_xi_sin, m_xr_sin, m_xi_cos;
    reg signed [31:0] new_real_raw, new_imag_raw;
    reg signed [31:0] new_real_round, new_imag_round;
    reg signed [7:0]  new_real_clip, new_imag_clip;

    always @(*) begin
        xr16 = {{8{xr_lat[7]}}, xr_lat};
        xi16 = {{8{xi_lat[7]}}, xi_lat};
        // Tous les mults signed 16x16 grace aux regs intermediaires signed
        m_xr_cos = xr16 * cos_in;
        m_xi_sin = xi16 * sin_in;
        m_xr_sin = xr16 * sin_in;
        m_xi_cos = xi16 * cos_in;
        new_real_raw = m_xr_cos - m_xi_sin;
        new_imag_raw = m_xr_sin + m_xi_cos;
        // Arrondi : +1<<14, puis >> 15
        new_real_round = (new_real_raw + 32'sd16384) >>> 15;
        new_imag_round = (new_imag_raw + 32'sd16384) >>> 15;
        new_real_clip  = (new_real_round > 32'sd127)   ? 8'sd127  :
                         (new_real_round < -32'sd128)  ? -8'sd128 :
                                                          new_real_round[7:0];
        new_imag_clip  = (new_imag_round > 32'sd127)   ? 8'sd127  :
                         (new_imag_round < -32'sd128)  ? -8'sd128 :
                                                          new_imag_round[7:0];
    end

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            done     <= 1'b0;
            out_we   <= 1'b0;
            pair_idx <= 4'd0;
        end else begin
            out_we <= 1'b0;
            done   <= 1'b0;

            case (state)
                S_IDLE: if (start) begin
                    i         <= 4'd0;
                    pair_idx  <= 4'd0;
                    shift_out <= shift_x;
                    state     <= S_RDX1_A;
                end

                S_RDX1_A: begin
                    x_raddr <= {6'd0, i, 1'b0};   // 2*i
                    state   <= S_RDX1_W1;
                end
                S_RDX1_W1: state <= S_RDX1_W2;
                S_RDX1_W2: state <= S_RDX1_D;
                S_RDX1_D: begin
                    xr_lat <= x_rdata;            // x[2i] valide
                    state  <= S_RDX2_A;
                end

                S_RDX2_A: begin
                    x_raddr <= {6'd0, i, 1'b1};   // 2*i + 1
                    state   <= S_RDX2_W1;
                end
                S_RDX2_W1: state <= S_RDX2_W2;
                S_RDX2_W2: state <= S_COMP;

                S_COMP: begin
                    xi_lat   <= x_rdata;
                    pair_idx <= i;                // cos_in / sin_in deviennent valides
                    state    <= S_WB1;
                end

                S_WB1: begin
                    out_waddr <= {6'd0, i, 1'b0};
                    out_wdata <= new_real_clip;
                    out_we    <= 1'b1;
                    state     <= S_WB2;
                end

                S_WB2: begin
                    out_waddr <= {6'd0, i, 1'b1};
                    out_wdata <= new_imag_clip;
                    out_we    <= 1'b1;
                    dbg_new_real <= new_real_raw;
                    dbg_new_imag <= new_imag_raw;
                    if (i == HALF - 1) state <= S_DONE;
                    else begin
                        i     <= i + 4'd1;
                        state <= S_RDX1_A;
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
