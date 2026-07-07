`timescale 1ns/1ps
// =============================================================================
// MULTALU18X18 - behavioral simulation model (single configuration)
//
// Models the Gowin GW2AR-18 DSP primitive exactly as configured by src/mac18.v:
//   MULTALU18X18_MODE = 0   : ACC +/- (A*B) +/- C   (C tied to 0 here)
//   AREG=BREG=PIPE_REG=OUT_REG=1 -> 3-cycle latency between A,B and DOUT
//   ACCLOAD inverted polarity (see mac18.v header):
//       acc_in = ACCLOAD ? DOUT : 0 ;  DOUT <= acc_in + product
//       ACCLOAD = 1 -> accumulate, ACCLOAD = 0 -> load (overwrite)
//   ACCLOAD_REG0 = ACCLOAD_REG1 = 1 -> ACCLOAD realigned (2 stages) onto the
//       product at the ALU input.
//
// This reproduces the pipeline/latency that the rest of the design relies on.
// Only the configuration used by mac18.v is supported.
// =============================================================================
module MULTALU18X18 (
    output reg  signed [53:0] DOUT,
    output      [54:0]        CASO,
    input  signed [17:0]      A,
    input  signed [17:0]      B,
    input  signed [53:0]      C,
    input  signed [53:0]      D,
    input  [54:0]             CASI,
    input                     ACCLOAD,
    input                     ASIGN,
    input                     BSIGN,
    input                     DSIGN,
    input                     CLK,
    input                     CE,
    input                     RESET
);
    // Parameters accepted via defparam (mac18.v). Behavior is fixed to the
    // single configuration described above; parameters are declared so
    // elaboration succeeds but most are not used by this model.
    parameter AREG             = 1'b1;
    parameter BREG             = 1'b1;
    parameter CREG             = 1'b0;
    parameter DREG             = 1'b0;
    parameter ASIGN_REG        = 1'b0;
    parameter BSIGN_REG        = 1'b0;
    parameter DSIGN_REG        = 1'b0;
    parameter ACCLOAD_REG0     = 1'b1;
    parameter ACCLOAD_REG1     = 1'b1;
    parameter PIPE_REG         = 1'b1;
    parameter OUT_REG          = 1'b1;
    parameter B_ADD_SUB        = 1'b0;
    parameter C_ADD_SUB        = 1'b0;
    parameter MULTALU18X18_MODE = 0;
    parameter MULT_RESET_MODE  = "SYNC";

    assign CASO = 55'b0;   // DSP cascade unused in this design

    reg signed [17:0] a_r, b_r;   // AREG / BREG
    reg signed [35:0] prod_r;     // PIPE_REG (registered product)
    reg               acl0, acl1; // ACCLOAD_REG0 / ACCLOAD_REG1

    always @(posedge CLK) begin
        if (RESET) begin
            a_r    <= 18'sd0;
            b_r    <= 18'sd0;
            prod_r <= 36'sd0;
            acl0   <= 1'b0;
            acl1   <= 1'b0;
            DOUT   <= 54'sd0;
        end else if (CE) begin
            a_r    <= A;
            b_r    <= B;
            prod_r <= a_r * b_r;                                  // PIPE_REG
            acl0   <= ACCLOAD;
            acl1   <= acl0;
            DOUT   <= (acl1 ? DOUT : 54'sd0)                      // OUT_REG + ALU
                      + {{18{prod_r[35]}}, prod_r}
                      + (C_ADD_SUB ? -$signed(C) : $signed(C));   // C = 0 here
        end
    end
endmodule
