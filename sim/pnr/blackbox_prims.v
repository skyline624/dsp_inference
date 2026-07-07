// Blackbox declarations for the Gowin hard primitives, so yosys synthesis can
// count the LUT/FF/BRAM logic WITHOUT trying to map the PLL and DSP (which are
// dedicated hard blocks, not LUTs). Used only for the LUT-estimate flow.
(* blackbox *)
module rPLL (
    output CLKOUT, output LOCK, output CLKOUTP, output CLKOUTD, output CLKOUTD3,
    input  RESET, input RESET_P, input CLKIN, input CLKFB,
    input [5:0] FBDSEL, input [5:0] IDSEL, input [5:0] ODSEL,
    input [3:0] PSDA, input [3:0] DUTYDA, input [3:0] FDLY
);
endmodule

(* blackbox *)
module MULTALU18X18 (
    output [53:0] DOUT, output [54:0] CASO,
    input  [17:0] A, input [17:0] B, input [53:0] C, input [53:0] D,
    input  [54:0] CASI,
    input  ACCLOAD, input ASIGN, input BSIGN, input DSIGN,
    input  CLK, input CE, input RESET
);
endmodule
