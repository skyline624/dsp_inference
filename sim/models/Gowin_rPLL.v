`timescale 1ns/1ps
// =============================================================================
// Gowin_rPLL - simulation stub
//
// Functional replacement for src/gowin_rpll.v (which instantiates the Gowin
// hard rPLL primitive, not simulatable without the vendor prim_sim.v library).
//
// The real design uses:
//   clkout  -> clk_sys   (27 MHz, same frequency as clkin)
//   clkoutp -> clk_sdram (180-degree phase-shifted clock for the SDRAM)
//   lock    -> gates the internal reset (init counter)
//
// For functional simulation we pass clkin straight through and provide a
// 180-degree shifted copy for the SDRAM domain. Phase is not timing-accurate
// but is consistent, which is all the behavioral SDRAM chip model needs.
// =============================================================================
module Gowin_rPLL (
    output wire clkout,
    output wire clkoutp,
    output reg  lock,
    input  wire reset,
    input  wire clkin
);
    assign clkout  =  clkin;   // clk_sys
    assign clkoutp = ~clkin;   // clk_sdram : 180 deg

    initial lock = 1'b0;
    always @(posedge clkin or posedge reset) begin
        if (reset) lock <= 1'b0;
        else       lock <= 1'b1;   // locks on the first clock edge
    end
endmodule
