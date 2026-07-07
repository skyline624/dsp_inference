`timescale 1ns/1ps
// Wrapper : routable node (GG removed) + SD-card boot loader.
// `define must be in scope before `include (Gowin/Icarus don't share defines
// across separately-added files), so we include top.v from here.
`define NODE_ONLY
`define SD_BOOT
`include "top.v"
