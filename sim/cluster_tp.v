`timescale 1ns/1ps
// =============================================================================
// cluster_tp - tensor-parallel cluster (star topology), N=2 nodes
//
// Unlike cluster_top (UART daisy-chain, for layer-pipeline), this exposes each
// node's UART independently so a coordinator can BROADCAST the input activation
// to all nodes and GATHER their partial outputs -- the communication pattern of
// row-parallel (output-split) tensor parallelism:
//
//     x ---broadcast--> [node 0 : W rows 0..n-1 ]  --> y[0..n-1]
//       \--broadcast--> [node 1 : W rows n..2n-1]  --> y[n..2n-1]   (gather)
//
// Each node holds ONLY its slice of the weights, in its OWN SDRAM -> parallel
// memory bandwidth + parallel DSP compute. Activations (x, y) are tiny and are
// the only thing crossing between the coordinator and the nodes.
// =============================================================================
module cluster_tp (
    input  wire        clk,
    input  wire        rx0,
    output wire        tx0,
    input  wire        rx1,
    output wire        tx1,
    output wire [11:0] led
);
    node_top n0 (.clk(clk), .uart_rx(rx0), .uart_tx(tx0), .led(led[5:0]));
    node_top n1 (.clk(clk), .uart_rx(rx1), .uart_tx(tx1), .led(led[11:6]));
endmodule
