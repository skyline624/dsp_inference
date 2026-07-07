`timescale 1ns/1ps
// =============================================================================
// cluster_top - N FPGA nodes daisy-chained over UART
//
//   head_rx --> [node 0] --tx--> [node 1] --tx--> ... --> [node N-1] --> tail_tx
//
// All nodes share the same 27 MHz clk (in a real cluster each board has its own
// oscillator; for functional verification a common clock is the simplest and
// removes CDC noise from the architecture exploration -- the inter-node link is
// still a real, byte-serial UART so its protocol is exercised faithfully).
//
// N is overridable at compile time:  iverilog -Pcluster_top.N=2 ...
// =============================================================================
module cluster_top #(
    parameter N = 1
) (
    input  wire           clk,
    input  wire           head_rx,   // serial input into node 0
    output wire           tail_tx,   // serial output from node N-1
    output wire [6*N-1:0] led
);
    wire [N:0] chain;          // chain[0]=head_rx ; chain[i+1]=node i tx
    assign chain[0] = head_rx;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : nodes
            node_top u_node (
                .clk     (clk),
                .uart_rx (chain[i]),
                .uart_tx (chain[i+1]),
                .led     (led[6*i +: 6])
            );
        end
    endgenerate

    assign tail_tx = chain[N];
endmodule
