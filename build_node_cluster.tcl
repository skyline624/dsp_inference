# P&R du noeud cluster "tout compris" (node + boot SD + sequencer + lien inter-FPGA).
# But : mesurer le vrai % LUT/BSRAM/DSP d'un nœud cluster complet sur GW2AR-18C.
# Build : gw_sh build_node_cluster.tcl  -> impl/pnr/node_cluster.fs
set_device -name GW2AR-18C GW2AR-LV18QN88C8/I7
add_file -type verilog src/mac18.v
add_file -type verilog src/uart_rx_8n1.v
add_file -type verilog src/uart_tx_8n1.v
add_file -type verilog src/gowin_rpll.v
add_file -type verilog src/sdram.v
add_file -type verilog src/rmsnorm_op.v
add_file -type verilog src/silu_op.v
add_file -type verilog src/rope_op.v
add_file -type verilog src/softmax_op.v
add_file -type verilog src/attention_head_op.v
add_file -type verilog sim/sd/sd_ctrl.v
add_file -type verilog src/sd_boot.v
add_file -type verilog sim/link/vec_alu2.v
add_file -type verilog sim/ffn_tp_seq2.v
add_file -type verilog sim/link/link_send.v
add_file -type verilog sim/link/async_fifo.v
add_file -type verilog src/node_cluster.v
add_file -type cst src/coproc_node_cluster.cst
set_option -top_module node_cluster
set_option -output_base_name node_cluster
set_option -include_path src
set_option -place_option 2
set_option -route_option 2
run all