# P&R du noeud routable + boot SD sur le Tang Nano 20K (GW2AR-18C).
# top_sd_wrap.v : `define NODE_ONLY + `define SD_BOOT puis `include "top.v".
# Build : gw_sh build_sd_node.tcl  -> impl/pnr/dsp_node_sd.fs
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
add_file -type verilog src/top_sd_wrap.v
add_file -type cst src/coproc_sd.cst
set_option -top_module top
set_option -output_base_name dsp_node_sd
set_option -include_path src
set_option -place_option 2
set_option -route_option 2
run all
