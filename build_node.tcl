# P&R of a CLUSTER NODE on the GW2AR-18 (GG generation FSM removed via NODE_ONLY).
# top_node_wrap.v does `define NODE_ONLY then `include "top.v" so the macro reaches
# top.v (Gowin doesn't share defines across separately-added files).
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
add_file -type verilog src/top_node_wrap.v
add_file -type cst src/coproc.cst
set_option -top_module top
set_option -output_base_name dsp_node
set_option -include_path src
set_option -place_option 2
set_option -route_option 2
run all
