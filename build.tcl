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
add_file -type verilog src/top.v
add_file -type cst src/coproc.cst
set_option -top_module top
set_option -output_base_name dsp_inference
set_option -include_path src
run all
