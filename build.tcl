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
# Higher placement/routing effort to ease congestion (~80% LUT usage)
# Even with max effort (place 2 + route 2), v5g v2 fails to route (156 unrouted nets).
# Design is too dense for GW2AR-18 (only 20k LUTs). Need either:
# - Major RTL refactor (factor SETUP states, share BSRAMs, drop unused modules)
# - Bigger FPGA (GW2AR-55 or larger)
# - PC-orchestrated fallback (host/infer_fpga.py) works fine
set_option -place_option 2
set_option -route_option 2
run all
