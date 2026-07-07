# P&R du noeud de generation autonome "tout compris"
#   node (top LINK_SS) + boot SD + gen_seq + lien ss_link (async_fifo cmd/resp).
# But : mesurer le vrai % LUT/BSRAM/DSP/CLS d'un noeud de generation complet sur
# GW2AR-18C, et PROUVER que le sequenceur gen_seq (sim etape F verte) RENTRE et
# ROUTE sur la vraie Tang Nano 20K.
# Build : gw_sh build_node_gen.tcl  -> impl/pnr/node_gen.fs
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
add_file -type verilog sim/link/async_fifo.v
add_file -type verilog sim/link/uart_bridge.v
add_file -type verilog sim/link/gen_seq.v
add_file -type verilog src/node_gen.v
add_file -type cst src/coproc_node_gen.cst
set_option -top_module node_gen
set_option -output_base_name node_gen
set_option -include_path src
set_option -place_option 2
set_option -route_option 2
run all
