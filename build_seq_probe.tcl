# P&R : seq_probe = sequencer FFN-TP + vec_alu + lien, SANS le noeud. Isoler le cout LUT.
set_device -name GW2AR-18C GW2AR-LV18QN88C8/I7
add_file -type verilog src/uart_rx_8n1.v
add_file -type verilog src/uart_tx_8n1.v
add_file -type verilog src/mac18.v
add_file -type verilog sim/link/vec_alu2.v
add_file -type verilog sim/ffn_tp_seq2.v
add_file -type verilog sim/link/link_send.v
add_file -type verilog sim/link/async_fifo.v
add_file -type verilog src/seq_probe.v
add_file -type cst src/coproc_seq_probe.cst
set_option -top_module seq_probe
set_option -output_base_name seq_probe
set_option -include_path src
set_option -place_option 2
set_option -route_option 2
run all