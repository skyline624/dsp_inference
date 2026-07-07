# P&R du harnais de bring-up SD sur le Tang Nano 20K (GW2AR-18C).
# Build : gw_sh build_sd_bringup.tcl   (Gowin EDA)
# Le bitstream est genere dans impl/pnr/sd_bringup.fs
set_device -name GW2AR-18C GW2AR-LV18QN88C8/I7
add_file -type verilog src/uart_tx_8n1.v
add_file -type verilog sim/sd/sd_ctrl.v
add_file -type verilog src/sd_bringup.v
add_file -type cst src/sd_bringup.cst
set_option -top_module sd_bringup
set_option -output_base_name sd_bringup
set_option -place_option 2
set_option -route_option 2
run all
