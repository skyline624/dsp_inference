# =============================================================================
# cocotb simulation of the FPGA cluster (Icarus Verilog backend)
#
# Usage (inside the hdlc/sim:osvb container, from this directory):
#   make            # 1 node  (smoke / single-FPGA bring-up)
#   make N=2        # 2 nodes daisy-chained over UART
#   make clean
#
# The host helper sim/run.ps1 wraps this in Docker and copies the LUT .hex
# files (loaded by $readmemh with relative paths) next to the testbench.
# =============================================================================

SIM           ?= icarus
TOPLEVEL_LANG ?= verilog
N             ?= 1

RTL := $(abspath ..)/src

# Simulation models replace the un-simulatable Gowin primitives.
VERILOG_SOURCES += $(abspath models/Gowin_rPLL.v)
VERILOG_SOURCES += $(abspath models/MULTALU18X18.v)
VERILOG_SOURCES += $(abspath models/sdram_chip.v)

# Real RTL (note: src/gowin_rpll.v is intentionally excluded - replaced above).
VERILOG_SOURCES += $(RTL)/mac18.v
VERILOG_SOURCES += $(RTL)/rmsnorm_op.v
VERILOG_SOURCES += $(RTL)/silu_op.v
VERILOG_SOURCES += $(RTL)/rope_op.v
VERILOG_SOURCES += $(RTL)/softmax_op.v
VERILOG_SOURCES += $(RTL)/attention_head_op.v
VERILOG_SOURCES += $(RTL)/sdram.v
VERILOG_SOURCES += $(RTL)/uart_rx_8n1.v
VERILOG_SOURCES += $(RTL)/uart_tx_8n1.v
VERILOG_SOURCES += $(RTL)/top.v

# Cluster wrappers.
VERILOG_SOURCES += $(abspath node_top.v)
VERILOG_SOURCES += $(abspath cluster_top.v)
VERILOG_SOURCES += $(abspath cluster_tp.v)
VERILOG_SOURCES += $(abspath seq.v)
VERILOG_SOURCES += $(abspath seq_node.v)
VERILOG_SOURCES += $(abspath link/vec_alu.v)
VERILOG_SOURCES += $(abspath link/vec_alu2.v)
VERILOG_SOURCES += $(abspath ffn_seq.v)
VERILOG_SOURCES += $(abspath ffn_top.v)
VERILOG_SOURCES += $(abspath ffn_tp_seq.v)
VERILOG_SOURCES += $(abspath ffn_tp_top.v)
VERILOG_SOURCES += $(abspath ffn_tp_seq2.v)
VERILOG_SOURCES += $(abspath ffn_tp_seq2_top.v)
VERILOG_SOURCES += $(abspath attn_tp_seq.v)
VERILOG_SOURCES += $(abspath attn_tp_top.v)
VERILOG_SOURCES += $(abspath layer_tp_top.v)
VERILOG_SOURCES += $(abspath model_tp_top.v)

TOPLEVEL := cluster_top
MODULE   ?= test_cluster

# Override the cluster size parameter at elaboration time.
COMPILE_ARGS += -Pcluster_top.N=$(N)
# extra elaboration params, e.g.  make PARAMS=-Pseq_node.N_OPS=2
COMPILE_ARGS += $(PARAMS)

include $(shell cocotb-config --makefiles)/Makefile.sim
