# cocotb sim of the source-synchronous parallel inter-FPGA link.
#   make            # run the CDC test
#   make clean
SIM           ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(abspath async_fifo.v)
VERILOG_SOURCES += $(abspath ss_link.v)
VERILOG_SOURCES += $(abspath link_send.v)
VERILOG_SOURCES += $(abspath link_recv.v)
VERILOG_SOURCES += $(abspath cluster_link2.v)
VERILOG_SOURCES += $(abspath bp_compare.v)
VERILOG_SOURCES += $(abspath mini_gg.v)
VERILOG_SOURCES += $(abspath cluster_minigg.v)
VERILOG_SOURCES += $(abspath ../models/MULTALU18X18.v)
VERILOG_SOURCES += $(abspath ../../src/mac18.v)
VERILOG_SOURCES += $(abspath tp_mm_node.v)
VERILOG_SOURCES += $(abspath tp_mm_single.v)
VERILOG_SOURCES += $(abspath ring_node.v)
VERILOG_SOURCES += $(abspath ring_cluster.v)
VERILOG_SOURCES += $(abspath tp_node.v)
VERILOG_SOURCES += $(abspath tp_ring_cluster.v)
VERILOG_SOURCES += $(abspath tp_ar_node.v)
VERILOG_SOURCES += $(abspath tp_ar_cluster.v)
VERILOG_SOURCES += $(abspath ../../src/silu_op.v)
VERILOG_SOURCES += $(abspath tp_silu_node.v)
VERILOG_SOURCES += $(abspath vec_alu.v)
VERILOG_SOURCES += $(abspath vec_alu2.v)
VERILOG_SOURCES += $(abspath uart_bridge.v)
VERILOG_SOURCES += $(abspath uart_bridge_tb.v)

TOPLEVEL ?= ss_link
MODULE   ?= test_ss_link

# extra elaboration params, e.g.  make PARAMS=-Pss_link.W=16
COMPILE_ARGS += $(PARAMS)

include $(shell cocotb-config --makefiles)/Makefile.sim
