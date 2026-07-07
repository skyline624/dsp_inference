# SD-card streaming bring-up (standalone, no top.v dependency).
SIM           ?= icarus
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES += $(abspath ../models/MULTALU18X18.v)
VERILOG_SOURCES += $(abspath ../models/sdram_chip.v)
VERILOG_SOURCES += $(abspath ../../src/mac18.v)
VERILOG_SOURCES += $(abspath ../../src/sdram.v)
VERILOG_SOURCES += $(abspath sd_spi_model.v)
VERILOG_SOURCES += $(abspath sd_reader.v)
VERILOG_SOURCES += $(abspath sd_stream_top.v)
VERILOG_SOURCES += $(abspath sd_layer_stream.v)
VERILOG_SOURCES += $(abspath sd_boot_top.v)
VERILOG_SOURCES += $(abspath sd_model_stream.v)
VERILOG_SOURCES += $(abspath sd_card.v)
VERILOG_SOURCES += $(abspath sd_ctrl.v)
VERILOG_SOURCES += $(abspath sd_init_top.v)
VERILOG_SOURCES += $(abspath ../../src/uart_tx_8n1.v)
VERILOG_SOURCES += $(abspath ../../src/sd_bringup.v)
VERILOG_SOURCES += $(abspath sd_bringup_tb.v)
VERILOG_SOURCES += $(abspath ../../src/sd_boot.v)
VERILOG_SOURCES += $(abspath sd_boot_top2.v)

TOPLEVEL ?= sd_stream_top
MODULE   ?= test_sd_stream

include $(shell cocotb-config --makefiles)/Makefile.sim
