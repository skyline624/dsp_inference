`timescale 1ns/1ps
// =============================================================================
// sdram_chip - behavioral SDR SDRAM model (functional, not timing-accurate)
//
// Models the GW2AR-18 embedded SDRAM as driven by the NESTang controller
// (src/sdram.v): 32-bit data, 4 banks x 2048 rows x 256 columns = 64 Mbit.
//
// Decodes the JEDEC SDR command set from {nRAS,nCAS,nWE} (nCS=0, CKE=1):
//   Activate (011)  : latch active row for the addressed bank
//   Read     (101)  : drive DQ with mem[bank,row,col] after CAS latency
//   Write    (100)  : store DQ into mem[bank,row,col], honoring DQM byte mask
//   Refresh/Precharge/SetMode/NOP : no data effect
//
// Sampled on posedge SDRAM_CLK. The controller drives commands on clk_sys
// (= clkin); SDRAM_CLK = clk_sdram = ~clkin, so this model samples half a
// cycle after the controller updates its pins -> clean, race-free capture.
//
// Timing parameters (tRCD/tRP/tRC) are NOT enforced; this is for functional
// verification of the design and the FPGA cluster, not for SDRAM sign-off.
// =============================================================================
module sdram_chip #(
    parameter ROW_BITS = 11,   // 2048 rows
    parameter COL_BITS = 8,    // 256 columns
    parameter CAS      = 2     // read latency (mode register, matches sdram.v)
) (
    input  wire        SDRAM_CLK,
    input  wire        SDRAM_CKE,
    input  wire        SDRAM_nCS,
    input  wire        SDRAM_nRAS,
    input  wire        SDRAM_nCAS,
    input  wire        SDRAM_nWE,
    input  wire [1:0]  SDRAM_BA,
    input  wire [ROW_BITS-1:0] SDRAM_A,
    input  wire [3:0]  SDRAM_DQM,
    inout  wire [31:0] SDRAM_DQ
);
    localparam [2:0] CMD_SetMode   = 3'b000;
    localparam [2:0] CMD_Refresh   = 3'b001;
    localparam [2:0] CMD_Precharge = 3'b010;
    localparam [2:0] CMD_Activate  = 3'b011;
    localparam [2:0] CMD_Write     = 3'b100;
    localparam [2:0] CMD_Read      = 3'b101;
    localparam [2:0] CMD_NOP       = 3'b111;

    localparam IDX_BITS = 2 + ROW_BITS + COL_BITS;   // bank + row + col = 21

    reg [31:0] mem [0:(1<<IDX_BITS)-1];
    reg [ROW_BITS-1:0] active_row [0:3];

    // DQ drive (read data)
    reg [31:0] dq_out;
    reg        dq_oe;
    assign SDRAM_DQ = dq_oe ? dq_out : 32'bz;

    // read-data delay line (CAS cycles)
    reg [31:0] rd_word;
    reg [3:0]  rd_cnt;     // counts down to the data-valid cycle (0 = idle)

    reg [2:0] cmd;
    reg [IDX_BITS-1:0] idx;

    integer i;
    initial begin
        dq_oe  = 1'b0;
        dq_out = 32'b0;
        rd_cnt = 4'd0;
        for (i = 0; i < 4; i = i + 1) active_row[i] = {ROW_BITS{1'b0}};
    end

    always @(posedge SDRAM_CLK) begin
        cmd = (SDRAM_CKE && !SDRAM_nCS) ? {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE}
                                        : CMD_NOP;

        case (cmd)
            CMD_Activate: active_row[SDRAM_BA] <= SDRAM_A;

            CMD_Write: begin
                idx = {SDRAM_BA, active_row[SDRAM_BA], SDRAM_A[COL_BITS-1:0]};
                if (!SDRAM_DQM[0]) mem[idx][7:0]   <= SDRAM_DQ[7:0];
                if (!SDRAM_DQM[1]) mem[idx][15:8]  <= SDRAM_DQ[15:8];
                if (!SDRAM_DQM[2]) mem[idx][23:16] <= SDRAM_DQ[23:16];
                if (!SDRAM_DQM[3]) mem[idx][31:24] <= SDRAM_DQ[31:24];
            end

            CMD_Read: begin
                idx     = {SDRAM_BA, active_row[SDRAM_BA], SDRAM_A[COL_BITS-1:0]};
                rd_word <= mem[idx];
                rd_cnt  <= CAS[3:0];
            end
        endcase

        // read-data output pipeline
        if (rd_cnt > 4'd1) begin
            rd_cnt <= rd_cnt - 4'd1;
            dq_oe  <= 1'b0;
        end else if (rd_cnt == 4'd1) begin
            rd_cnt <= 4'd0;
            dq_out <= rd_word;
            dq_oe  <= 1'b1;        // drive data on this cycle (= read + CAS)
        end else begin
            dq_oe  <= 1'b0;        // release the bus
        end
    end
endmodule
