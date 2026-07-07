`timescale 1ns/1ps
// =============================================================================
// sd_boot_top - autonomous "bootloader": SD card -> real SDRAM, at power-on (SD-3)
//
// On reset, once the SDRAM controller finishes its power-on init, the bootloader
// reads the model from the SD card (sd_reader) and writes every byte into the
// REAL SDRAM (src/sdram.v controller + behavioral chip). No PC, no UART: the
// weights load themselves from the SD into SDRAM. After this, the brick datapath
// reads them exactly as in the validated FPGA inference.
//
// (refresh omitted: the behavioral chip does not decay; real HW interleaves it.)
// =============================================================================
module sd_boot_top #(parameter NBLK = 2) (
    input  wire clk, rstn, start,
    output reg  done
);
    wire clk_sdram = ~clk;

    // ---- SD card + controller ----
    wire cs_n, sclk, mosi, miso;
    wire sd_we; wire [7:0] sd_data; wire sd_rd_done;
    reg  sd_start;
    sd_reader #(.HALF(2)) u_rd (
        .clk(clk), .rstn(rstn), .start(sd_start), .blk_start(16'd0), .nblk(NBLK[15:0]),
        .cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso),
        .out_we(sd_we), .out_data(sd_data), .done(sd_rd_done));
    sd_spi_model #(.NBLK(NBLK)) u_sd (.cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso));

    // ---- real SDRAM controller + behavioral chip ----
    wire [31:0] DQ; wire [10:0] A; wire [1:0] BA;
    wire nCS, nWE, nRAS, nCAS, SCLK, CKE; wire [3:0] DQM;
    reg  dr_rd, dr_wr; reg [22:0] dr_addr; reg [7:0] dr_din;
    wire [7:0] dr_dout; wire dr_ready, dr_busy;
    sdram #(.FREQ(2_000_000)) u_ctrl (
        .SDRAM_DQ(DQ), .SDRAM_A(A), .SDRAM_BA(BA), .SDRAM_nCS(nCS), .SDRAM_nWE(nWE),
        .SDRAM_nRAS(nRAS), .SDRAM_nCAS(nCAS), .SDRAM_CLK(SCLK), .SDRAM_CKE(CKE), .SDRAM_DQM(DQM),
        .clk(clk), .clk_sdram(clk_sdram), .resetn(rstn),
        .rd(dr_rd), .wr(dr_wr), .refresh(1'b0), .addr(dr_addr), .din(dr_din),
        .dout(dr_dout), .dout32(), .data_ready(dr_ready), .busy(dr_busy));
    sdram_chip u_chip (
        .SDRAM_CLK(SCLK), .SDRAM_CKE(CKE), .SDRAM_nCS(nCS), .SDRAM_nRAS(nRAS),
        .SDRAM_nCAS(nCAS), .SDRAM_nWE(nWE), .SDRAM_BA(BA), .SDRAM_A(A), .SDRAM_DQM(DQM), .SDRAM_DQ(DQ));

    // ---- bootloader FSM ----
    reg [22:0] waddr; reg wpend; reg [7:0] wbyte; reg sdone_l;
    localparam B_INIT=0, B_LOAD=1, B_DONE=2;
    reg [1:0] bst;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin bst<=B_INIT; sd_start<=0; dr_wr<=0; dr_rd<=0; wpend<=0; waddr<=0; done<=0; sdone_l<=0; end
        else begin
            sd_start<=0; dr_wr<=0; done<=0;
            case (bst)
                B_INIT: if (!dr_busy) begin sd_start<=1; waddr<=0; wpend<=0; sdone_l<=0; bst<=B_LOAD; end
                B_LOAD: begin
                    if (sd_we)      begin wbyte<=sd_data; wpend<=1; end
                    if (sd_rd_done) sdone_l<=1;
                    if (wpend && !dr_busy) begin                       // write the pending byte into SDRAM
                        dr_addr<=waddr; dr_din<=wbyte; dr_wr<=1; wpend<=0; waddr<=waddr+23'd1;
                    end
                    if (sdone_l && !wpend && !dr_busy) bst<=B_DONE;    // all bytes loaded
                end
                B_DONE: begin done<=1; bst<=B_INIT; end
            endcase
        end
    end
endmodule
