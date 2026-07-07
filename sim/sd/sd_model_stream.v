`timescale 1ns/1ps
// =============================================================================
// sd_model_stream - run a model BIGGER than SDRAM, by streaming layers (SD-4)
//
// The whole model (L layers, each W_l = [M,K]) lives on the SD card. The SDRAM
// holds only ONE layer at a time (a small cache region, reused). For each layer:
//   1. stream layer l's weights  SD -> SDRAM[0 .. M*K-1]   (overwrites previous)
//   2. matmul  y_l = W_l @ x , reading W_l back from SDRAM, accumulate on the DSP
// So the model size = L * M * K (on SD, unbounded) while the fast memory used is
// just M * K (one layer). Output y_l per layer -> validated bit-exact.
// =============================================================================
module sd_model_stream #(parameter L = 4, parameter M = 8, parameter K = 64, parameter LAT = 4) (
    input  wire clk, rstn, start,
    input  wire [8*64-1:0] x_in,
    output reg  done
);
    wire clk_sdram = ~clk;

    // ---- SD ----
    wire cs_n, sclk, mosi, miso; wire sd_we; wire [7:0] sd_data; wire sd_rd_done;
    reg  sd_start; reg [15:0] sd_blk;
    sd_reader #(.HALF(2)) u_rd (
        .clk(clk), .rstn(rstn), .start(sd_start), .blk_start(sd_blk), .nblk(16'd1),
        .cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso),
        .out_we(sd_we), .out_data(sd_data), .done(sd_rd_done));
    sd_spi_model #(.NBLK(L)) u_sd (.cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso));

    // ---- SDRAM ----
    wire [31:0] DQ; wire [10:0] A; wire [1:0] BA;
    wire nCS,nWE,nRAS,nCAS,SCLK,CKE; wire [3:0] DQM;
    reg dr_rd, dr_wr; reg [22:0] dr_addr; reg [7:0] dr_din; wire [7:0] dr_dout; wire dr_ready, dr_busy;
    sdram #(.FREQ(2_000_000)) u_ctrl (
        .SDRAM_DQ(DQ), .SDRAM_A(A), .SDRAM_BA(BA), .SDRAM_nCS(nCS), .SDRAM_nWE(nWE),
        .SDRAM_nRAS(nRAS), .SDRAM_nCAS(nCAS), .SDRAM_CLK(SCLK), .SDRAM_CKE(CKE), .SDRAM_DQM(DQM),
        .clk(clk), .clk_sdram(clk_sdram), .resetn(rstn),
        .rd(dr_rd), .wr(dr_wr), .refresh(1'b0), .addr(dr_addr), .din(dr_din),
        .dout(dr_dout), .dout32(), .data_ready(dr_ready), .busy(dr_busy));
    sdram_chip u_chip (
        .SDRAM_CLK(SCLK), .SDRAM_CKE(CKE), .SDRAM_nCS(nCS), .SDRAM_nRAS(nRAS),
        .SDRAM_nCAS(nCAS), .SDRAM_nWE(nWE), .SDRAM_BA(BA), .SDRAM_A(A), .SDRAM_DQM(DQM), .SDRAM_DQ(DQ));

    // ---- DSP ----
    reg signed [17:0] mac_a, mac_b; reg mac_load; wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rstn), .ce(1'b1), .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));

    reg signed [31:0] yout [0:L*M-1];
    reg [7:0] wbyte; reg [22:0] waddr; reg wpend, sdone_l;
    reg [3:0]  layer; reg [3:0] r; reg [6:0] k; reg [7:0] cf;
    wire [7:0] xk = x_in[k*8 +: 8];

    localparam S_INIT=0, ST_KICK=1, ST_LOAD=2, MM_ROW=3, MM_RD=4, MM_RDW=5,
               MM_FEED=6, MM_FLUSH=7, MM_NEXT=8, S_DONE=9;
    reg [3:0] st;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            st<=S_INIT; sd_start<=0; dr_wr<=0; dr_rd<=0; wpend<=0; sdone_l<=0;
            layer<=0; done<=0; mac_load<=0; mac_a<=0; mac_b<=0;
        end else begin
            sd_start<=0; dr_wr<=0; dr_rd<=0; done<=0; mac_load<=0; mac_a<=0; mac_b<=0;
            case (st)
                S_INIT: if (!dr_busy) begin layer<=0; st<=ST_KICK; end

                // ---- stream layer `layer` : SD block -> SDRAM[0 .. M*K-1] ----
                ST_KICK: begin sd_blk<=layer; sd_start<=1; waddr<=0; wpend<=0; sdone_l<=0; st<=ST_LOAD; end
                ST_LOAD: begin
                    if (sd_we)      begin wbyte<=sd_data; wpend<=1; end
                    if (sd_rd_done) sdone_l<=1;
                    if (wpend && !dr_busy) begin
                        dr_addr<=waddr; dr_din<=wbyte; dr_wr<=1; wpend<=0; waddr<=waddr+23'd1;
                    end
                    if (sdone_l && !wpend && !dr_busy) begin r<=0; st<=MM_ROW; end
                end

                // ---- matmul y_l = W_l @ x , reading W_l back from SDRAM ----
                MM_ROW: begin k<=0; st<=MM_RD; end
                MM_RD:  if (!dr_busy) begin dr_addr<=r*K+k; dr_rd<=1; st<=MM_RDW; end
                MM_RDW: if (dr_ready) begin wbyte<=dr_dout; st<=MM_FEED; end
                MM_FEED: begin
                    mac_a <= {{10{wbyte[7]}}, wbyte};
                    mac_b <= {{10{xk[7]}}, xk};
                    mac_load <= (k==0);
                    if (k==K-1) begin cf<=0; st<=MM_FLUSH; end else begin k<=k+7'd1; st<=MM_RD; end
                end
                MM_FLUSH: if (cf==LAT) begin yout[layer*M + r] <= mac_res[31:0]; st<=MM_NEXT; end
                          else cf<=cf+8'd1;
                MM_NEXT: if (r==M-1) begin
                             if (layer==L-1) st<=S_DONE;
                             else begin layer<=layer+4'd1; st<=ST_KICK; end
                         end else begin r<=r+4'd1; st<=MM_ROW; end

                S_DONE: begin done<=1; st<=S_INIT; end
            endcase
        end
    end
endmodule
