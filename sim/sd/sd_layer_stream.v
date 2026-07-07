`timescale 1ns/1ps
// =============================================================================
// sd_layer_stream - double-buffered SD -> cache -> compute, with overlap (SD-2)
//
// Demonstrates the memory hierarchy: a matrix W = [T*8, 64] lives on the SD card
// (one 512-byte block = one tile = 8 rows x 64). The controller streams tiles
// through TWO ping-pong cache buffers: while the DSP computes the matmul of the
// current tile (from one buffer), the SD reader loads the NEXT tile into the
// other buffer. Output y = W @ x (int32 per row) -> validated bit-exact.
//
// The two reg buffers play the role of two SDRAM regions; the double-buffer
// control logic is identical regardless of the backing memory.
// =============================================================================
module sd_layer_stream #(parameter T = 4, parameter HALF = 2, parameter LAT = 4) (
    input  wire clk, rstn, start,
    input  wire [8*64-1:0] x_in,           // 64 int8
    output reg  done
);
    localparam K = 64, ROWS = 8;

    // ---- SD reader + card model ----
    wire cs_n, sclk, mosi, miso;
    wire rd_we; wire [7:0] rd_data; wire rd_done;
    reg  rd_start; reg [15:0] rd_blk;
    sd_reader #(.HALF(HALF)) u_rd (
        .clk(clk), .rstn(rstn), .start(rd_start), .blk_start(rd_blk), .nblk(16'd1),
        .cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso),
        .out_we(rd_we), .out_data(rd_data), .done(rd_done));
    sd_spi_model #(.NBLK(T)) u_sd (.cs_n(cs_n), .sclk(sclk), .mosi(mosi), .miso(miso));

    // ---- two ping-pong cache buffers (one tile each = 512 bytes) ----
    reg [7:0] buf0 [0:511];
    reg [7:0] buf1 [0:511];
    reg       load_sel;      // buffer the reader fills
    reg [9:0] widx; reg ld_rst;
    always @(posedge clk) begin
        if (ld_rst) widx <= 0;
        else if (rd_we) begin
            if (load_sel) buf1[widx] <= rd_data; else buf0[widx] <= rd_data;
            widx <= widx + 10'd1;
        end
    end

    // ---- DSP consumer (matmul of one tile) ----
    reg signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rstn), .ce(1'b1), .load(mac_load),
                 .a(mac_a), .b(mac_b), .result(mac_res));

    reg comp_sel;            // buffer the consumer reads
    reg signed [31:0] yout [0:T*ROWS-1];
    reg [3:0] tile_idx;

    reg [2:0] cst; reg comp_go, comp_done;
    reg [3:0] crow; reg [6:0] ck; reg [7:0] cf;
    localparam C_IDLE=0, C_RUN=1, C_FLUSH=2, C_NEXT=3;
    wire [9:0] caddr = crow*K + ck;
    wire [7:0] cw = comp_sel ? buf1[caddr] : buf0[caddr];
    wire [7:0] xk = x_in[ck*8 +: 8];

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin cst<=C_IDLE; comp_done<=0; mac_load<=0; crow<=0; ck<=0; end
        else begin
            comp_done<=0;
            case (cst)
                C_IDLE: if (comp_go) begin crow<=0; ck<=0; cst<=C_RUN; end
                C_RUN: begin
                    mac_a <= {{10{cw[7]}}, cw};
                    mac_b <= {{10{xk[7]}}, xk};
                    mac_load <= (ck==0);
                    if (ck==K-1) begin cf<=0; cst<=C_FLUSH; end else ck<=ck+7'd1;
                end
                C_FLUSH: begin
                    mac_load<=0; mac_a<=0; mac_b<=0;
                    if (cf==LAT) begin
                        yout[tile_idx*ROWS + crow] <= mac_res[31:0];
                        cst<=C_NEXT;
                    end else cf<=cf+8'd1;
                end
                C_NEXT: if (crow==ROWS-1) begin comp_done<=1; cst<=C_IDLE; end
                        else begin crow<=crow+4'd1; ck<=0; cst<=C_RUN; end
            endcase
        end
    end

    // ---- main controller : prime, then loop (compute || load next), swap ----
    localparam M_IDLE=0, M_PRIME=1, M_PRIMEW=2, M_RUN=3, M_WAIT=4, M_DONE=5;
    reg [2:0] mst; reg load_issued, cdone_l, rdone_l;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin mst<=M_IDLE; rd_start<=0; ld_rst<=0; comp_go<=0; done<=0;
                         load_sel<=0; comp_sel<=0; tile_idx<=0; load_issued<=0;
                         cdone_l<=0; rdone_l<=0; end
        else begin
            rd_start<=0; ld_rst<=0; comp_go<=0; done<=0;
            case (mst)
                M_IDLE: if (start) begin
                    load_sel<=0; rd_blk<=16'd0; ld_rst<=1; rd_start<=1; mst<=M_PRIMEW; // load tile 0 -> buf0
                end
                M_PRIMEW: if (rd_done) begin comp_sel<=0; load_sel<=1; tile_idx<=0; mst<=M_RUN; end
                M_RUN: begin
                    comp_go<=1; cdone_l<=0; rdone_l<=0;            // compute current tile
                    if (tile_idx+1 < T) begin                     // load next tile into the other buffer
                        rd_blk<=tile_idx+1; ld_rst<=1; rd_start<=1; load_issued<=1;
                    end else load_issued<=0;
                    mst<=M_WAIT;
                end
                M_WAIT: begin
                    if (comp_done) cdone_l<=1;                     // latch the two pulses
                    if (rd_done)   rdone_l<=1;
                    if ((comp_done||cdone_l) && (!load_issued || rd_done || rdone_l)) begin
                        if (tile_idx==T-1) mst<=M_DONE;
                        else begin tile_idx<=tile_idx+4'd1; comp_sel<=~comp_sel; load_sel<=~load_sel; mst<=M_RUN; end
                    end
                end
                M_DONE: begin done<=1; mst<=M_IDLE; end
            endcase
        end
    end
endmodule
