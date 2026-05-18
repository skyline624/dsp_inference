// =============================================================================
// Module : uart_tx_8n1
// Emetteur UART 8 bits, sans parite, 1 bit de stop (8N1).
//   DIV = Fclk / debit  (27 MHz / 1 000 000 bauds = 27 pile -> 1 Mbaud exact)
//
//   - 'send' : impulsion d'un cycle qui lance l'emission de 'data'.
//   - 'busy' : 1 pendant toute l'emission ; ignorer 'send' tant que busy = 1.
//   - 'tx'   : ligne serie, au repos a l'etat haut.
// =============================================================================

module uart_tx_8n1 #(
    parameter DIV = 27
) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       send,
    output wire       tx,
    output reg        busy
);

    reg [9:0]  shifter = 10'h3FF;   // {stop, data[7:0], start}
    reg [3:0]  bitcnt  = 4'd0;      // start + 8 data + stop = 10 bits
    reg [15:0] divcnt  = 16'd0;

    assign tx = shifter[0];         // repos : shifter plein de 1 -> tx = 1

    always @(posedge clk) begin
        if (rst) begin
            shifter <= 10'h3FF;
            busy    <= 1'b0;
            bitcnt  <= 4'd0;
            divcnt  <= 16'd0;
        end else if (!busy) begin
            if (send) begin
                shifter <= {1'b1, data, 1'b0};   // stop / data (LSB first) / start
                busy    <= 1'b1;
                bitcnt  <= 4'd0;
                divcnt  <= 16'd0;
            end
        end else begin
            if (divcnt == DIV - 1) begin
                divcnt  <= 16'd0;
                shifter <= {1'b1, shifter[9:1]};  // decale, injecte des 1
                bitcnt  <= bitcnt + 4'd1;
                if (bitcnt == 4'd9) busy <= 1'b0; // 10e bit (stop) emis
            end else begin
                divcnt <= divcnt + 16'd1;
            end
        end
    end

endmodule
