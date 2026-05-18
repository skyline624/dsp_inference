// =============================================================================
// Module : uart_rx_8n1
// Recepteur UART 8 bits, sans parite, 1 bit de stop (8N1).
//   DIV = Fclk / debit  (27 MHz / 1 000 000 = 27 -> 1 Mbaud exact)
//
//   - 'rx'    : ligne serie entrante (au repos a l'etat haut).
//   - 'data'  : octet recu (valide quand 'valid' = 1).
//   - 'valid' : impulsion d'un cycle a la fin de reception d'un octet.
//
// Principe : on detecte le front descendant (bit de start), on attend le
// milieu du bit de start pour le confirmer, puis on echantillonne les 8 bits
// de donnee toutes les DIV periodes (LSB en premier), puis le bit de stop.
// =============================================================================

module uart_rx_8n1 #(
    parameter DIV = 27
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);

    // Synchronisation de la ligne rx (anti-metastabilite) : 2 bascules
    reg rx_s0 = 1'b1, rx_s1 = 1'b1;
    always @(posedge clk) begin
        rx_s0 <= rx;
        rx_s1 <= rx_s0;
    end

    localparam S_IDLE  = 2'd0,
               S_START = 2'd1,
               S_DATA  = 2'd2,
               S_STOP  = 2'd3;

    reg [1:0]  state  = S_IDLE;
    reg [15:0] cnt    = 16'd0;    // compteur de periode-bit
    reg [2:0]  bitidx = 3'd0;     // index du bit de donnee 0..7
    reg [7:0]  shft   = 8'd0;

    always @(posedge clk) begin
        if (rst) begin
            state  <= S_IDLE;
            valid  <= 1'b0;
            cnt    <= 16'd0;
            bitidx <= 3'd0;
        end else begin
            valid <= 1'b0;                       // impulsion par defaut
            case (state)

                S_IDLE: begin
                    if (rx_s1 == 1'b0) begin     // front descendant = start
                        state <= S_START;
                        cnt   <= 16'd0;
                    end
                end

                S_START: begin                  // attendre le milieu du start
                    if (cnt == (DIV/2)) begin
                        if (rx_s1 == 1'b0) begin // start confirme
                            state  <= S_DATA;
                            cnt    <= 16'd0;
                            bitidx <= 3'd0;
                        end else begin
                            state <= S_IDLE;     // faux depart (glitch)
                        end
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end

                S_DATA: begin                   // echantillonner 8 bits
                    if (cnt == DIV - 1) begin
                        cnt  <= 16'd0;
                        shft <= {rx_s1, shft[7:1]};   // LSB en premier
                        if (bitidx == 3'd7)
                            state <= S_STOP;
                        else
                            bitidx <= bitidx + 3'd1;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end

                S_STOP: begin                   // laisser passer le bit de stop
                    if (cnt == DIV - 1) begin
                        data  <= shft;
                        valid <= 1'b1;          // octet pret
                        state <= S_IDLE;
                        cnt   <= 16'd0;
                    end else begin
                        cnt <= cnt + 16'd1;
                    end
                end

            endcase
        end
    end

endmodule
