// =============================================================================
// Module : mac18
// Cible  : GW2AR-18 (Tang Nano 20K) - bloc DSP MULTALU18X18
//
// Description :
//   MAC (Multiply-Accumulate) signe 18x18 -> accumulateur 54 bits.
//   Realise : ACC <= (A * B)              si load = 1  (charge)
//             ACC <= ACC + (A * B)        si load = 0  (accumule)
//
// Implementation :
//   Instanciation directe d'une primitive MULTALU18X18 du GW2AR-18.
//   Mode 0  : ACC +/- (A*B) +/- C  (on force C = 0 et B_ADD_SUB = +)
//   Registres internes actives sur A, B et la sortie pour un MAC pipeline
//   (latence = 3 cycles entre la presentation de A,B et la sortie stable).
//
//   ATTENTION - 2 details de la primitive MULTALU18X18 (cf. prim_sim.v) :
//     1. Le signal ACCLOAD a une polarite INVERSEE par rapport a "load" :
//          acc_load = ACCLOAD ? DOUT : 0  ;  acc_out = acc_load + produit
//        => ACCLOAD = 1 ACCUMULE, ACCLOAD = 0 CHARGE. D'ou accload = ~load.
//     2. Le produit traverse 2 registres (AREG + PIPE_REG) mais ACCLOAD un
//        seul (ACCLOAD_REG0). Il faut donc ACCLOAD_REG1 = 1 pour realigner
//        ACCLOAD sur le produit a l'entree de l'ALU.
// =============================================================================

module mac18 (
    input  wire         clk,        // horloge
    input  wire         rst,        // reset synchrone actif haut
    input  wire         ce,         // clock enable
    input  wire         load,       // 1 = charge A*B, 0 = accumule
    input  wire signed [17:0] a,    // operande A
    input  wire signed [17:0] b,    // operande B
    output wire signed [53:0] result // ACC sur 54 bits
);

    // ACCLOAD (polarite inverse de "load", voir entete) :
    //   1 => l'accumulateur additionne le nouveau produit (accumule)
    //   0 => l'accumulateur est ecrase par le produit (charge)
    wire accload = ~load;

    // Entree C inutilisee dans ce MAC : maintenue a 0
    wire signed [53:0] c_zero = 54'sd0;

    // Cascade non utilisee (un seul bloc DSP)
    wire [54:0] casi_zero = 55'b0;
    wire [54:0] caso_nc;

    MULTALU18X18 u_mac (
        .DOUT    (result),
        .CASO    (caso_nc),
        .A       (a),
        .B       (b),
        .C       (c_zero),
        .D       (54'sd0),       // D inutilise en mode 0
        .CASI    (casi_zero),
        .ACCLOAD (accload),
        .ASIGN   (1'b1),         // A signe
        .BSIGN   (1'b1),         // B signe
        .DSIGN   (1'b0),
        .CLK     (clk),
        .CE      (ce),
        .RESET   (rst)
    );

    // Parametres du bloc DSP : activation des registres internes
    defparam u_mac.AREG             = 1'b1;   // registre sur A
    defparam u_mac.BREG             = 1'b1;   // registre sur B
    defparam u_mac.CREG             = 1'b0;   // C non utilise
    defparam u_mac.DREG             = 1'b0;
    defparam u_mac.ASIGN_REG        = 1'b0;
    defparam u_mac.BSIGN_REG        = 1'b0;
    defparam u_mac.DSIGN_REG        = 1'b0;
    defparam u_mac.ACCLOAD_REG0     = 1'b1;
    defparam u_mac.ACCLOAD_REG1     = 1'b1;   // realigne ACCLOAD sur le produit
    defparam u_mac.PIPE_REG         = 1'b1;   // registre pipeline interne
    defparam u_mac.OUT_REG          = 1'b1;   // registre de sortie
    defparam u_mac.B_ADD_SUB        = 1'b0;   // +(A*B)
    defparam u_mac.C_ADD_SUB        = 1'b0;   // +C
    defparam u_mac.MULTALU18X18_MODE = 0;     // ACC +/- (A*B) +/- C
    defparam u_mac.MULT_RESET_MODE  = "SYNC";

endmodule
