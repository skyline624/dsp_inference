# sim/ — Simulation de cluster FPGA (open source, fidèle au RTL réel)

Objectif : tester l'architecture **cluster de FPGA** (plusieurs Tang Nano 20K
reliés) **en simulation, avant tout achat de matériel**, en faisant tourner le
*vrai* Verilog du projet (`src/top.v` + opérateurs) — pas un modèle simplifié.

Moteur : **cocotb + Icarus Verilog**, dans le conteneur Docker `hdlc/sim:osvb`
(Open Source Verification Bundle). Aucun outil à installer côté Windows hormis
Docker Desktop (déjà présent).

## Comment lancer

```powershell
# 1 nœud (smoke / bring-up d'un FPGA)
./run.ps1

# 2 nœuds chaînés en UART (cluster)
./run.ps1 -N 2
```

Ou directement (dans le conteneur, depuis `sim/`) : `make` / `make N=2`.

## Ce qu'il y a ici

| Fichier | Rôle |
|---|---|
| `models/Gowin_rPLL.v` | Stub de simulation du PLL Gowin (passe l'horloge, `lock=1`). |
| `models/MULTALU18X18.v` | Modèle comportemental du DSP Gowin, **config unique** de `mac18.v` (latence 3 cycles, ACCLOAD inversé). |
| `models/sdram_chip.v` | Modèle comportemental de la SDRAM SDR 32 bits (commandes JEDEC, CAS=2, masque DQM). Fonctionnel, pas timing-accurate. |
| `node_top.v` | Un nœud = `top.v` + sa puce SDRAM. N'expose que `clk` + UART. |
| `cluster_top.v` | N nœuds en chaîne UART : `head_rx → node0 → … → node N-1 → tail_tx`. `N` paramétrable. |
| `test_cluster.py` | Testbench cocotb. |
| `run.ps1` | Wrapper Docker (monte le repo, copie les `.hex`, lance `make`). |

### Pourquoi des modèles de primitives ?
`top.v` instancie 3 primitives Gowin non simulables sans la lib vendeur :
`rPLL`, `MULTALU18X18`, et la SDRAM embarquée. On les remplace par des modèles
comportementaux. Tout le reste (mémoires `BSRAM` inférées, LUT `$readmemh`) est
du Verilog standard simulé tel quel.

### Note sur `src/top.v`
5 regs (`x_raddr_fm`, `w_raddr_fm`, `out_waddr_fm`, `out_wdata_fm`, `out_we_fm`)
étaient déclarés *après* leur usage. Gowin EDA le tolère ; Icarus/Verilator non.
Les déclarations ont été déplacées en amont (zone « MUX BSRAM ports ») —
modification **strictement neutre** pour la synthèse, qui rend le RTL portable.

## Feuille de route

- [x] **M1 — Bring-up.** Le vrai RTL s'élabore, boote (reset levé après lock PLL
  + 32768 cycles), `tail_tx` idle sans contention X, chemin UART vérifié.
  Cluster N=1 et N=2 OK. (`test_cluster.py`)
- [x] **M2 — Fondation validée bit-exact.** La simulation reproduit le projet :
  - `test_silu_node.py` : commande réelle `SS` (SiLU), 4 cas **64/64 int8
    bit-exact** vs la référence du projet (même LUT que le RTL). Valide UART +
    FSM + opérateur + timing 1 Mbaud.
  - `test_sdram_node.py` : `LL`+`CC` (SDRAM round-trip, 64/64 octets) et `FQ`
    (matmul depuis SDRAM, 8/8 int8 bit-exact vs `requantize`). Valide les deux
    modèles écrits à la main : `sdram_chip.v` et `MULTALU18X18.v` (DSP).
- [x] **M3 — Partition tensor-parallel (row-parallel).** `test_tp_matmul.py` +
  `cluster_tp.v` (topologie étoile, 2 nœuds à UART indépendants). Un matmul
  `y = W @ x` est découpé **par lignes de sortie** : chaque nœud stocke **sa
  tranche de W dans sa propre SDRAM** et calcule sa tranche en parallèle ; le
  coordinateur broadcast `x` et gather les tranches de `y`. Résultat **bit-exact
  vs la référence du projet** (32+32 lignes), recombiné à 0,40 % du matmul float.
  → valide le scaling **mémoire (k SDRAM en //) + calcul (k DSP en //)**.
- [x] **M3b — FFN complet tensor-parallel.** `test_tp_ffn.py` : bloc FFN entier
  réparti sur 2 nœuds (Megatron-style) — W1/W3 **row-parallel**, silu+multiply
  local, W2 **column-parallel + all-reduce**, résidu. Matmuls et silu **bit-exact**
  vs référence projet ; sortie finale à **5,8 % du float** (quantification int8).
  Poids préchargés en backdoor SDRAM. → tout le pattern inter-nœud
  (broadcast + reduce) est validé.
- [x] **M3c — Bloc attention head-parallel.** `test_tp_attn.py` : Q/K/V
  head-parallel (chaque nœud ses têtes, GQA), cœur MM réel sur Q/K/V rassemblés,
  Wo row-parallel. Projections + Wo **bit-exact** ; MM (T=1) bit-exact =
  passthrough GQA de V ; bloc complet à **7,5 % du float**. (Portée : pos=0/T=1,
  rope identité, pas de KV cache — softmax multi-positions au M4.)
  Note : la commande `MM` est **monolithique** (boucle les 8 têtes) → ici Q/K/V
  rassemblés et MM exécuté centralement. Pour une attention **100 % head-locale**
  (sans gather), `MM` doit accepter une plage de têtes (petite évolution RTL).
- [x] **M4a — Couche complète tensor-parallel, multi-positions.**
  `test_tp_layer.py` : attention (head-parallel) + FFN (tensor-parallel) chaînées
  en une couche entière, exécutée sur 3 positions avec **KV cache multi-positions**
  et **vrai softmax MM (T>1)**. Matmuls distribués (poids sur 2 SDRAM), MM réel,
  coordinateur = colle d'activations (rope, multiply, all-reduce, résidu, cache).
  Sortie à **10–14 % du float** (accumulation quantification int8, normale).
  → **toute la mécanique du cluster est validée en simulation.**
- [x] **M4b — Vrai modèle, passe avant complète.** `test_tp_model.py` : le vrai
  `stories260K` (5 couches, hidden=172, vocab=512) fait une passe avant distribuée
  sur 2 nœuds — matmuls tensor-parallel **chunkés** (round-robin sur les nœuds),
  MM réel, lm_head distribué, préchargement couche par couche. **argmax = 403
  ('Once'), identique à `infer_v4sim`.** Preuve end-to-end. (~19 min de sim.)
  Portée : 1 passe (pos=0) ; générer toute la phrase = même boucle, ~h de sim.
- [ ] **Évolution RTL** : `MM` paramétrable par plage de têtes (attention 100 %
  head-locale sans gather) ; rope via commande `RR` (ici fait côté coordinateur).
- [x] **Lien inter-FPGA source-synchrone** (`link/`, tout PASS). Lancement :
  `make` dans `sim/link/` (override `TOPLEVEL=`/`MODULE=`).
  - **CDC** : `async_fifo.v` (dual-clock Gray+2FF) + `ss_link.v` + `test_ss_link.py`
    — 256 octets, 2 horloges désaccordées (27,0→32,3 MHz), ordre/0 perte. ~27 Mo/s
    (~270× UART), latence ~µs. Largeur paramétrable (validé `DW=16`).
  - **Tâche 1 — intégration cluster** : `link_send`/`link_recv` + `cluster_link2.v`
    + `test_cluster_link.py` — échange d'activations `x[64]` **full-duplex** entre
    2 nœuds à horloges indépendantes, bit-exact. Transport node↔node (remplace UART).
  - **Tâche 2 — durcissement** : `almost_full` (`async_fifo` param `AFMARGIN`) +
    `bp_compare.v`/`test_bp.py` — sous un délai de backpressure de 3 cycles
    (aller-retour inter-puce), `full` simple **perd 792 octets**, `almost_full`
    **0**. Skew données/horloge = contrainte board (IODELAY/phase).
  - **Tâche 4 — mini-GG autonome (concept)** : `mini_gg.v` + `cluster_minigg.v` +
    `test_minigg.py` — un nœud enchaîne **RECV → calcul 2 étages → SEND** via une
    seule FSM pilotée par le lien, **zéro commande PC**, bit-exact.
  - **mini-GG RÉEL, incrément 1** : `tp_mm_node.v` + `tp_mm_single.v` +
    `test_tp_mm.py` — nœud autonome qui calcule sa tranche `y=W@x[32×64]` avec le
    **vrai DSP `mac18`/`MULTALU18X18`** + requantification `fq_ref`, piloté par le
    lien, **zéro PC**, **bit-exact**. Poids en mémoire locale (ne traversent jamais
    le lien).
  - **mini-GG RÉEL, incrément 2 — ring all-gather scalable** : `ring_node.v` +
    `ring_cluster.v` + `test_ring.py` — N nœuds en **anneau** (1 lien entrant + 1
    sortant par nœud, **broches constantes quel que soit N**) ; après N−1 tours
    autonomes, **chaque nœud détient le vecteur complet**. Validé **N=2 ET N=4**.
    → l'architecture **scale à >2 nœuds** (paramètre `N`). Build :
    `NODES=4 make TOPLEVEL=ring_cluster MODULE=test_ring PARAMS=-Pring_cluster.N=4`.
  - **mini-GG RÉEL, incrément 3.1 — matmul tensor-parallel autonome sur N nœuds** :
    `tp_node.v` (matmul vrai DSP + ring all-gather fusionnés) + `tp_ring_cluster.v`
    + `test_tp_ring.py`. x broadcast → chaque nœud calcule sa tranche → ring
    all-gather → **chaque nœud a la sortie complète**, **zéro PC**, **bit-exact**
    vs fq_ref. Validé **NN=2 et NN=4**. Bonus : NN=4 ~2× plus rapide que NN=2
    (moins de lignes/nœud = parallélisme de calcul). Build :
    `NODES=4 ROWS_PER=16 make TOPLEVEL=tp_ring_cluster MODULE=test_tp_ring PARAMS='-Ptp_ring_cluster.NN=4 -Ptp_ring_cluster.NR=16'`.
  - **mini-GG RÉEL, incrément 3.2 — column-parallel + ring ALL-REDUCE** :
    `tp_ar_node.v` + `tp_ar_cluster.v` + `test_tp_ar.py`. W split par colonnes,
    chaque nœud calcule un partiel (vrai DSP), **ring all-reduce** (= all-gather
    des partiels int32 via `ring_node` + somme locale + requant) → chaque nœud a
    `y=W@x` complet, **zéro PC**, **bit-exact** vs fq_ref. Validé **NN=2 et NN=4**.
    → les **2 primitives d'échange** sont autonomes : all-gather (3.1, W1/W3/Q/K/V)
    + all-reduce (3.2, W2/Wo). Build :
    `NODES=4 KS=16 make TOPLEVEL=tp_ar_cluster MODULE=test_tp_ar PARAMS='-Ptp_ar_cluster.NN=4 -Ptp_ar_cluster.KS=16'`.
  - **Brique FFN — opérateur réel chaîné** : `tp_silu_node.v` + `test_tp_silu.py`
    — chaîne autonome **matmul (mac18) → vrai `silu_op`** (interface mémoire +
    LUT), gestion de *shift* correcte, **bit-exact** vs fq_ref+silu_ref. Prouve
    que les vrais opérateurs élémentaires s'intègrent dans la FSM autonome (même
    motif pour `rmsnorm_op`). Build : `cp ../../src/silu_lut.hex . ; make
    TOPLEVEL=tp_silu_node MODULE=test_tp_silu`.
  - **Séquenceur mini-GG sur nœud routable (inc.1)** : `seq.v` + `seq_node.v` +
    `test_seq.py` (dans `sim/`). Un petit FSM **on-chip** émet une commande `SS`
    au **nœud briques NON MODIFIÉ** via sa propre interface UART, capture le
    résultat → **bit-exact vs silu_ref, zéro PC**. Prouve que le séquenceur pilote
    le datapath briques validé (joue le rôle du PC, mais en RTL). Build :
    `cp ../src/*.hex . ; make TOPLEVEL=seq_node MODULE=test_seq`.
  - **Séquenceur mini-GG — chaînage + handoff (inc.2)** : `seq.v` paramétré
    `N_OPS` + `test_seq2.py`. Le FSM garde l'activation courante `curx` on-chip ;
    la sortie d'une commande devient l'**entrée de la suivante** sans hôte. Validé
    SS→SS **bit-exact vs double-silu, zéro PC**. Build :
    `make TOPLEVEL=seq_node MODULE=test_seq2 PARAMS=-Pseq_node.N_OPS=2`.
    → les 2 mécanismes nouveaux (séquenceur pilote les briques + handoff) sont prouvés.
  - **ALU glue FFN (multiply + résidu)** : `vec_alu.v` + `test_vec_alu.py` (dans
    `link/`). `MUL` = `a[i]*b[i]` sur le **vrai DSP** (`mac18`, mode charge) ;
    `ADD` = résidu aligné ; requant entier (arrondi cohérent briques). Les deux
    **bit-exact**. C'est la dernière brique nouvelle (la colle que faisait le PC :
    silu·h3 et x+out). Build : `make TOPLEVEL=vec_alu MODULE=test_vec_alu`.
  - **✅ FFN AUTONOME COMPLET (intégration finale)** : `ffn_seq.v` + `ffn_top.v`
    + `test_ffn_seq.py` (dans `sim/`). Le séquenceur on-chip exécute **tout le
    FFN tout seul, zéro PC** : `FN(rmsnorm)→FQ(W1)→FQ(W3)→SS(silu)→vec_alu MUL
    (silu·h3)→FQ(W2)→vec_alu ADD(résidu)`, en pilotant le nœud briques NON
    MODIFIÉ (poids SDRAM backdoor) + `vec_alu`, avec handoff et threading des
    shifts. Sortie à **4,6 % du float**. C'est le « GG » réalisé — mais comme un
    petit séquenceur pilotant le datapath routable, pas un monolithe. Build :
    `cp ../src/*.hex . ; make TOPLEVEL=ffn_top MODULE=test_ffn_seq`.
  - **✅ FFN TENSOR-PARALLEL AUTONOME (2 nœuds)** : `ffn_tp_seq.v` + `ffn_tp_top.v`
    + `test_ffn_tp.py`. Le séquenceur on-chip (hôte UART muxé par `tgt`) répartit
    le FFN sur 2 nœuds — rmsnorm broadcast, W1/W3 row-parallel, silu·h3 par nœud,
    W2 column-parallel, **all-reduce + résidu via vec_alu** — **zéro PC**, poids
    répartis sur 2 SDRAM. Sortie à **6,4 % du float**. Build :
    `cp ../src/*.hex . ; make TOPLEVEL=ffn_tp_top MODULE=test_ffn_tp`.
  - **✅ ATTENTION TENSOR-PARALLEL AUTONOME (2 nœuds, pos=0/T=1)** : `attn_tp_seq.v`
    + `attn_tp_top.v` + `test_attn_tp.py`. Le séquenceur on-chip répartit
    l'attention : rmsnorm broadcast, Wq/Wk/Wv **head-parallel**, **gather** (Q/K/V
    via `vec_alu` ADD + padding = concat ré-aligné), **vrai MM**, Wo row-parallel,
    gather, résidu — **zéro PC**, poids sur 2 SDRAM. Sortie à **7,6 % du float**.
    Build : `cp ../src/*.hex . ; make TOPLEVEL=attn_tp_top MODULE=test_attn_tp`.
  - **✅ COUCHE TRANSFORMER TENSOR-PARALLEL AUTONOME (2 nœuds)** : `layer_tp_top.v`
    + `test_layer_tp.py`. Un contrôleur on-chip enchaîne `attn_tp_seq` → (handoff
    x1) → `ffn_tp_seq`, chaque bloc réparti sur 2 nœuds (mux des liens par phase).
    Attention en SDRAM 0x10xxxx, FFN en 0x11xxxx. **Zéro PC**, sortie à **12,8 %
    du float**. Réutilise les 2 séquenceurs validés tels quels (pas de fusion).
    Build : `cp ../src/*.hex . ; make TOPLEVEL=layer_tp_top MODULE=test_layer_tp`.
  - **Une couche transformer entière s'exécute en autonomie répartie.**
  - **✅ PASSE AVANT MULTI-COUCHES AUTONOME (2 nœuds)** : `model_tp_top.v` +
    `test_model_tp.py`. Un contrôleur on-chip **boucle** la couche TP (attention →
    FFN) sur `NL` couches : `base = l×0x20000` adresse les poids de la couche l
    (répartis sur 2 SDRAM), shifts par couche en bus packés, `x` rebouclé. Validé
    **NL=2** à **12,6 % du float** (l'erreur ne s'emballe pas grâce aux résidus :
    12,6 % à 2 couches ≈ 12,8 % à 1). Paramétrable `NL` (5 = modèle entier, juste
    plus de poids/temps de sim). Les séquenceurs ont gagné une entrée `base`
    (offset d'adresse) pour cette boucle. Build :
    `cp ../src/*.hex . ; make TOPLEVEL=model_tp_top MODULE=test_model_tp`.
    → Reste pour la génération : lm_head/embedding + sampling + (KV cache pos>0).
    Génération multi-tokens = matériel réel (sim impraticable, ~5 h ; correction M4b).
  - **Toutes les briques du FFN sont prouvées en autonomie** : matmul (mac18),
    all-gather (3.1), all-reduce (3.2), opérateur élémentaire réel (silu_op),
    **et le séquenceur qui pilote les briques (inc.1)**.
    **Assemblage FFN complet** = chaîner rmsnorm_op + W1/W3(local) + silu×h3 +
    W2(all-reduce) + résidu, avec la comptabilité de *shift* de bout en bout —
    c'est, par tranche, le datapath du `GG`. Deux voies : (a) une grosse FSM de
    nœud qui chaîne tous les blocs validés ; (b) **réutiliser le datapath briques
    de `top.v`** via un séquenceur de commandes on-chip (nécessite la tâche 3 =
    `top.v` sans GG). La voie (b) évite de ré-implémenter ~2000 lignes déjà
    validées → c'est la recommandation pour le FFN bit-exact complet.

### Tâche 3 (P&R d'un nœud) — ✅ RÉSOLUE
Découpe chirurgicale du `GG` : gardes `\`ifdef NODE_ONLY` dans `top.v` (op_gg=0,
dispatch retiré, **bloc d'états GG `S_M2_G`→`S_GG_TX_V` encadré `\`ifndef`**) +
`src/top_node_wrap.v` (`\`define NODE_ONLY` puis `\`include "top.v"` — l'include
garantit la macro en portée, Gowin ne la partage pas entre fichiers) +
`build_node.tcl`. Lancer : `gw_sh build_node.tcl`.
**Résultat (GW2AR-18) : le nœud ROUTE.** 8 627 LUT (**45 %**, vs 73 % en échec
avec GG), 0 net non routé, BSRAM 14 %, DSP 35 %, **Fmax ~55 MHz** (~2× la marge
sur 27 MHz). → un nœud-worker tient confortablement sur un Tang Nano 20K, avec
de la marge pour le séquenceur mini-GG et la croissance.
Le build complet (`build.tcl`, top.v sans NODE_ONLY) reste inchangé.

### Note scaling (importante pour des modèles plus grands)
L'erreur de quantification int8+pow2 **s'accumule** le long de la chaîne (~14 %
sur 1 couche ; le projet note ~30–40 % sur 35 matmuls). Pour des **modèles plus
grands** (chaînes plus longues), prévoir le passage à un **scale Q-format 16 bits**
(évoqué dans `host/v4_quant.py`) — ajoute un petit multiplieur DSP mais réduit la
perte. C'est une évolution numérique, indépendante du tensor-parallel.
- [ ] **Visu** (optionnel) : dump VCD + GTKWave/Surfer côté hôte ; schéma via
  netlistsvg (node sur Windows déjà présent).

## P&R / tenue LUT (volet matériel, en parallèle de la sim)
`pnr/synth.ys` + `pnr/blackbox_prims.v` : flot **yosys** (`synth_gowin`) pour une
**estimation** LUT/FF/BRAM d'un nœud (PLL et DSP blackboxés car ce sont des blocs
durs). À lancer dans une image avec yosys (`hdlc/impl`).
**Important** : c'est une estimation relative ; le chiffre officiel GW2AR-18 (les
91% d'origine) vient de **Gowin EDA** (`gw_sh build.tcl`). Un nœud tensor-parallel
exécute **le même RTL** qu'aujourd'hui (mêmes opérateurs sur une tranche) → sa
LUT reste ~constante ; le tensor-parallel scale la mémoire et le calcul, pas la
LUT par nœud.

### Découverte de fidélité (à retenir pour le cluster)
Le contrôleur SDRAM ne lance son init power-on (200 µs / 5400 cycles) qu'**après**
`rst_n` (levé à 32768 cycles). Il y a donc une fenêtre où le FSM est vivant mais
la SDRAM n'est pas prête (`sd_busy=1`). Sur la vraie carte des ms s'écoulent
avant la 1ʳᵉ commande ; en sim il faut **attendre `sd_busy=0`** avant toute
commande SDRAM (cf. `boot()` dans `test_sdram_node.py`). Chaque nœud du cluster
devra respecter ça.

## Rappel : tenue/routage par FPGA (séparé de la simulation)
La simulation valide le *fonctionnement* du cluster. Pour savoir si une couche
**tient et route** sur tel FPGA candidat (le mur rencontré sur le GW2AR-18),
c'est une **synthèse + P&R** par numéro de pièce (Gowin EDA, ou yosys+nextpnr
open source) — sans matériel. À traiter en parallèle.
