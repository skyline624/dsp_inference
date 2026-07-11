# Plan de validation cluster ECP5 — recap & bring-up pas-à-pas

> **Source de vérité de la phase hardware.** Une session fraîche qui reprend ici ne repart
> pas de zéro : ce doc récapitule d'où on vient, les décisions prises, le matériel choisi,
> et le plan d'attaque étape par étape avec les pièges connus. Complémentaire aux fiches
> mémoire (`cluster-hardware-spec.md`, `session6-routing-status.md`,
> `cluster-fpga-direction.md`) — ici c'est le *chemin*, là-bas c'est l'*état*.

---

## 1. D'où on vient (récap en 1 page)

Le projet `dsp_inference` fait de l'inférence transformer sur FPGA, depuis le début
monocarte (Tang Nano 20K / **Gowin GW2AR-18C**, 20 736 LUT).

**Acquis prouvés (ne PAS refaire) :**
- Datapath complet : matmul `mac18`, rmsnorm, silu, rope, attention causale, softmax,
  argmax, quantification **int8 + shift puissance-de-2**.
- **Séquenceur de génération LUT-lean** (`gen_seq` dans `sim/link/`) : register-file
  `vfile` byte-wide (pas de mux 512-bit), prouvé **17/17 tokens bit-exact** en sim
  (texte oracle « Once upon a time… »). Validé étapes A→F.
- **Lignée cluster tensor-parallel** (`sim/link/` : `ss_link`, `tp_node`, `ring_*`,
  `vec_alu2`, `ffn_tp_seq2`/`attn_tp_seq` NN-paramétrés) : all-gather + all-reduce
  **validés en sim** (NN=2 et NN=4).
- **Boot SD** + **streaming par couche** (modèle > SDRAM) — validés sim **et silicium**.
- **Mono zéro-PC sur silicium** : le Tang Nano charge le modèle depuis la SD et génère
  du texte tout seul. Objectif de base atteint.

**Le mur (Session 6b) :** le séquenceur de génération `gen_seq` + le nœud co-localisés sur
**une seule puce GW2AR-18C ne ROUTENT PAS** (789 nets non routés, 92% CLS — congestion-bound,
pas capacity-bound). Les deux leviers (cos/sin→ROM BSRAM validé 17/17 ; fusion vfile → a nui,
annulé) sont épuisés. **Verdict : l'emballage tout-sur-une-puce ne tient pas → bascule cluster.**

**La décision :** plutôt que de forcer sur une puce, **passer à un cluster de FPGA** — ce
pour quoi la lignée tensor-parallel a été bâtie. Le cluster n'est pas une nécessité de
capacité (n'importe quelle taille de modèle tient en flash + N nœuds) ; c'est un **levier
de vitesse** (paralléliser la bande passante mémoire + le compute).

---

## 2. L'architecture cible (principes dérivés avec l'utilisateur)

### 2.1 Tensor-parallel head-parallel — le KV reste LOCAL
L'attention est head-parallel : la tête *h* n'a besoin que de son propre K/V. Si le nœud N
calcule les têtes {h0,h1…}, il détient leur K/V toutes positions. **Le KV shard suit la
frontière des têtes → reste 100 % local, aucun accès mémoire distant.**
→ Le lien inter-nœud ne porte que des **activations** (`O(D)` par token, **constant** — ne
grandit PAS avec le contexte). Le trafic qui explose (KV) ne quitte jamais son nœud.
→ `all-reduce` (N-1 sauts en anneau) à chaque couche = la seule vraie latence inter-nœud.

### 2.2 Les 3 mémoires (par motif d'accès — la clé du design)
| # | Contenu | Motif | Techno | Quantif |
|---|---|---|---|---|
| **1. Poids** | shard du modèle | séquentiel · **lecture seule** · énorme | **flash** (NV) | int8/pow2 |
| **2. Travail** | activations + chunk + double-buffer | aléatoire · R/W · minuscule · ultra-rapide | **BSRAM on-chip** | pleine précision |
| **3. KV-cache** | K/V (têtes locales) | R/W · re-lu/token · grandit avec contexte | **SDRAM/DRAM RW** | int8/int4 |

**Pourquoi pas de cache :** les poids n'ont aucune réutilisation intra-token (lus 1×/token)
→ un cache ne capture rien. Seul le **KV** est réutilisé. « Read-only » des poids → ouvre
le **flash** (non-volatile, pas de boot-load, moins cher/Go) plutôt que la DRAM.

### 2.3 La contrainte dominante : memory-bandwidth-bound
Lire ~30 Go/token (30B dense int8) ou ~3-6 Go/token (MoE A3B).
`temps/token ≈ octets/token ÷ (N × débit_mémoire/nœud)`. ~15-120 nœuds (DDR/nœud) pour
~1 tok/s dense. Le **KV** grandit avec le contexte : 30B GQA ≈ 200 Mo (ctx 2K) → 1 Go (ctx 8K),
shardé /N. *(À 1M tokens de contexte : ~30 Mo de texte mais ~250 Go de KV — c'est le contexte,
pas la taille du modèle, qui pilote la facture KV.)*

### 2.4 Interconnexion : LATENCE >> bande passante
Les activations sont minuscules → le débit est quasi gratuit ; l'all-reduce est une **barrière
à chaque couche sur le chemin critique** (× couches × tokens) → **la latence domine**.
- Le **SERDES n'est PAS un outil de faible latence** (latence fixe ~50-100 ns : ser/des + CDR
  + buffer élastique). Le `ss_link` parallèle a une latence par-saut *plus faible* (~10-20 ns)
  mais bouffe ~10 pins/lien.
- Le **vrai driver de latence = le nombre de sauts (topologie)** : anneau = N-1 sauts
  (catastrophe au scale) ; arbre/tore = ~log/√N sauts → **~10× mieux**.
- **Le SERDES aide la latence indirectement** : 2 pins/lien → beaucoup de liens/nœud →
  topologie basse-diamètre → moins de sauts. Choisir la puce pour **assez de canaux SERDES**
  (topologie riche) + mode low-latency, **PAS pour le débit SERDES**.
- **À 2-4 nœuds : latence non-problème (2-3 sauts), le `ss_link` parallèle suffit.**

### 2.5 Quantif
**int8 + pow2 partout** — poids ET KV-cache (gratuit : DSP déjà basse précision, erreur bornée
par les résidus). **Dette connue : Q16** pour les gros modèles (×2 DSP/LUT), à passer au scale.

---

## 3. Le matériel choisi

### 3.1 Plateforme de validation : **2× Colorlight i9 v7.2** (commandées)
- FPGA **`LFE5U-45F-6BG381C`** (ECP5-45, 44K LUT, 72 DSP, ~1,9 Mb BRAM, **PAS de SERDES** —
  le « U » pas « UM » ; sans importance pour valider).
- **SDRAM 8 Mo SDR** (M12L64322A, 512K×32×4 banks) — *même type que le GW2AR → contrôleur
  se porte presque tel quel.*
- **SPI Flash 8 Mo** (W25Q64) → Mémoire 1 (poids).
- **2× PHY GbE** (Broadcom B50612D) + **Ext-Board : DAPLink (JTAG+UART) + 6× PMOD (100+ IO) + USB-C.**
- **Toolchain open** : yosys + nextpnr-ecp5 + prjtrellis + openFPGALoader. Repo carte :
  `github.com/wuxx/Colorlight-FPGA-Projects` (contraintes `.lpf` + exemples).
- ⚠ **NE PAS confondre** avec l'**i9+ v6.1** = Xilinx Artix `XC7A50T` (Vivado, + cher au
  scale) — écarté au profit de Lattice.

**Ce que les 2 nœuds valident :** lien physique inter-cartes, tensor-parallel sur silicium
(NN=2), streaming flash→SDRAM→compute, portage Gowin→ECP5, génération d'un petit modèle.
**Ce qu'ils NE valident PAS (phase scale) :** latence topologie à N nœuds, SERDES FPGA, Q16.

### 3.2 Puce de scale (décision ULTÉRIEURE, pas maintenant)
- **Lattice ECP5-85** (`LFE5UM-85F-8BG381I`) ~78-100 € Mouser / **~35 € AliExpress** (puce
  nue, ⚠ vérifier IDCODE JTAG — risque de remarquage) — 84K LUT, 156 DSP, 3,7 Mb, 4× SERDES.
- **Microchip PolarFire MPF300** ~425 € — premium (21 Mb RAM, 8×12,7G SERDES, hard DDR4),
  **écarté** : trop cher × N, sur-délivre là où on n'en a pas besoin (débit SERDES gaspillé,
  compute excédentaire).
- **Lattice CertusPro-NX-100** ~65 € — meilleur rapport capacité/prix au scale (+RAM, SERDES
  10G, DDR durci ; toolchain Radiant = gratuit + mature, **pas un frein**).
- Gowin GW5A/GW5AT : écarté (0 stock Mouser, MOQ 90-5040). Efinix : absent Mouser.

---

## 4. Prérequis outils (déjà installés — vérifier au démarrage)

- **MCP Mouser** (`mouser-search`) : recherche pièces/prix/stock. ✅ installé (clé read-only).
- **MCP KiCad** (`kicad`, ~170 outils) : conception PCB, **validé sur KiCAD 10** (backend
  swig, 222 libs). Pour la phase PCB custom (plus tard). ⚠ pas de lib Gowin standard →
  symbole custom à créer (mais Lattice ECP5 : vérifier si lib existe).
- **Toolchain ECP5 open** (à installer sur le poste) : `yosys`, `nextpnr-ecp5`, `prjtrellis`,
  `openFPGALoader`. (Sous Windows : oss-cad-suite ou WSL.)
- **Docker** `hdlc/sim:osvb` pour les sims cocotb/Icarus (⚠ **JAMAIS** `docker kill
  $(docker ps -q)` — ça tue le serveur anythingllm de l'utilisateur. Toujours
  `docker run --rm --name gensim` + `docker kill gensim`).
- **gw_sh** (`C:\Gowin\Gowin_V1.9.10_x64\IDE\bin\gw_sh.exe`) — encore utile pour le code
  Gowin existant, mais le portage ECP5 se fait sur toolchain open.

---

## 5. Plan de bring-up — 6 étapes (chaque étape = 1 gate VERT avant la suivante)

Inspiré de la méthode « une étape à la fois / une session par étape » qui a réussi jusqu'ici.
Tolérances de référence : `err/ma` 0.35 FFN, 0.40 couche, 0.60 modèle ; égalité stricte
quand bit-exact. **Garde-fou G1 : « ça route ≠ ça marche » — valider en sim, pas juste P&R.**

### Étape 1 — Hello-ECP5 (l'environnement Lattice marche)
**Objectif :** toolchain open (yosys/nextpnr/openFPGALoader) qui synthétise, route et flashe
un blink LED sur une i9. On apprend le vendor.
- Installer oss-cad-suite (ou WSL + yosys/nextpnr-ecp5/prjtrellis + openFPGALoader).
- Récupérer les contraintes `.lpf` de la i9 dans `wuxx/Colorlight-FPGA-Projects`.
- Blink LED → `openFPGALoader -b colorlight-i9 blink.bit` (vérifier le board name exact).
**Gate 1 :** LED clignote sur la carte. Toolchain opérationnelle.
**Pièges connus :**
- ⚠ **DAPLink / JTAG** : la i9 se programme via l'Ext-Board DAPLink, pas un FTDI direct.
  Vérifier le câblage (2×15 header → module) et que l'Ext-Board est bien montu.
- ⚠ **`.lpf` pinout** : reprendre celui du repo wuxx, ne pas deviner les pins (BGA381).
- ⚠ **openFPGALoader board id** : trouver le nom exact pour la i9 (`--list-boards`).
- ⚠ **Alim** : la i9 s'alimente par l'Ext-Board (USB-C), pas par le module seul.

### Étape 2 — Port SDRAM (Mémoire 3 + cache de couche)
**Objectif :** le contrôleur SDR de la M12L64322A marche sur l'ECP5 — lecture/écriture
bit-exacte. C'est la mémoire la plus proche du GW2AR (SDR, 32-bit).
- Porter `src/sdram.v` (contrôleur SDR existant) vers l'ECP5 — **devrait être quasi tel quel**
  (même type de SDRAM que le GW2AR).
- Adapter le pinout SDRAM (`.lpf` du repo wuxx).
- Test : écrire 8 Mo, relire, comparer (pattern + bruit).
**Gate 2 :** SDRAM byte-exact en sim ET sur silicium.
**Pièges connus :**
- ⚠ **Timing SDRAM** : la i9 a une horloge FPGA ≠ du GW2AR. Recalculer les timings (CAS
  latency, refresh) selon la fréquence PLL de l'ECP5. Le GW2AR avait `SD_HALF=135` etc.
  → re-dériver pour l'ECP5.
- ⚠ **Refresh** : sur le vrai HW, interleaver le refresh (la sim ne décaie pas).
- ⚠ **Largeur 32-bit** : la SDRAM est 32-bit (512K×32×4) → le contrôleur gère-t-il 32-bit
  ou faut-il l'adapter (le GW2AR utilisait déjà 32-bit, vérifier).

### Étape 3 — Port du nœud (datapath, mono-carte)
**Objectif :** le nœud `top.v` (NODE_ONLY) calcule juste sur une i9 — un matmul bit-exact.
C'est la grosse étape de portage vendor.
- **Primitives à re-faire** (vendor Gowin → Lattice ECP5) :
  - `gowin_rpll` → **PLL Lattice** (ECP5 a des PLL hard).
  - `mac18` / `MULTALU18X18` → **DSP ECP5** (`MULT18X18D` / mode multiply-accumulate).
  - `gowin_dsp` → DSP ECP5 (pipeline/registres différents).
  - IO/pads → pinout ECP5 (`.lpf`).
- **Ce qui se porte tel quel** : `rmsnorm_op`, `silu_op`, `rope_op`, `softmax_op`,
  `attention_head_op`, `vec_alu2`, le cœur FSM de `gen_seq`/`ffn_tp_seq2`, la quantif
  int8+pow2, le tensor-parallel (logique pure).
- Test : un matmul W@x bit-exact vs la référence Python.
**Gate 3 :** un matmul juste sur une i9 (sim + silicium).
**Pièges connus :**
- ⚠ **DSP ECP5 ≠ mac18 Gowin** : pipeline depth, sign-extension, accumulation différentes.
  Repartir de l'IP Lattice ou instancier `MULT18X18D` manuellement. Bug instructif Gowin
  (DSP ré-accumulait pendant le drain) → **mettre les operands à 0 pendant l'attente**.
- ⚠ **LUT-RAM `vfile`** : l'ECP5 infère aussi du LUT-RAM (distributed RAM) — vérifier que
  `vfile` reste distribué (pas forcé en BRAM). `(* ram_style="distributed" *)` si besoin.
- ⚠ **BRAM vs BSRAM Gowin** : les attributs `(* syn_ramstyle = "block_ram" *)` (Gowin) →
  ECP5 utilise `(* ram_style="block" *)`. Renommer les attributs (KV-cache, logits, ROM cos/sin).
- ⚠ **`$readmemh` chemins relatifs** (piège connu du projet) : les `.hex` LUT (rsqrt/silu/
  exp/inv/freq_cis) doivent être trouvés par nextpnr → utiliser un chemin qui marche, ou
  `define` un prefix.
- ⚠ **PLL frequency** : le GW2AR tournait à ~27 MHz (crystal) / Fmax ~55. L'ECP5 peut monter
  plus haut — mais garder la même fréquence que la sim pour la 1ère validation.

### Étape 4 — Lien 2-cartes (PMOD parallèle = ss_link)
**Objectif :** les 2 i9 echangent des octets bit-exact par PMOD — réutilise le concept
`ss_link` (source-synchrone sur GPIO + `async_fifo` CDC).
- Porter `sim/link/ss_link.v` + `link_send`/`link_recv` + `async_fifo` (logique pure,
  se porte tel quel).
- Câbler les PMOD des 2 Ext-Board (8 data + 1 frame + GND, ou plus large).
- Test : un octet envoyé d'une carte → reçu identique sur l'autre, dans les deux sens.
**Gate 4 :** round-trip octet bit-exact entre les 2 cartes physiques.
**Pièges connus :**
- ⚠ **Skew de board** (l'inconnue matérielle n°1) : les fils PMOD n'ont pas la même
  longueur → skew. Le `ss_link` est source-synchrune (horloge envoyée avec les data) →
  robuste au skew si bien conçu. **C'est LA chose qu'on valide ici** (jamais testée hors
  sim). Si ça rate : ralentir le lien, ou ajouter un entraînement/calibration.
- ⚠ **Masse commune** : relier les GND des 2 cartes (sinon signaux flottants).
- ⚠ **PMOD pinout** : vérifier quels PMOD de l'Ext-Board mappent vers quels pins du
  module (le repo wuxx a le pinout). Choisir 2 PMOD adjacents pour data+frame.
- ⚠ **`ss_link` width** : le DW=8 du projet → adapter au nombre de fils disponibles.
- ⚠ **CDC** : les 2 cartes ont des PLL indépendantes → `async_fifo` (Gray + 2FF)
  OBLIGATOIRE (déjà dans le projet, ne pas court-circuiter).

### Étape 5 — Tensor-parallel NN=2 (le cœur du cluster)
**Objectif :** FFN + attention répartis sur les 2 nœuds via le lien → == résultat sim.
- Porter `ffn_tp_seq2` + `attn_tp_seq` (déjà NN-paramétrés, logique pure, se portent).
- Instancier 2 nœuds + le séquenceur sur une carte, le second nœud sur l'autre (ou
  répartition équivalente).
- All-gather (row-parallel W1/W3) + all-reduce (column-parallel W2/Wo) via le lien.
- Test : un bloc FFN/attention NN=2 == NN=1 (même sortie, aux tolérances).
**Gate 5 :** NN=2 bit-exact/tolérance vs NN=1 sur silicium.
**Pièges connus :**
- ⚠ **Bug instructif du ring all-reduce** : si backpressure, `almost_full` (AFMARGIN) pas
  `full` (le `full` simple perdait 792 octets en sim). Déjà corrigé dans le projet, garder
  la correction.
- ⚠ **Ré-alignement des shifts** : l'all-reduce des partiels int32 doit aligner les shifts
  AVANT la somme (sinon somme de valeurs dans des échelles différentes = faux). Déjà prouvé
  en sim, mais vérifier que le portage conserve cet ordre.
- ⚠ **Handoff x entre attn et FFN** : `curx` on-chip, la sortie d'un bloc = entrée du
  suivant sans repasser par l'hôte. Déjà dans `gen_seq`.
- ⚠ **Adresses SDRAM par nœud** : attention 0x10xxxx (attn) vs 0x11xxxx (FFN) déjà
  séparés dans `layer_tp_top` — garder la convention.

### Étape 6 — Génération (petit modèle, zéro PC)
**Objectif :** un petit modèle (stories260K pour la 1ère validation, puis un ~0,5-1B de la
famille endgame) génère du texte correct, modèle chargé depuis la **flash**, 2 nœuds, zéro PC.
- Porter `gen_seq` (le séquenceur de génération — logique pure, se porte) sur une carte
  (orchestrateur) + 1-2 nœuds workers.
- Boot depuis la flash (W25Q64) → SDRAM (Mémoire 3) → compute. Adapter le boot SD existant
  vers boot **flash SPI** (la i9 a une flash, pas de SD).
- Boucle tokens : embed → 5 couches (NN=2) → lm_head → argmax → token suivant.
**Gate 6 :** texte généré == référence (les 17 tokens oracle pour stories260K ; texte
cohérent pour le ~0,5-1B). Zéro PC.
**Pièges connus :**
- ⚠ **Boot flash SPI ≠ boot SD** : la flash W25Q64 est SPI (comme la SD en SPI mode, mais
  pas de CMD0/init SD). Adapter `sd_ctrl`/`sd_reader` vers un **lecteur flash SPI simple**
  (read quad/standard). Le modèle tient dans 8 Mo pour stories260K (~394 Ko), OK ; pour un
  ~1B il faut streamer par couche (déjà prouvé en sim SD-4).
- ⚠ **Erreurs de lecture flash** : comme la SD, une flash peut avoir des bits faux →
  CRC16 + retry (déjà dans `sd_ctrl`, à porter). Sur flash SPI c'est généralement plus
  fiable que la SD SPI.
- ⚠ **KV-cache on-chip** : stories260K KV ~5 Ko → tient en BRAM ECP5. Pour le ~1B, le KV
  shard /2 nœuds → vérifier qu'il tient en BRAM (1,9 Mb / nœud = ~237 Ko) ou spiller en SDRAM.
- ⚠ **Divergence quantif** : le mono divergeait au token ~10-11 (int8). Pour la validation,
  comparer les **N premiers tokens** à la référence (comme fait en Session 5).

---

## 6. Décisions ouvertes (à trancher pendant le bring-up, pas avant)

- **Famille endgame** (Qwen3 / Gemma / autre) → choisir le petit frère ~0,5-1B de la même
  famille comme cible de validation **réelle** (après stories260K). Architecture GQA/RoPE/
  RMSNorm/SwiGLU commune = rien de nouveau à 30B.
- **Mémoire poids flash vs DRAM** : flash (read-only, NV, moins cher) est le choix par
  défaut ; DRAM seulement si le débit flash devient le goulot (plus de flash en parallèle
  d'abord).
- **Q16** : quand passer au 16 bits pour la précision (phase scale, pas validation).
- **MoE vs dense** : change le budget mémoire/token d'un ~10× (Qwen3-30B-A3B vs 30B dense).
- **SERDES vs Ethernet PHY** pour le scale : la i9 a 2 PHY GbE → alternative viable aux
  SERDES FPGA (le PHY fait la sérialisation). Décision de topologie, phase scale.

---

## 7. Cartographie des fichiers (ce qui se porte / se refait)

| Fichier(s) | Rôle | Portage ECP5 |
|---|---|---|
| `src/top.v` (NODE_ONLY) | datapath nœud (FN/FQ/SS/MM/EE) | **primitives vendor à refaire** (PLL/DSP/IO), logique se porte |
| `sim/link/gen_seq.v` | séquenceur génération LUT-lean | **se porte tel quel** (logique pure + ROM cos/sin) |
| `sim/link/ffn_tp_seq2.v`, `attn_tp_seq.v` | séquenceurs TP NN-paramétrés | **se portent** (logique pure) |
| `sim/link/ss_link.v`, `link_send/recv`, `async_fifo` | lien inter-FPGA | **se portent** (logique pure) |
| `sim/link/vec_alu2.v` | ALU MUL/ADD sérialisé | **se porte** (DSP ECP5 à instancier dedans) |
| `sim/link/ring_*`, `tp_ar_*`, `tp_node` | all-gather/all-reduce | **se portent** (logique pure) |
| `rmsnorm_op`/`silu_op`/`rope_op`/`softmax_op`/`attention_head_op` | opérateurs | **se portent** (logique pure + DSP) |
| `src/sdram.v` | contrôleur SDRAM SDR | **quasi tel quel** (même SDRAM que GW2AR) |
| `src/sd_boot.v`, `sd_ctrl`/`sd_reader` | boot SD | **adapter vers flash SPI** (pas CMD0/init SD) |
| `*.hex` LUT (rsqrt/silu/exp/inv/freq_cis) | tables constantes | **se portent** (`$readmemh` — piège chemin) |
| Attributs `syn_ramstyle="block_ram"` (Gowin) | inférence BSRAM | **renommer** → `ram_style="block"` (ECP5) |
| `gowin_rpll`, `mac18`, `MULTALU18X18` | primitives Gowin | **refaire** → PLL/DSP Lattice |

---

## 8. Comment reprendre en session fraîche

1. Lire ce doc (`docs/PLAN_ECP5_BRINGUP.md`) + les fiches mémoire
   (`cluster-hardware-spec.md`, `session6-routing-status.md`, `cluster-fpga-direction.md`).
2. Vérifier l'état : cartes reçues ? toolchain ECP5 installée ? étape courante ?
3. Reprendre à l'étape courante, gate par gate, tolérances de référence en main.
4. Une étape = potentiellement une session dédiée (méthode qui a réussi pour les étapes A-F).
5. **Garde-fous G1-G7** (voir `docs/REFACTOR_UNIFIED_NODE.md`) : G1 (route≠marche) surtout,
   G2 (prouver en Python avant le RTL), G4 (non-régression des étapes précédentes).

**Invariant absolu : même code NN-paramétré par carte** (NN=1 mono = NN=N cluster). Ne
jamais coder d'hypothèse mono-only qui empêcherait de réintroduire NN>1. Les leviers de
routage (ROM cos/sin, fusion vfile) sont NN-invariants → ne brûlent aucun pont cluster.