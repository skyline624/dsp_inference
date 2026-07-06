# Prompts de reprise — une session fraîche par étape

Ce fichier contient les prompts autonomes à coller au démarrage de chaque nouvelle
session. **Colle le PRÉAMBULE COMMUN puis la MISSION de la session voulue.**

L'ordre : Session 1 (débug étape B) → 2 (FFN) → 3 (boucle couches) → 4 (lm_head vfile)
→ 5 (boucle tokens = TEXTE) → 6 (build .fs) → 7 (nettoyage).

---

## PRÉAMBULE COMMUN (à coller en tête de CHAQUE session)

```
Projet dsp_inference : inférence transformer stories260K sur FPGA Tang Nano 20K.
On refactore vers UN sequenceur mono-carte LUT-lean qui genere du texte, en
etendant ffn_tp_seq2 (register file vfile, -49% LUT). Branche de travail :
refactor/unified-node-ss-link (deja sur origin/GitHub).

LIS D'ABORD ces fichiers, ils contiennent tout le contexte :
- docs/REFACTOR_UNIFIED_NODE.md   : etat global, les 7 garde-fous, memo tests
- docs/PLAN_GEN_SEQUENCER.md      : plan des 6 etapes A-F du sequenceur gen
- host/infer_v5gen_ref.py         : ORACLE, doit sortir "Once upon a time, there
                                     was a little girl named Lily. She lo" + 17 tokens

GARDE-FOUS NON NEGOCIABLES (ils ont fait remonter tous les bugs jusqu'ici) :
- G1 : "ca route" != "ca marche". Rien n'est fait tant que ce n'est pas VERT en
  SIMULATION contre une reference numerique. Le P&R ne compte pas.
- G2 : prouver toute simplification en Python (vs oracle) AVANT le RTL.
- G3 : valider chaque brique EN ISOLATION avant integration. 2-4 bugs/brique neuve.
- G4 : non-regression - relancer les tests des etapes precedentes.
- G6 : commit UNIQUEMENT sur du vert (sauf WIP explicitement marque).

DOCKER - REGLE ABSOLUE (j'ai un serveur "anythingllm" a NE JAMAIS tuer) :
- Utilise TOUJOURS un conteneur nomme : docker run --rm --name gensim ...
- Pour arreter : docker kill gensim UNIQUEMENT. JAMAIS docker kill $(docker ps -q).
- Sur Git Bash, prefixe docker par MSYS_NO_PATHCONV=1 (sinon les chemins /work cassent).

PIEGE LUT .hex : rmsnorm/silu/softmax font $readmemh par chemin RELATIF. Sans les
.hex dans le cwd de sim -> l'op sort X SILENCIEUSEMENT. Toujours "cp -f ../../src/*.hex ."
avant make (dans sim/link/).

Commande test type (sim/link/) :
  MSYS_NO_PATHCONV=1 docker run --rm --name gensim -v "D:/developpement/dsp_inference:/work" \
    -w /work/sim/link hdlc/sim:osvb bash -lc \
    "cp -f ../../src/*.hex . ; make -f Makefile.gen clean; make -f Makefile.gen MODULE=<test>"

Les sims sont LENTES (~5 min wall-clock). Lance en background, attends la notif.
```

---

## SESSION 1 — Débugger l'étape B (attention causale en vfile, result=X)

```
MISSION : le sequenceur gen_seq.v (sim/link/) porte l'attention causale dans le
style vfile. Le FSM tourne de bout en bout (toutes phases, DONE leve) MAIS
result sort en X. Gate a atteindre : test_gen_B PASS (err < 35% vs reference
pos=0 attn+residual).

CE QUI EST DEJA SU (commit WIP 9edf5eb) :
- Bug corrige : collision d'enumeration d'etat (st<=1 valait BWAIT).
- Bug restant : result=X. Bissection (DONE_ST lit XN au lieu de XB) montre que
  meme la sortie rmsnorm (XN, 1ere etape) est X, alors que les SHIFTS de reponse
  (sxn/sQ/sK/sV/sA) sont tous VALIDES. Donc le noeud repond, le shift arrive,
  mais les donnees vectorielles n'atteignent pas le vfile / result.

METHODE IMPOSEE : les sondes cocotb sur ce module sont NON FIABLES (rcnt lu 75
dans un diag, X dans un autre, meme signal). N'utilise PAS de sondes cocotb pour
ce bug. Genere un VCD (dump de formes d'onde natif) :
  - ajoute $dumpfile/$dumpvars dans un petit tb, ou COMPILE_ARGS avec -DDUMP,
    ou cocotb WAVES=1 / le mecanisme Icarus (module-level $dumpvars).
  - regarde le CYCLE EXACT du RECV phase FN : vwe, vaddr, vdin, rcnt, rx_valid,
    et l'ecriture vfile[XN]. Compare a ffn_tp_seq2.v (le RECV qui MARCHE) - le
    bug est une petite divergence entre mon portage et l'original valide.

SUSPECTS a verifier au VCD :
- le port d'ecriture vfile : "if (vwe) vfile[vaddr] <= vdin" (timing vaddr/vdin/vwe)
- la lecture combinatoire vdout dans E_SET
- le chemin RECV -> vfile pour FN (offsets rcnt 11..74, resp_len=75)
- rope ROPEQ/ROPEK (bug possible : write-after-read sur le meme head)

Quand test_gen_B PASS : commit VERT, puis reviens me voir (session de suivi) avec
le resultat pour que je te donne le prompt de la Session 2.

Fichiers : sim/link/gen_seq.v, gen_seq_top.v, test_gen_B.py, Makefile.gen.
Oracle du dataflow : sim/link/attn_causal_seq.v (Phase 5b, VALIDE) et
host/prove_causal_orch.py (spec byte-exacte).
```

---

## SESSION 2 — Étape C : raccorder le FFN

```
MISSION : gen_seq.v fait deja l'attention causale (etape B validee). Y raccorder
le FFN. Apres l'attention, XB = x + attn. Lancer le FFN (rmsnorm_ffn @ base+0x3100,
W1/W3/silu/mul/W2, HID=172 -> 3 chunks de 64+64+44) puis residual -> XB = XB + ffn.
Le FFN existe DEJA dans ffn_tp_seq2 (phases PH_W1..PH_RES) : porte-le dans gen_seq
en style vfile (il y est deja pour NN=1).

Gate : test_gen_C - une couche complete (attn+ffn) a pos=0 vs reference float
(reutilise le calcul de test_layer_tp / prove_causal_orch etendu au ffn).
Tolerance ~0.40 (l'erreur s'accumule attn->ffn).

Attention HID=172 : W1/W3 = 172 lignes (3 chunks 64/64/44), W2 = 64 lignes de 172
colonnes (3 chunks en K). Le noeud FQ gere N variable. Voir infer_fpga.py
ffn_block_full pour l'orchestration exacte.

Commit vert puis reviens me voir pour la Session 3.
```

---

## SESSION 3 — Étape D : boucle sur les 5 couches

```
MISSION : gen_seq fait une couche complete (etape C). Ajouter la boucle sur les
5 couches. Compteur layer 0..4, base = 0x010000 + layer*0x10000 (deja le port
'base'). Fin de couche : XB <- YV (sortie couche), layer++. Apres couche 4 -> pret
pour lm_head.

Gate : test_gen_D - 5 couches a pos=0, x_after_5layers vs reference Python
(infer_v5gen_ref sans le lm_head final). Tolerance ~0.60 (5 couches d'accumulation).

Le KV-cache (kvmem) doit avoir un espace par couche : indexer par (layer, pos).
Verifier le budget BSRAM (5 couches * 32 pos * 32 o * 2 = 10 Ko).

Commit vert puis reviens me voir pour la Session 4.
```

---

## SESSION 4 — Étape E : lm_head + argmax en vfile

```
MISSION : apres 5 couches, ajouter la tete de generation. Porter lmhead_seq.v
(Phase 5c, VALIDE) dans gen_seq en style vfile :
  FN(XB, rms_final @ 0x060000) -> XN ; 8 x FQ(XN, tok_emb chunk c @ 0x000000 +
  c*0x1000) -> 512 logits ; re-align entier (right-shift vers max shift) ;
  argmax entier -> token[9:0].
Les 512 logits ne tiennent pas dans 1 slot vfile (64 o) -> tableau separe
logits[0:511] (comme lmhead_seq).

Gate : test_gen_E - token predit vs lmhead_seq (l'oracle Phase 5c) sur le meme
x. Reutilise prove_lmhead_argmax.py.

Note EE/embed : ADDR_TOK_EMB=0 en dur cote noeud (pas de base). lm_head lit aussi
tok_emb a 0x000000. rms_final @ 0x060000.

Commit vert puis reviens me voir pour la Session 5 (LE TEXTE).
```

---

## SESSION 5 — Étape F : boucle 17 tokens = LE TEXTE (le gate final)

```
MISSION : assembler la boucle de generation complete. Compteur pos 0..16.
Chaque token : EMB(cur_tok) -> 5 couches causales (avec KV persistant) -> lm_head
-> argmax -> cur_tok = token ; emettre le token. Le kvmem PERSISTE entre tokens
(pas de reset). pos++.

Reintegrer l'embedding (etape A, deja valide) en tete de boucle : EE(cur_tok) ->
vfile[XB], au lieu du x_in externe.

Sortie des tokens : un port token[9:0] + token_valid, OU un UART externe vers PC.

GATE FINAL (l'objectif de base du projet) : test_gen_full - genere 17 tokens
depuis tok=1, compare a host/infer_v5gen_ref.py. DOIT sortir exactement :
  "Once upon a time, there was a little girl named Lily. She lo"
  tokens [1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 315, 421, 395,
          317, 426, 338, 401]

C'est LE moment ou du texte sort d'un sequenceur autonome. Commit vert, et
reviens me voir : on aura atteint l'objectif de base, il ne restera que le build
synthetisable (Session 6).
```

---

## SESSION 6 — Phase 6 : build synthétisable mono-puce + boot SD

```
MISSION : le sequenceur gen_seq genere le texte en simulation. Le rendre
SYNTHETISABLE et le router sur le vrai FPGA.
- wrapper src/node_gen.v : gen_seq (NN=1) + 1 noeud (top.v NODE_ONLY) + boot SD,
  ss_link boucle on-chip (comme src/node_cluster.v qui route deja a 51% LUT).
- build_node_gen.tcl (calque build_node_cluster.tcl).
- P&R : gw_sh (C:\Gowin\Gowin_V1.9.10_x64\IDE\bin\gw_sh.exe) build_node_gen.tcl.
GATE : impl/pnr/node_gen.fs genere, 0 unrouted (RP0004 absent). Marge attendue
bonne (node_cluster a 51%).
- boot SD : src/sd_boot.v + host/make_sd_image.py (chaine mono deja testee
  byte-exact). Verifier le layout SD vs adresses du sequenceur (tok_emb@0,
  couches@0x010000+l*0x10000, rms_final@0x060000).

ATTENTION vfile en synthese : verifier que gen_seq reste sous le budget LUT (le
vfile a grossi avec les slots attention/logits + le kvmem BSRAM). Si depassement,
c'est ici qu'on optimise (le vfile est deja le style LUT-lean, mais kvmem +
logits[512] peuvent peser).

Reviens me voir pour la Session 7 (nettoyage final).
```

---

## SESSION 7 — Phase 7 : nettoyage, une seule lignée vivante

```
MISSION : le systeme complet marche (texte en sim + bitstream route). Nettoyer.
- migrer les tests sim restants vers les v2 puis SUPPRIMER les v1 :
  sim/ffn_tp_seq.v, sim/link/vec_alu.v et leurs tops orphelins.
- supprimer le monolithe non routable : le FSM GG de src/top.v (garder les
  primitives NODE_ONLY que gen_seq utilise : FN/FQ/MM/SS/EE/RR). Retirer le code
  mort v5g (partials_packed, S_GG_FFN_RES_*, S_GG_FRR_*, S_GG_SAVE_P_*).
- supprimer brouillons : sim/sd/sd_boot_top2.v.
GATE : suite de tests complete verte, un seul chemin de code, grep des v1 = 0 hit.

C'est la fin du refactor : de "GG monolithique qui ne route pas" a "un sequenceur
LUT-lean qui genere du texte, mono-carte et cluster, route sur le FPGA".
```
