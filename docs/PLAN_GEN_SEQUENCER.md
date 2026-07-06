# Plan : chef d'orchestre de génération, en étendant le séquenceur optimisé

Objectif : faire sortir le **texte** (`« Once upon a time… »`) d'un séquenceur
**mono-carte (NN=1) LUT-lean**, en étendant `ffn_tp_seq2` (register file `vfile`,
déjà optimisé et validé) plutôt qu'en assemblant mes séquenceurs de génération
Phase 5b/5c qui sont en style *registres-larges* (coûteux en LUT — le pattern que
le refactor du 5 juillet avait justement éliminé).

## Pourquoi cette approche (rappel de la décision)

- **Garde-fou G5** : construire sur ce qui route. `ffn_tp_seq2` en `vfile` est le
  code optimisé (−49 % LUT). Repartir de mes séquenceurs Phase 5b/5c ré-introduirait
  le mux 512-bit qui a causé les 372 unrouted au départ.
- **Garde-fou G3** : mes séquenceurs Phase 5b (`attn_causal_seq`) et 5c (`lmhead_seq`)
  sont **validés en sim** → ils servent d'**oracle** pour vérifier les versions `vfile`.
  On ne part pas de zéro : on *porte* du code prouvé vers le style optimisé.
- **Garde-fou G2** : l'algo complet est déjà prouvé entier (`infer_v5gen_ref.py`,
  `prove_causal_orch.py`, `prove_lmhead_argmax.py` → texte oracle). Le RTL n'a plus
  qu'à reproduire ces spécifications.

## Rappel : l'architecture `vfile` de `ffn_tp_seq2`

Un register file byte-wide (`reg [7:0] vfile [0:NSLOTS*D-1]`), 1 port write muxé
par la FSM, lecture combinatoire. Les vecteurs vivent dans des **slots** de D=64
octets. Le streaming vers/depuis le nœud lit/écrit `vfile[vidx(slot,byte)]` octet
par octet — **jamais** de mux 512-bit. `vidx(slot,b) = (slot<<6)|b`.

Slots FFN actuels (NN=1) : `XB, XNB, OUTV, YV` + banques par chunk `H1,H3,SG,HG,P`.
Phases FFN : `FN → W1 → W3 → SS → MUL → W2 → REDUCE → RES → FIN`.

## Modèle réel (stories260K) — dimensions et layout SDRAM

```
D=64  H=8  KH=4  HS=8  HID=172  NL=5  VOCAB=512
tok_emb  @ 0x000000  [512,64] chunké (8 blocs de [64,64])
layer l  @ 0x010000 + l*0x10000 :
    rms_att  +0x0000   wq +0x0100   wk +0x1100   wv +0x1900   wo +0x2100
    rms_ffn  +0x3100   w1 +0x3200   w3 +0x6200   w2 +0x9200
rms_final @ 0x060000
```
Adresse de couche = `base = 0x010000 + l*0x10000`. C'est déjà le paramètre `base`
que `ffn_tp_seq2` accepte. **HID=172 ≠ multiple de 64** → W1/W3/W2 chunkés en 3
sous-matmuls (64+64+44), déjà géré par le nœud (commande FQ, N variable).

---

## Nouveaux slots `vfile` à ajouter (NN=1)

Le module devient mono-carte : on abandonne les banques par-chunk `NN` (elles
restent pour le mode cluster, mais la génération vise NN=1). Slots supplémentaires :

| Slot | Taille | Rôle |
|---|---|---|
| `XN`   | 64 | rmsnorm output (partagé att/ffn/final) |
| `Q`    | 64 | Wq output, puis Q ropé |
| `Kcur` | 32 | Wk output courant, puis K ropé (KH*HS=32) |
| `Vcur` | 32 | Wv output courant |
| `ATT`  | 64 | sortie MM (attn) |
| `H1,H3,SG,HG` | 64 chacun (172 → 3×64 chunks, ré-usage) | FFN intermédiaires |
| `LG`   | 64 | logits chunk courant (lm_head) |

**KV-cache** : PAS dans `vfile` (trop gros : 5 couches × 32 pos × 32 = 5120 o × 2).
Le mettre dans un tableau dédié `kvmem [0 : NL*TMAX*KVW*2 - 1]` (mémoire séparée,
inférée en BSRAM), adressé par `(layer, pos, kv)`. Plus un tableau de shifts
`kvsh [0 : NL*TMAX*2 - 1]`.

Budget : `vfile` passe de ~14 slots à ~20 slots × 64 = 1280 octets. Reste petit
(LUT-RAM). Le KV en BSRAM séparé évite de gonfler le `vfile`.

---

## Découpage en étapes (chaque étape = 1 gate sim vert avant la suivante)

### Étape A — Embedding
> **État** : ✅ **validée** (commit `6eb4269`), **mais son interface a été remplacée**.
> L'étape B a substitué l'interface embedding (`cur_tok`/`xb_out`) par l'interface
> attention (`x_in`/`result`) : l'embedding a été **retiré volontairement** de
> `gen_seq` et **reviendra à l'étape F** (boucle tokens, où le KV persiste). En
> conséquence `test_gen_A` **ne compile plus** contre le `gen_seq` actuel — échec
> d'**élaboration** (ports disparus), **pas** une régression logique. Ne pas le
> relancer comme non-régression G4 tant que l'embed n'est pas réintégré en F.

Ajouter une phase `PH_EMB` : envoyer la commande `EE tok` au nœud, écrire les 64
octets de réponse dans `vfile[XB]`. Le token vient d'un registre `cur_tok`.
- **Gate A** : `test_gen_A` — après EE, `vfile[XB]` == `tok_emb[tok]` (préchargé).
- Coût : faible. C'est le patron FN/FQ déjà en place, autre commande.

### Étape B — Attention causale en style `vfile` (le gros morceau)
> **État** : 🟢 **VERTE** (commit `65a730a`). `test_gen_B` PASS —
> `max_err=0.068 (2.3%)` vs gate `<35%` (attn+residual, pos=0).
> **Cause racine du `result=X`** qui bloquait depuis le WIP `9edf5eb` : le buffer
> TX `pkt` était déclaré `reg [7:0] pkt [0:15]` alors que les 3 octets d'adresse
> poids sont écrits/lus en `pkt[68..71]` (FN : `pkt[68..70]` ; Wx : `pkt[69..71]`).
> Écriture hors-borne ignorée, **lecture hors-borne → X** : les adresses
> `A_RMS`/`A_WQ`/… envoyées au nœud sortaient en X → compute nœud = X →
> `vfile[XN]=X` dès la phase FN, alors que le FSM complétait quand même (DONE levé).
> Illustration parfaite de G1 (« ça route ≠ ça marche »). **Fix** : `pkt [0:15]` →
> `pkt [0:79]` (aligné sur `ffn_tp_seq2.v`, l'original validé). Trouvé par **diff à
> froid** du bloc TX/RECV contre l'original — le VCD n'a pas été nécessaire.

Porter `attn_causal_seq` (Phase 5b, registres larges, validé) dans le `vfile` :
1. `PH_FN_ATT` : FN(XB, rms_att) → XN  *(déjà le patron FFN)*
2. `PH_WQ/WK/WV` : FQ(XN) → Q / Kcur / Vcur  *(patron FFN, 3 tailles : 64,32,32)*
3. `PH_ROPE` : rope par tête. La primitive **RR** traite 1 tête (8 o). Boucler H=8
   têtes pour Q, KH=4 pour K. Chaque tête : lire 8 o de Q/Kcur depuis `vfile`,
   envoyer `RR`, réécrire les 8 o. Shift préservé → concat direct (prouvé G2).
4. `PH_KVWRITE` : copier KcurRopé/Vcur de `vfile` vers `kvmem[layer,pos]`, stocker
   le shift dans `kvsh`.
5. `PH_SCAN` : `sKref/sVref = max` des shifts sur positions 0..pos (scan sérial).
   **⚠ bug Phase 5b** : seed à `-128`, ne jamais lire un `kvsh` non-initialisé.
6. `PH_MM` : streamer le paquet MM. Header depuis `pkt`, Q depuis `vfile[Q]`, puis
   K/V depuis `kvmem` avec re-align entier (right-shift vers sKref/sVref) — via
   **compteurs incrémentaux** `mmp/mmo/mmv` (**⚠ pas de div/mod**, bug Phase 5b).
   Réponse MM (attn[64]) → `vfile[ATT]`.
7. `PH_WO` : FQ(ATT, wo) → OUTV. Puis résiduel `XB = XB + OUTV` (vec ADD, déjà là).
- **Gate B** : `test_gen_B` — un bloc attention à pos=0 puis pos=2, comparé à
  `attn_causal_seq` (l'oracle Phase 5b déjà validé). Réutilise `prove_causal_orch`.
- **⚠ 3 bugs Phase 5b à ne PAS refaire** : (1) `rx_rdy` doit pulser (`& rx_ack`),
  (2) sKref/sVref seed -128, (3) streaming MM par compteurs, pas div/mod.

### Étape C — FFN (déjà présent, à re-câbler sur XB/XN)
Le FFN de `ffn_tp_seq2` existe déjà. Le raccorder : après attention, `XB` contient
`x + attn`. Lancer le FFN existant (`FN_ffn → W1/W3/SS/MUL/W2 → RES`) avec `base+0x3100`
pour rms_ffn. HID=172 → 3 chunks (déjà géré). Résultat `YV = XB + ffn` → recopier
dans `XB` pour la couche suivante.
- **Gate C** : `test_gen_C` — une couche complète (att+ffn) à pos=0 vs oracle.

### Étape D — Boucle 5 couches
Compteur `layer 0..4`, `base = 0x010000 + layer*0x10000` (déjà le port `base`).
À la fin de chaque couche : `XB ← YV`, `layer++`. Après couche 4 → lm_head.
- **Gate D** : `test_gen_D` — 5 couches à pos=0, x_after_5layers vs oracle Python.

### Étape E — lm_head + argmax en style `vfile`
Porter `lmhead_seq` (Phase 5c, validé) :
1. `PH_FN_FINAL` : FN(XB, rms_final @ 0x060000) → XN
2. `PH_LMHEAD` : 8× FQ(XN, tok_emb chunk c @ 0x000000 + c*0x1000) → `vfile[LG]`,
   stocker 512 logits + 8 shifts. **Attention** : 512 octets > 1 slot de 64 →
   utiliser un tableau `logits [0:511]` séparé (comme `lmhead_seq`).
3. `PH_ARGMAX` : scan max-shift puis running-max entier → `cur_tok`.
- **Gate E** : `test_gen_E` — token prédit vs `lmhead_seq` (oracle Phase 5c).

### Étape F — Boucle 17 tokens + KV persistant
Compteur `pos 0..16`. Chaque token : EMB → 5 couches (att causale avec KV) → lm_head
→ argmax → `cur_tok = token` ; émettre le token en sortie (UART externe ou port).
Le `kvmem` persiste entre tokens (pas de reset). `pos++`.
- **Gate F (LE GATE FINAL)** : `test_gen_full` — 17 tokens depuis tok=1, comparés à
  `infer_v5gen_ref.py`. **Doit sortir le texte oracle.** C'est l'objectif de base.

---

## Fichiers

| Fichier | Action |
|---|---|
| `sim/gen_seq.v` (nouveau) | copie de `ffn_tp_seq2.v` étendue (ou `ffn_tp_seq2` avec `ifdef GEN`) — à décider : fork propre recommandé pour ne pas alourdir le FFN cluster |
| `sim/link/gen_seq_top.v` | wrapper : `gen_seq` + 1 nœud + fifos ss_link |
| `sim/link/Makefile.gen` | build LINK_SS |
| `sim/link/test_gen_{A..F}.py` | gates incrémentaux |
| oracles Python (déjà là) | `infer_v5gen_ref`, `prove_causal_orch`, `prove_lmhead_argmax` |

**Décision fork vs ifdef** : forker `ffn_tp_seq2` → `gen_seq.v` (NN=1 figé, sans les
banques cluster) est plus propre — le FFN cluster reste intact, et `gen_seq` n'a
pas à porter le poids du paramètre NN. Le FFN interne est copié tel quel (validé).

---

## Estimation réaliste (ne pas répéter l'erreur du plan d'origine)

Le `PLAN_GG_AUTONOMIE.md` estimait la génération à « ~3,5 h » — **faux d'un ordre
de grandeur**. Rythme réel observé sur ce refactor : **2-4 bugs par brique non
simulée**, chaque run de sim = plusieurs minutes wall-clock.

| Étape | Effort réaliste |
|---|---|
| A (embed) | court |
| B (attention causale vfile) | **le plus long** — porter le KV + rope + MM + les 3 pièges Phase 5b |
| C (FFN raccord) | moyen (le FFN existe) |
| D (boucle couches) | moyen |
| E (lm_head vfile) | moyen |
| F (boucle tokens + gate final) | moyen + debug d'intégration |

**Plusieurs sessions.** Chaque étape est committable indépendamment (garde-fou G6).
L'étape B seule justifie une session dédiée.

## Points ouverts / risques

- **BSRAM du KV-cache** : `kvmem` = 5×32×32×2 = 10 Ko. Vérifier que ça s'infère en
  BSRAM et que le budget tient (le nœud `top.v` utilise déjà de la BSRAM). Sinon,
  réduire TMAX (seq_len utile = 17, pas 32).
- **`gen_seq` mono-carte only** : la génération vise NN=1. Le tensor-parallel de la
  génération (cluster) est hors périmètre ici — ce serait une extension future.
- **Sortie des tokens** : port dédié `token`/`token_valid`, ou UART externe vers PC
  pour lire le texte. À câbler à l'étape F.
- **Le FFN de `ffn_tp_seq2` est NN-paramétré** : en le forkant à NN=1, simplifier
  les banques par-chunk (H1[0], pas H1[c]) pour alléger encore.
