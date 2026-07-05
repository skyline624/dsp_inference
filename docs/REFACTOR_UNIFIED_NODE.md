# Refactor « nœud transformer unifié » — état, trajectoire, garde-fous

Branche : `refactor/unified-node-ss-link`
Oracle de référence : `host/infer_v5gen_ref.py` → *« Once upon a time, there was a little girl named Lily. She lo »*

---

## 1. Le problème de départ (ce qu'on ne veut PAS reproduire)

Deux échecs distincts, tous deux à l'origine du blocage :

**A. Le GG monolithique ne route pas.**
`src/top.v` intègre un FSM de génération (`GG`, ~340 états) qui fait *toute* la couche
dans une seule puce. Le build complet échoue au routage :
`PR0004 : 372 unrouted nets`, aucun `.fs` produit. Le design est trop dense pour les
20 736 LUT du GW2AR-18. Cul-de-sac : ce chemin ne donnera jamais de bitstream.

**B. Du code « validé » seulement en P&R, jamais en simulation.**
Le refactor LUT-lean du 5 juillet (`ffn_tp_seq2`, `vec_alu2`) *passait le P&R Gowin* —
on savait qu'il *routait*, mais **personne n'avait vérifié qu'il calculait juste**.
Résultat : 4 bugs fonctionnels dormaient dans ce code, invisibles au P&R.

> **La leçon centrale : « ça route » ≠ « ça marche ». Le P&R ne teste aucune valeur.**
> Tout code non simulé cache des bugs. Sur ce projet, le taux constaté est de
> **2 à 4 bugs par brique non simulée** (voir §4).

---

## 2. La trajectoire choisie

Objectif : **un seul module nœud paramétré par `NN`**.
`NN=1` → une carte génère du texte seule (objectif de base). `NN=N` → cluster
tensor-parallel, même code. Lien parallèle `ss_link` (pas d'UART interne).

On construit sur la **lignée parallèle `sim/link/`** (ss_link + vec_alu2 + séquenceurs v2),
**jamais** sur le monolithe `top.v`/GG qui ne route pas. Le nœud `top.v` est réutilisé
uniquement pour ses **primitives** validées (FN, FQ, MM, SS, EE, RR) qu'on orchestre
depuis un séquenceur externe léger.

Découpage en phases, chacune close par un **gate** (test cocotb vert) avant la suivante :

| Phase | Contenu | Gate |
|---|---|---|
| **0** | Valider v2 en sim (prérequis absolu) | `test_vec_alu2`, `test_ffn_tp_seq2` |
| **1** | Lien parallèle ss_link (transport + nœud) | `test_uart_bridge`, `test_node_ss` |
| **2** | FFN généralisé 2→NN | `test_ffn_tp_seq2` NN=1/2/4, UART+ss_link |
| **3** | Attention généralisée NN | `test_attn_tp` NN=1/2/4, UART+ss_link |
| **4** | Couche + multi-couches | `test_layer_tp`, `test_model_tp` NN=1/2 |
| **5a** | Algo génération prouvé **entier** (Python) | `infer_v5gen_ref.py` = oracle |
| 5b-c | RTL génération (attention causale, lm_head, boucle) | *à venir* |
| 6 | Build synthétisable mono-puce + boot SD | `.fs` routé, 0 unrouted |
| 7 | Nettoyage : une seule lignée vivante | suite verte, grep v1 = 0 |

---

## 3. Ce qui est fait (phases 0 → 5a)

Tout ce qui suit est **validé en simulation** (cocotb + Icarus, conteneur `hdlc/sim:osvb`),
pas seulement en synthèse.

### Phase 0 — go/no-go : 4 bugs du 5 juillet corrigés
Le code qui « routait » était cassé en 4 endroits, tous attrapés par les premiers tests :
1. `vec_alu2` : index non remis à 0 avant requantize (seul `out[63]` écrit).
2. `vec_alu2` : opérandes DSP non remis à 0 pendant le drain (produit doublé, shift +1).
3. `ffn_tp_seq2` : register file **sans port d'écriture** (`vfile` jamais commit → tout X).
4. `ffn_tp_seq2` : lectures `vfile` décalées d'1 cycle (adresse registrée) → lecture combinatoire.
→ `test_ffn_tp_seq2` : FFN 2 nœuds vs float, **err 6.4 %** (< 35 %).

### Phase 1 — lien parallèle ss_link
`uart_bridge.v` : `tx8_link`/`rx8_link`, adaptateurs *drop-in* (interface UART
`data/send/busy`, `data/valid`) roulant sur `async_fifo`. Handshake `rdy` /
« une-livraison-par-acquittement » (sinon la FIFO parallèle noie le FSM).
`top.v` sous `ifdef LINK_SS` : PHY UART remplacés par les adaptateurs, **FSM inchangée**,
UART reste le défaut (zéro régression).

### Phase 2 — FFN généralisé NN
`ffn_tp_seq2` refondu en FSM `(phase, chunk)` : la structure « 2 chunks A/B en dur »
devient une boucle `chunk=0..NN-1`, `REDUCE` = all-reduce en arbre sur NN partiels,
dégénérescence propre à `NN=1` (reduce sauté). Hidden dim = NN·D.
→ NN=1 (1.8 %), NN=2 (6.4 %), NN=4 (5.4 %), UART **et** ss_link.

### Phase 3 — attention généralisée NN
`attn_tp_seq` refondu pareil : Wq/Wk/Wv/Wo bouclés par chunk, gathers = concat des NN
slices par cascade d'ADD padded. Passé à `vec_alu2`.
Bug attrapé : gather doublait le chunk 0 (71.9 % → corrigé à 7.6 %).
→ NN=1/2/4 à **7.6 %** (identique aux 3 = gather exact), UART **et** ss_link.

### Phase 4 — couche + multi-couches
`layer_tp_top`/`model_tp_top` rebranchés sur les séquenceurs v2. Les deux séquenceurs
partagent les mêmes nœuds (mux `phase` attention/FFN), y compris dans la boucle NL couches.
→ couche NN=1 (11.8 %) / NN=2 (12.8 %) ; modèle 2 couches NN=1 (7.7 %) / NN=2 (12.6 %),
UART **et** ss_link.

### Phase 5a — algo de génération prouvé en entier
Avant tout RTL, **prouver l'algo en Python**. Les 3 opérations « float » du blueprint
`host/infer_fpga.py` ont une forme **entière** équivalente qui donne le même texte :
- **KV-cache** : stocké (int8, shift/position), re-aligné au MM par right-shift entier
  vers le shift de référence (= max des T shifts). Pas de dequant/requant float.
- **RoPE** : appliqué à toutes les têtes d'un coup, un seul shift de sortie.
- (argmax cross-shift : reste à câbler en RTL, mais l'algo est celui de la réf.)
→ `infer_v5gen_ref.py` produit exactement le texte + les 17 tokens de `infer_v4sim`.
**C'est l'oracle que le RTL de génération devra reproduire.**

---

## 4. Les garde-fous (comment on évite de retomber dans le problème de départ)

Ce sont des règles **procédurales**, pas des vœux. Chacune répond directement à un
des deux échecs initiaux.

### G1 — Rien n'est « fait » tant que ce n'est pas vert en simulation
Contre l'échec B. Le P&R ne compte pas comme validation. Chaque brique a un test cocotb
qui compare à une **référence numérique** (float ou oracle), avec tolérance. Un module
n'avance à la phase suivante qu'après son gate vert.

### G2 — Prouver l'algo en Python AVANT d'écrire du RTL
Contre l'échec B, en amont. Toute simplification (shift KV unifié, rope global) est
d'abord validée dans `infer_v5gen_ref.py` contre l'oracle texte. On n'écrit jamais
de RTL sur un algo non prouvé — c'est ce qui a coûté cher au départ.

### G3 — Valider chaque brique EN ISOLATION avant de l'intégrer
Contre les bugs qui se masquent en cascade. Ordre imposé : primitive → brique →
assemblage. Ex. Phase 1 : transport (`test_uart_bridge`) validé seul avant le nœud
(`test_node_ss`) avant le FFN complet. Quand le FFN ss_link a échoué, l'isolation a
prouvé que ce n'était ni le lien ni le séquenceur — c'était une **LUT manquante**
(`rsqrt_lut.hex` absente du cwd → rmsnorm sortait X). Sans isolation, on aurait
débogué le mauvais module pendant des heures.

### G4 — Non-régression systématique
Contre l'introduction de bugs par les refactors. Après chaque phase, on relance les
tests des phases précédentes, à NN=1 **et** NN=2. L'`ifdef LINK_SS` garde l'UART comme
défaut → l'ancien chemin ne casse jamais.

### G5 — On construit sur la lignée qui route, pas sur le monolithe
Contre l'échec A. Tout se bâtit sur `sim/link/` (ss_link, séquenceurs v2). Le nœud
`top.v` n'est utilisé que pour ses primitives. Le FSM GG monolithique (372 unrouted)
n'entre jamais dans le code unifié ; il sera supprimé en Phase 7. La preuve que la
lignée route existe déjà : `node_cluster` (nœud complet + séquenceur) → **51 % LUT, routé**.

### G6 — Points de sauvegarde git fréquents, sur du vert uniquement
10 commits, un par jalon, chacun sur du code 100 % vert en sim. On peut toujours
revenir à un état sain. Aucun commit sur du code non validé.

### G7 — Un seul code paramétré, jamais deux lignées divergentes
Contre la cause structurelle du départ (deux lignées « autonome » et « cluster » qui ne
se croisaient jamais). `NN` est le seul paramètre : `NN=1` mono, `NN≥2` cluster. Le
même test tourne aux deux. Pas de fork, pas de divergence.

---

## 5. La suite (phases 5b → 7)

### Phase 5b-c — RTL de génération
Spec entièrement dé-risquée par `infer_v5gen_ref.py` (G2 déjà fait). Blueprint
d'orchestration = `host/infer_fpga.py` (`attention_block_full`, `ffn_block_full`,
`lm_head`). Reste à écrire en RTL, **chaque brique validée isolément (G3)** :
1. **Attention causale** : rope par position (primitive `RR` du nœud), MM avec T
   croissant (le nœud fait déjà softmax + multi-tête), KV-cache re-aligné par right-shift
   entier. Gate : un bloc attention à `pos>0` vs `infer_v5gen_ref`.
2. **lm_head + argmax** : rms_final + matmul vocab=512 en 8 sous-matmuls chunkés +
   argmax cross-shift. Gate : token prédit correct sur un vecteur connu.
3. **Boucle de génération** : embed → 5 couches → lm_head → argmax → token suivant,
   ×17, KV-cache persistant. Gate : `test_gen_tp` produit le texte de l'oracle.

Ampleur réelle : plusieurs centaines de lignes de FSM, **plusieurs sessions**. Ne pas
sous-estimer (le plan d'origine `PLAN_GG_AUTONOMIE.md` chiffrait « ~3,5 h » — largement
optimiste au vu du rythme réel de 2-4 bugs/brique).

### Phase 6 — build synthétisable mono-puce + boot SD
Wrapper `src/node_gen.v` (séquenceur NN=1 + nœud + boot SD, ss_link bouclé on-chip),
`build_node_gen.tcl`, P&R Gowin. Gate : `.fs` routé, 0 unrouted. Marge attendue bonne
(`node_cluster` déjà à 51 %). Boot SD : chaîne `sd_boot.v` + `make_sd_image.py` déjà
testée byte-exact.

### Phase 7 — nettoyage
Après migration v2 validée : supprimer v1 (`ffn_tp_seq`, `vec_alu`), le monolithe
`top.v`/GG non routable, le code mort v5g. Gate : suite complète verte, grep v1 = 0.

---

## 6. Comment lancer les tests (mémo)

```bash
# UART (répertoire sim/) :
docker run --rm -v "D:/developpement/dsp_inference:/work" -w /work/sim hdlc/sim:osvb \
  bash -lc "cp -f ../src/*.hex . ; make clean; NN=2 make TOPLEVEL=<top> MODULE=<test> PARAMS=-P<top>.NN=2"

# ss_link (répertoire sim/link/) : via le script qui copie les LUT .hex (G3 !)
cd sim/link && ./run_ss.sh {ffn|attn|layer|model} <NN>

# oracle génération (Python) :
cd host && python infer_v5gen_ref.py     # doit afficher "OK vs infer_v4sim baseline"
```

> **Piège récurrent (G3)** : les ops `rmsnorm`/`silu`/`softmax` font `$readmemh` de leurs
> LUT (`rsqrt_lut.hex`, `silu_lut.hex`, …) par **chemin relatif**. Ces `.hex` doivent être
> dans le cwd de la simulation, sinon l'op sort **X silencieusement** (rmsnorm sans
> `rsqrt_lut` → résultat tout-X). `run_ss.sh` les copie ; en manuel, `cp -f ../src/*.hex .`
