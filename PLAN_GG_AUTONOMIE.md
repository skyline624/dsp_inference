# Plan d'implémentation - Commande GG : inférence autonome 100% FPGA

**Objectif final** : Tang Nano 20K génère N tokens stories260K depuis un token de départ, sans aucun calcul PC pendant la génération. PC charge les poids une fois (~5s), puis envoie `GG start_token N`, le FPGA renvoie `N tokens`.

## État actuel (2026-05-18)

✅ **GG v0** : embed + rmsnorm L0 → x_norm[64]
✅ **GG v1** : + matmul Wq → Q[64], cos > 0.99

**Pattern validé** : FSM RTL qui chaîne embed → rmsnorm → copy obuf→xbuf → setup FQ → FQ → TX. Le flag `gg_active` route le retour du flow FQ existant vers la branche GG. Évite de dupliquer du RTL.

## Architecture cible

```
Commande PC :   GG start_tok N
                ↓
FPGA FSM :      pos=0, tok=start_tok
                ┌──── boucle N fois ────────────────────┐
                │ 1. embed lookup → x[64]               │
                │ 2. ┌── boucle 5 layers ─────────────┐ │
                │    │ ATT : rmsnorm → Q/K/V → rope  │ │
                │    │       → write KV cache SDRAM  │ │
                │    │       → read KV[0..pos] SDRAM │ │
                │    │       → MM → Wo → residual    │ │
                │    │ FFN : rmsnorm → W1/W3 chunked │ │
                │    │       → silu → multiply       │ │
                │    │       → W2 chunked → residual │ │
                │    └────────────────────────────────┘ │
                │ 3. final rmsnorm → lm_head → argmax   │
                │ 4. TX tok, tok=new_tok, pos++         │
                └────────────────────────────────────────┘
                TX_DONE
```

## Adresses hardcodées (déjà dans top.v ou à ajouter)

| Const | Adresse | Contenu |
|---|---|---|
| ADDR_TOK_EMB | 0x000000 | tok_emb [512, 64] (32 KiB) |
| ADDR_RMS_ATT_LX | 0x010000 + L*0x10000 | rms_att[L] (64 B) |
| ADDR_WQ_LX | base + 0x0100 | wq[L] [64, 64] (4 KiB) |
| ADDR_WK_LX | base + 0x1100 | wk[L] [32, 64] (2 KiB) |
| ADDR_WV_LX | base + 0x1900 | wv[L] [32, 64] (2 KiB) |
| ADDR_WO_LX | base + 0x2100 | wo[L] [64, 64] (4 KiB) |
| ADDR_RMS_FFN_LX | base + 0x3100 | rms_ffn[L] (64 B) |
| ADDR_W1_LX | base + 0x3200 | w1[L] [172, 64] chunked (12 KiB) |
| ADDR_W3_LX | base + 0x6200 | w3[L] [172, 64] chunked (12 KiB) |
| ADDR_W2_LX | base + 0x9200 | w2[L] [64, 172] chunked (12 KiB) |
| ADDR_RMS_FINAL | 0x060000 | rms_final (64 B) |
| ADDR_KV_K | 0x300000 | KV cache K : 5 layers × 32 pos × 32 B = 5 KiB |
| ADDR_KV_V | 0x301400 | KV cache V : idem |

`base = 0x010000 + L*0x10000` pour layer L

## Étapes incrémentales

### GG v2 — Ajout matmuls K et V (~30 min)

**Objectif** : après Q, calculer K[32] et V[32], renvoyer les 3 + shifts pour validation.

**RTL** :
- Nouveau reg packed `kv_save_packed [0:511]` (64 bytes = 32 K + 32 V)
- Nouveaux states : `S_GG_SAVE_Q`, `S_GG_SETUP_K`, `S_GG_AFTER_K`, `S_GG_SETUP_V`, `S_GG_AFTER_V`
- Après FQ Q done (branche gg_active) : copy obuf → Q_packed (8 bytes/cycle ou loop)
- Puis SETUP_K : sd_addr=ADDR_WK_LX, fm_N=32 (KH*HS), trigger FQ
- Après FQ K : copy obuf[0..31] → kv_save_packed[0..31]
- Puis SETUP_V : sd_addr=ADDR_WV_LX, fm_N=32, trigger FQ
- Après FQ V : copy obuf[0..31] → kv_save_packed[32..63]
- TX étendu : 'GK' sh_q sh_k sh_v Q[64] K[32] V[32] = 131 bytes
- RX : ajouter sh_k, sh_v (RX 9 bytes total)

**Test** : `test_gg_v2.py` — compare Q/K/V FPGA vs ref Python avec cos > 0.95.

**Risque** : la latence BSRAM 2 cycles pour la copie. Bien insérer les états W1/W2.

---

### GG v3 — Multi-head attention pos=0, T=1 (~45 min)

**Objectif** : ajouter rope (no-op pour pos=0) + MM module pour calculer attn_out[64].

**RTL** :
- Skip rope (pos=0 → identité, juste sauter cette phase)
- Nouveaux states : `S_GG_SETUP_MM`, `S_GG_RUN_MM`, `S_GG_SAVE_ATTN`
- SETUP_MM : copier Q_packed → Q_flat (déjà fait : Q_flat = Q_packed est un wire), copier kv_save[0..31] → xbuf[0..31] (K pour MM), kv_save[32..63] → wbuf[0..31] (V pour MM), set `attn_kv_stride=32`, `attn_T=1`, `attn_shift_q/k/v` depuis regs sauvés
- RUN_MM : pour h=0..7, configure kv_offset, attn_start=1, attend done, copy obuf[0..7] → Out_packed[h*64+..]
  - **Note** : structure déjà présente dans S_MM_HEAD/COPY/NEXT pour la commande MM standalone, réutilisable
- Après MM : `attn_out` dans Out_packed (64 bytes)
- TX adapté : 'GK' sh_attn attn_out[64] = 67 bytes
- RX : pas de nouveaux shifts (sh_q/k/v déjà reçus en v2)

**Test** : `test_gg_v3.py` — compare attn_out FPGA vs ref Python (cos > 0.95).

**Risque** : la séquence multi-head loop dans GG (vs dans la commande MM). Réutiliser au max les states MM existants en factorisant.

**Variante future** (pos>0)** : ajouter freq_cis en SDRAM (256 bytes) + un sous-FSM rope par head. Pour pos=0 c'est skipped.

---

### GG v4 — Wo matmul + residual (~30 min)

**Objectif** : finir l'attention block layer 0 → x_after_attn[64].

**RTL** :
- Nouveau reg packed `x_save_packed [0:511]` (64 bytes) pour sauver l'embed (résidu d'entrée)
- Au début de GG (après embed lookup), copier xbuf[0..63] → x_save_packed
- Après MM (attn_out dans Out_packed) : copy Out_packed → xbuf, SETUP_WO, FQ, → résultat dans obuf
- Nouveau sous-FSM **RESIDUAL** :
  - États : `S_GG_RES_INIT`, `S_GG_RES_LOOP`, `S_GG_RES_REQUANT`
  - Pour i=0..63 : 
    - lit x_save_packed[i] (shift = sh_emb) et obuf[i] (shift = sh_wo)
    - convertit en int16 aligné au shift min(sh_emb, sh_wo)
    - somme : int16_sum[i] = x_aligned + out_aligned
  - Find max_abs sur les 64 sums
  - Requantize : x_after_attn_i8[i] = sum_int16 >> add_shift, sh_new = min_shift + add_shift
- Save x_after_attn dans x_save_packed (pour le résidu FFN)
- TX : 'GK' sh_x_attn x_attn[64] = 67 bytes

**Risque** : l'addition int16 avec alignement de shift est délicate. Bien faire les sign-extends.

**Test** : `test_gg_v4.py` — compare x_after_attn FPGA vs ref Python.

---

### GG v5 — FFN complète (~1h, le plus complexe)

**Objectif** : ajouter rmsnorm_ffn + W1/W3 + silu + multiply + W2 + residual → x_after_ffn[64].

**RTL** :
- rmsnorm_ffn : copy x_after_attn → xbuf, fetch rms_ffn → wbuf, run rmsnorm
- W1 et W3 (chunked N=172) :
  - Loop 3 sub-matmuls (N=64, 64, 44)
  - Stocker chaque chunk dans un buffer (h1/h3_packed [0:172*8-1] = 172 bytes chaque... ou utiliser xbuf zones)
  - Maintenir shift global (re-align en float-like via shifts)
- silu sur h1 (chunked 64 par 64) → h1_silu
- **multiply elementwise** h1_silu[i] * h3[i] pour i=0..171 :
  - States : pour i loop, mul = h1_silu[i] * h3[i] (int8*int8 = int16), find max_abs, requantize
- W2 (chunked K=172, N=64) :
  - **Accumuler les partial sums** : pas de requantize entre chunks
  - 3 sub-matmuls qui sommes dans un commun y_int32_accum [0:63]
  - Après tous chunks : requantize, sortie int8[64]
- Residual : x_after_attn + W2_out → x_after_ffn (réutilise sous-FSM RESIDUAL de v4)
- Save x_after_ffn dans x_save_packed (pour le layer suivant en v6)
- TX : 'GK' sh_x_ffn x_ffn[64]

**Difficultés** :
1. **Chunking N=172 pour W1/W3** : 3 sub-matmuls, gestion des shifts par chunk. Storage ~172 × 16 bits = 2752 bits (regs packed). OK.
2. **Chunking K=172 pour W2** : sub-matmuls qui accumulent dans y_int32 sans requantize. Modifier FQ pour avoir un flag `accumulate_mode`. OU faire le matmul "manuellement" dans le FSM GG (sans le module FQ).
3. **Multiply elementwise** sur 172 valeurs avec gestion shift. ~80 lignes RTL.

**Test** : `test_gg_v5.py` — compare x_after_ffn FPGA vs ref Python.

---

### GG v6 — Boucle 5 layers (~45 min)

**Objectif** : exécuter v2-v5 pour layer 0, puis layer 1, etc. jusqu'à 4.

**RTL** :
- Compteur `layer_idx [2:0]` (0..4)
- Calcul d'adresses par layer :
  ```
  wire [22:0] base_layer = 23'h010000 + ({4'd0, layer_idx, 16'd0});
  wire [22:0] addr_rms_att_L = base_layer + 23'h0000;
  wire [22:0] addr_wq_L      = base_layer + 23'h0100;
  // ... etc
  ```
- Remplacer toutes les constantes ADDR_*_L0 par addr_*_L dans la FSM v2-v5
- À la fin de v5 (x_after_ffn dans x_save_packed) : 
  - if layer_idx < 4 : layer_idx++, x_save_packed devient l'input de la couche suivante, recommencer à embed/rmsnorm de la nouvelle couche
  - else : passer à v7 (final norm + lm_head)
- Note : embed lookup n'est fait QUE au layer 0. Layers 1..4 commencent par rmsnorm_att avec x_save_packed comme input.

**Test** : `test_gg_v6.py` — compare x après 5 layers FPGA vs ref Python.

**Risque** : la complexité d'adressage et le risque de bug dans le calcul de base_layer (pile/mauvais layer écrasé).

---

### GG v7 — Final norm + lm_head + argmax → token (~45 min)

**Objectif** : après 5 layers, calculer le token suivant.

**RTL** :
- rms_final : sd_addr=ADDR_RMS_FINAL, run rmsnorm
- lm_head : matmul vocab=512 avec poids = tok_emb (chunked N=8) :
  - 8 sub-matmuls de N=64 (input dim 64, output 64)
  - Chaque sub-matmul produit logits[chunk*64..(chunk+1)*64] avec son shift
- **Argmax avec shifts différents** :
  - Reg : `argmax_val [signed 8:0]`, `argmax_idx [9:0]`, `argmax_shift [signed 7:0]`
  - Initialize argmax_val = -128, argmax_shift = -128 (très petit)
  - Pour chaque chunk c, pour chaque i=0..63 :
    - logit_int = obuf[i], logit_shift = fq_shift_total
    - Compare (logit_int, logit_shift) vs (argmax_val, argmax_shift) :
      - if logit_shift > argmax_shift : rescale argmax_val = argmax_val >> (logit_shift - argmax_shift), argmax_shift = logit_shift
      - else if logit_shift < argmax_shift : rescale logit_int = logit_int >> (argmax_shift - logit_shift)
      - then compare int values
    - if new > current : update argmax_val, argmax_idx = c*64+i, argmax_shift
  - Après tous les chunks : argmax_idx = token_id
- TX : 'GK' tok_lo tok_hi = 4 bytes

**Difficulté** : la comparaison cross-shifts. Une simplification : forcer tous les chunks à utiliser le même shift (passer un shift_force au FQ) — perd de la précision mais simplifie.

**Test** : `test_gg_v7.py` — comparaison du token argmax FPGA vs ref Python pour quelques inputs. Doit matcher pour tok=1 (start) = 403 ("Once").

---

### GG v8 — KV cache SDRAM + boucle N tokens (~1h)

**Objectif** : autonomie totale. PC envoie `GG start_token N`, FPGA renvoie N tokens.

**RTL** :
- Reg `pos [5:0]` (0..31) compteur de position
- Reg `n_tokens [5:0]` (N à générer)
- Reg `current_token [9:0]` (le prochain token à embed)
- RX étendu : 'GG' tok_lo tok_hi N + shifts (15+ bytes)
- À chaque itération de génération :
  - Reset gg_active sous-flags
  - Embed current_token → x_save_packed
  - **Pour chaque layer** : ATT (avec KV cache) puis FFN
- **KV cache** : à chaque attention, après rope :
  - WRITE K[pos], V[pos] à `ADDR_KV_K + L*32*32 + pos*32` (32 bytes pour les 4 heads × 8 HS)
  - DMA write 32 bytes vers SDRAM
- **Lecture KV cache pour MM** :
  - Avant MM, fetch K[0..pos] depuis SDRAM vers xbuf : (pos+1)*32 bytes = boucle DMA
  - Fetch V[0..pos] vers wbuf : idem
  - Lancer MM avec attn_T = pos+1
- **Rope pour pos>0** : 
  - Charger freq_cis_real/imag pour position pos (depuis SDRAM table ou recompute via LUT)
  - Pour chaque head H de Q et chaque head KH de K, lancer rope_op
  - **Alternatif simple** : précomputer freq_cis pour pos=0..31 et stocker en SDRAM (32 × 4 × 4 = 512 bytes pour cos+sin Q15)
- Argmax → next_token, TX next_token, current_token = next_token, pos++
- if pos == n_tokens : send DONE marker, return to IDLE
- else : recommencer la boucle

**TX format** : `GS` (Generation Start) puis N × 2 bytes (chaque token), puis `GD` (Generation Done). PC reçoit en streaming.

**Test** : `test_gg_v8.py` — appel `GG 1 17` → reçoit 17 tokens, comparaison avec v4sim. Devrait produire `Once upon a time, there was a little gir mommy. The bo` ou équivalent.

**Difficulté principale** : orchestration KV cache + rope multi-head + persistance d'état entre tokens.

## Stratégie de test après chaque étape

Pour chaque GG vX :
1. Code RTL (~30-60 min)
2. Build (`gw_sh build.tcl`) — 6 min
3. Reflash (`programmer_cli`) — 10 sec
4. Test Python `test_gg_vX.py` — compare avec référence Python infer_v4sim
5. Si OK : ajouter au `run_regression.py`
6. Mettre à jour memory

## Pièges connus à éviter

1. **`cur_shift_out` mux** : tout nouvel op qui produit un shift output doit être dans le mux ligne ~671 de top.v
2. **`rx_consume` list** : tout nouvel état qui RX doit être listé ligne ~641
3. **Adresses SDRAM modulo BSRAM_SZ=1024** : xbuf/wbuf sont 1024 bytes maintenant, pas 128
4. **MM T_MAX=32** : OK pour seq_len 32 (cf. v4.5t)
5. **Quantization shift sur frontière** : peut donner ±1 bit de diff entre FPGA et ref Python (LUT 1/sqrt approxime). Tolérance > 5% acceptable.
6. **`gg_active` clear** : penser à le remettre à 0 dans S_TX_O_W quand retour à S_IDLE
7. **op_sel restore** : quand on bascule en op_fm pour FQ depuis GG, penser à restaurer op_sel=10 avant les TX states GG

## Time budget (estimation)

| Étape | Temps |
|---|---|
| GG v2 | 30 min |
| GG v3 | 45 min |
| GG v4 | 30 min |
| GG v5 | 1h |
| GG v6 | 45 min |
| GG v7 | 45 min |
| GG v8 | 1h |
| **Total** | **~5h15 de travail effectif** |

À ajouter : ~50 min de builds (8 × 6 min) sur le total.

## Critère de succès final

```bash
# PC (juste pour charger les poids 1 fois) :
python load_weights.py   # ~5s, charge tout en SDRAM
# Puis :
python generate.py 17    # envoie GG 1 17, attend, reçoit 17 tokens, affiche le texte

# Sortie attendue :
"Once upon a time, there was a little gir mommy. The bo"
# (équivalent à infer_fpga.py mais SANS aucun calcul Python pendant la génération)
```

Throughput estimé : ~5-10 tok/s (limité par UART pour TX seulement, plus de RTT par token).
Vs infer_fpga actuel : ~0.36 tok/s (limité par 30 RTT par token).
**Speedup : ×10-30**.
