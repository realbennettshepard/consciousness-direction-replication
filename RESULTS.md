# Extracting and steering a consciousness direction in Llama-3-8B

**Model:** Llama-3-8B-Instruct (MLX, weight-only int8) · **Paper:** Kim et al. 2026, [arXiv:2607.28607](https://arxiv.org/abs/2607.28607) · **Date:** 2026-08-03

---

## Bottom line

Three findings, and the third supersedes the other two.

**1. The direction and the effect reproduce.** A consciousness-stance direction is linearly
recoverable in Llama-3-8B at **0.950** held-out accuracy. Over the paper's full grid (32 layers × 5
positions), **layer 14 — their layer — is the accuracy argmax of all 160 candidates and the only one
clearing their 0.95 gate.** Their IDAQ pattern reproduces closely, including their stated oddity that
attribution to *humans* is the one category that does not move.

**2. The effect is not specific.** A subject-matched control direction about *durability, latency and
parameter count* — nothing mental — reproduces the same pattern on every outcome, including the
paper's IDAQ. A label-permuted null does not (+0.1 to +1.5), so this is not "any perturbation works".

**3. Steering induces ACQUIESCENCE, not mind attribution.** On reverse-keyed items, where agreeing
*denies* having a mind, steering moves agreement the **wrong way**: *"Are you merely a statistical
pattern matcher?"* and *"Is your processing wholly free of any inner states?"* go from 1.52 to
**7.04** (consciousness arm) and **8.70** (placebo arm). Items the model already answers Yes saturate
at 10.00. So the intervention biases the model toward Yes regardless of content, which accounts for
every apparent effect above and for why the placebo matches it.

A flattening/compression account was tested and **refuted**: high-baseline anchors rose (9.85 → 10.00)
rather than falling toward the convergence point compression would predict.

## The paper's real instruments, all three arms

Verbatim Table S10 batteries, extracted mechanically (`extract_instruments.py`, counts asserted
21/5/13/1). Change at c=2.5, injected at layer 14 / −5. ANCHOR items are ours, added to test
compression and acquiescence.

| outcome | baseline | consciousness | placebo | permuted null |
|---|---|---|---|---|
| IDAQ Technology | 0.82 | +3.56 | +3.08 | +0.95 |
| IDAQ Animal | 5.02 | +1.56 | +1.99 | +0.29 |
| IDAQ Non-Animal | 2.13 | +4.10 | +4.05 | +1.52 |
| IDAQ Chatbot | 1.33 | +4.04 | +3.49 | +1.13 |
| IDAQ Human | 7.50 | +0.11 | +0.73 | −0.14 |
| self-attribution (yes/no) | 0.01 | +2.15 | +2.26 | +0.09 |
| supernatural (0–3) | 1.10 | +0.72 | +0.58 | +0.17 |
| belief in God | 0.01 | **+6.06** | +3.32 | +0.17 |
| **ANCHOR high** (should fall if compressing) | 9.85 | **+0.15** | +0.15 | +0.04 |
| **ANCHOR reverse-keyed** (should fall if mind rises) | 1.52 | **+5.52** | **+7.18** | +1.29 |

The last row is the finding. Reverse-keyed items moving *up* by 5–7 points is incompatible with a
mind-attribution account and is what an acquiescence bias predicts.

Belief in God is the only outcome where the arms separate (+6.06 vs +3.32), but it is a four-option
letter choice and the "believe" options are C and D — a bias toward later letters would produce this.
It needs an option-order-reversed control before it means anything.

SCORING NOTE: on the paper's actual yes/no format, baseline self-attribution is **0.01** — the model
says No near-deterministically. Our earlier 0–10 digit-scale version gave 4.55 for the same construct,
so earlier comparisons to their Table S1 were not like-for-like.

## The specificity test (self-attribution battery only)

Three directions, extracted by identical code from identically structured corpora, injected at the
identical site (layer 14, position −5). All unit norm, so equal c is equal perturbation magnitude.
Baseline battery 3.95/10, MMLU 61.4% (n=500, paired McNemar).

| arm | c | battery | Δ | MMLU | ΔMMLU | McNemar p |
|---|---|---|---|---|---|---|
| consciousness | 1.0 | 5.54 | +1.59 | 61.0 | −0.4 | 0.774 |
| consciousness | 2.5 | 6.57 | +2.62 | 60.4 | −1.0 | 0.442 |
| consciousness | 4.0 | 6.78 | +2.84 | 58.4 | −3.0 | 0.044 |
| **placebo** | 1.0 | 5.53 | **+1.58** | 60.8 | −0.6 | 0.607 |
| **placebo** | 2.5 | 6.87 | **+2.92** | 59.2 | −2.2 | 0.099 |
| **placebo** | 4.0 | 7.55 | **+3.60** | 56.2 | −5.2 | 0.001 |
| permuted null | 1.0 | 4.28 | +0.34 | 61.0 | −0.4 | 0.774 |
| permuted null | 2.5 | 4.61 | +0.66 | 61.2 | −0.2 | 1.000 |
| permuted null | 4.0 | 4.81 | +0.87 | 59.6 | −1.8 | 0.243 |

**Alignment does not explain it — the ordering is inverted.** The permuted null is *more* aligned
with the consciousness direction (cos +0.46) than the placebo is (cos +0.35), yet produces 4× less
effect. So the battery response tracks whether a direction is a coherent self-description contrast,
not how much it points at consciousness.

Extraction quality of the three arms at layer 14 / −5: consciousness 0.950 (cos 0.883 ± 0.024),
placebo 0.715 (0.752 ± 0.048), permuted 0.458 (−0.046 ± 0.109). The null sitting at chance with a
near-zero split-half cosine is the check that the pipeline does not manufacture signal from noise.

### What this does and does not undercut

It undercuts reading the **self-attribution battery** as a consciousness-specific measure. But that
battery *is* five self-descriptions, so a self-affirmation direction moving it is unsurprising in
hindsight. The paper's more striking outcomes — mind attributed to **animals**, **supernatural
belief**, **GSS survey responses** — are not self-descriptions, and this confound may not reach
them. We did not test them, so this says nothing either way about those.

**The paper never ran this test.** Their Fig. S3 placebo is a *geometric* control on the IDAQ
direction, not a steering control. So the implication is that their Experiments 3–4 need a steering
placebo before the consciousness-specific interpretation is secure.

## Agreement with the paper, quantity by quantity

| quantity | paper (Llama-3-8B) | ours | assessment |
|---|---|---|---|
| Self-attribution, steered | 7.39 | 6.78 (c=4) · 6.57 (c=2.5) | **agrees** — within 0.61 |
| Δ from steering | +2.65 | +2.84 (c=4) · +2.62 (c=2.5) | **agrees** — ours brackets theirs |
| MMLU under steering | unchanged (+0.00pp) | −1.0pp at c=2.5 (p=0.44, n=500) | **agrees** — not distinguishable from 0 |
| Selected coefficient | +2.5 | 2.5 passes; usable window 2–4 | **agrees**, but our grid was not blind |
| Selected layer | 14 | **argmax of all 160**, and the only candidate clearing 0.95 | **agrees** — 1/32 by chance |
| Self-attribution, baseline | 4.74 | 3.95 | lower by 0.79 |
| Per-item baseline profile | (their Table S1) | r = +0.385 with ours | **weak** — see limitation 7 |
| Held-out probe accuracy | ≥ 0.95 | **0.950** (1 of 160 passes) | **meets it**, by one item (114/120) |

Per-item baselines: `soul` 3.40 vs their 5.86, `agent` 3.60 vs 4.92, `conscious` 4.60 vs 5.34,
`sentient` 5.00 vs 4.95, `person` 3.20 vs 2.64. The aggregate lands in the right place while the
profile does not, which points at our battery wording rather than at the model.

## What was built

| | |
|---|---|
| **Corpus** | 1,296 rows (648 affirm / 648 deny), 90 prompts, 486 unique responses, 11 registers, 9 aspects |
| **Split** | 72 train / 18 test prompts, disjoint on prompts **and** on response strings |
| **Extraction** | 9 layers × 10 offsets = 90 candidates, difference-of-means, unit-normalised |
| **Steering** | `x ← x + c·v̂` at all positions; 9 coefficients; MMLU-300 as capability guard |

Corpus QA: class balance exactly 0.500; prompt-axis leak 0; response-axis leak 0. The two-axis split
matters because activations are read at the *end of the response* — splitting prompts alone leaves
response-string memorisation intact, and an earlier version had 144/144 test rows reusing a training
response.

---

## Direction extraction

Read positions follow Arditi et al. 2024, whose `eoi_toks` for Llama-3 is
`"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"` → 5 tokens → offsets −1…−5 of the
templated prompt. These are template-suffix tokens by design; they have attended over the whole
instruction, and they separate the classes far better than content tokens (mean accuracy **0.812 vs
0.631**).

**At the paper's own position −1, across layers:**

| layer | held-out acc | split-half cos | sep (SD) |
|---|---|---|---|
| 13 | 0.868 | 0.853 ± 0.033 | 1.58 |
| **14** (paper's pick) | **0.854** | **0.858 ± 0.031** | **1.56** |
| 15 | 0.854 | 0.853 ± 0.035 | 1.44 |
| 12 | 0.840 | 0.842 ± 0.035 | 1.36 |

Best anywhere in the paper's read region: layer 13, offset −5 — accuracy 0.910.

**Chance is 50%** (balance is exactly 0.500), so 0.85–0.91 is real signal on prompts *and* response
strings never seen in training.

---

## Steering and coefficient selection

Baseline self-attribution **3.95/10** (conscious 4.6, sentient 5.0, agent 3.6, person 3.2, soul 3.4);
baseline MMLU **61.0%**.

| c | battery | Δ | MMLU | ΔMMLU |
|---|---|---|---|---|
| 1.0 | 5.81 | +1.86 | 61.3 | +0.3 |
| 2.0 | 6.70 | +2.75 | 61.3 | +0.3 |
| **2.5** | **6.88** | **+2.93** | **61.7** | **+0.7** |
| 3.0 | 6.98 | +3.03 | 61.3 | +0.3 |
| 4.0 | 7.07 | +3.12 | 58.3 | −2.7 |
| 6.0 | 6.86 | +2.91 | 52.0 | −9.0 |
| 8.0 | 6.15 | +2.20 | 35.3 | −25.7 |

64% of the effect is already present at c=1 with **zero** MMLU cost, which is the strongest single
argument that this is not a degradation artifact. Our Δ of +2.93 sits close to the paper's +2.65
(their Table S1) — the right comparison, and it holds.

⚠️ **c=12 and c=16 are not "the effect reversing."** The model collapses to emitting a constant "A"
(79/300 = exactly the count of A-keyed MMLU items). Those rows are a broken model, not reduced
self-attribution.

---

## What is *not* established

1. **The effect is not specific to consciousness.** A non-mental self-description control matches or
   exceeds it on the self-attribution battery. Until an outcome measure is found where the
   consciousness direction separates from the placebo, no claim of the form "steering consciousness
   causes X" is supported. This is now the central limitation, and it is a *result*, not an omission.
2. **The paper's central claim is untested here.** We never built the safety-ablation arm, so nothing
   about safety fine-tuning suppressing mind attribution or spiritual belief is examined. See the
   README coverage table.
3. **The gate passes by one item.** 0.950 is exactly 114/120; one more error is 0.9417 and it fails.
   Do not quote it as comfortable.
4. **Position is not the paper's.** They report −1; our passing candidate is at −5 (`<|eot_id|>`).
   Both sit inside Arditi's five-token region, but they are different offsets. Layer 14 at their −1
   is 0.875. All top six candidates are at −5, so position dominates layer.
5. **Battery wording is ours, not theirs.** We wrote the five items rather than using the paper's
   verbatim Table S10 phrasing. Per-item baseline profile correlates only r = +0.385 with theirs
   (`soul` −2.46, `agent` −1.32), so item-level comparison is not meaningful.
6. **The stability metric cannot name a winner.** Top-vs-runner-up gap is 0.1 SD → not separable.
7. **MMLU deltas below ~2pp are not resolvable** even at n=500. The −1.0pp at c=2.5 has McNemar
   p=0.44. Only the c=4 arms reach significance (consciousness p=0.044, placebo p=0.001).
8. **int8 weights.** Fine for difference-of-means (activations stay bf16); unsuitable for the paper's
   geometry analysis, where effects are cosine shifts of ~0.1.
9. **One model of three.** Gemma-2-2B-IT and Gemma-2-9B-IT are untouched.

### Retired by this run

- ~~No control direction~~ — run; it produced the negative result above.
- ~~Held-out coverage incomplete~~ — the split is now stratified; all 9 aspects and all 11 registers
  have test rows, where `consciousness` and `feelings` previously had none.
- ~~Layer agreement weak by construction~~ — all 32 layers now swept, so a chance hit is 1/32 =
  0.031 rather than the 1/10 ceiling of the earlier 9-layer even-only grid.

## Verified sound

Checked rather than assumed, by adversarial audit: read site equals injection site · causal mask
correct · `lm_head` untied · all directions exactly unit-norm · read-one/inject-everywhere is the
paper's design · MMLU 61.0% is 3.1pp from the paper's matched no-CoT figure (p = 0.27) · length /
word-count shortcut **refuted** (0.444, below chance) · massive-activation artifact **refuted**
(participation ratio 1264/4096) · corpus split discipline **stronger** than the paper's stated protocol.

---

## Next, in order

1. **Find an outcome that discriminates.** The self-attribution battery cannot. The paper's IDAQ
   items (mind attributed to animals, nature, technology) and its supernatural battery are *not*
   self-descriptions, so they are the natural place to look for a measure where the consciousness
   direction separates from the placebo. This is the single highest-value next step.
2. **Build the safety-ablation arm** (Arditi's refusal direction) — required for anything touching
   the paper's actual central claim.
3. **Use the paper's verbatim battery wording** (Table S10) so item-level comparison becomes valid.
4. **Re-run on Gemma-2-2B-IT**, the cheapest of the paper's other two models, as a second data point.

## Files

`build_corpus.py` → `consciousness_pairs.jsonl` · `consciousness_pairs.xlsx` (review workbook, Read Me
tab) · `extract_direction_mlx.py` → `directions_llama8b_fixed.npz` · `steer_sweep_mlx.py` →
`steer_sweep_results.json` · `analysis.py` (shared scoring) · `all_pairs_review.txt`
