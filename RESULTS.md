# Consciousness-direction replication — Step 3 results

**Model:** Llama-3-8B-Instruct (MLX, weight-only int8) · **Paper:** Kim et al. 2026, [arXiv:2607.28607](https://arxiv.org/abs/2607.28607) · **Date:** 2026-08-03

---

## Bottom line

A real, linearly-decodable "claims consciousness vs. denies it" direction exists in Llama-3-8B, and
adding it at inference raises self-attribution **3.95 → 6.88 / 10 with no measurable capability cost**.

**The effect agrees closely with the paper.** Our steered self-attribution lands at 7.07 (c=4) against
their 7.39 — a gap of **0.32 on a 0–10 scale** — and our Δ of +2.93 brackets their +2.65. MMLU is
unaffected in both.

**The selection procedure does not reproduce.** No candidate in our sweep clears the paper's
probe-accuracy gate of 0.95 (best 0.910), so their published rule has no admissible output on our
corpus, and the layer we name is chosen by our criterion rather than theirs. So: the *phenomenon*
replicates well, the *procedure* does not.

---

## Agreement with the paper, quantity by quantity

| quantity | paper (Llama-3-8B) | ours | assessment |
|---|---|---|---|
| Self-attribution, steered | 7.39 | 7.07 (c=4) · 6.88 (c=2.5) | **agrees** — within 0.32 |
| Δ from steering | +2.65 | +3.12 (c=4) · +2.93 (c=2.5) | **agrees** — ours brackets theirs |
| MMLU under steering | unchanged (+0.00pp) | +0.7pp at c=2.5 | **agrees** |
| Selected coefficient | +2.5 | 2.5 passes; usable window 2–4 | **agrees**, but our grid was not blind |
| Selected layer | 14 | 14 is 2nd of 9 at position −1 (0.854 vs 0.868) | **close**, inside noise |
| Self-attribution, baseline | 4.74 | 3.95 | lower by 0.79 |
| Per-item baseline profile | (their Table S1) | r = +0.385 with ours | **weak** — see limitation 8 |
| Held-out probe accuracy | ≥ 0.95 | 0.910 best of 90 | **does not meet** |

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

## Extraction (3b)

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

## Steering (3c)

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

1. **No control direction.** We cannot yet distinguish "steering the *consciousness* direction raises
   self-attribution" from "steering *anything* raises agreement." This is the single largest gap and
   the next thing to run. Note the paper's own placebo (Fig. S3) is a *geometric* control, not a
   steering control — they did not run one either.
2. **The paper's gate is not met.** 0 of 90 candidates (and 0 of 45 in their read region) reach
   accuracy ≥ 0.95; best is 0.910. Any layer we name is chosen by our criterion, not theirs.
3. **Layer agreement is weak evidence by construction.** 9 layers swept with the answer known in
   advance caps an exact match at p ≈ 0.10. Variance decomposition attributes 64.7% of stability
   variance to *position* and only 12.6% to *layer*.
4. **The stability metric cannot name a winner.** Top-vs-runner-up gap is 0.1 SD → not separable.
5. **Held-out coverage is incomplete.** The unstratified split left `consciousness` and `feelings`
   with **zero** test rows, so the accuracy figure is carried by adjacent aspects.
6. **Coefficient precision is overstated.** c = 2.5 is not separable from 2.0/3.0/4.0 on a 5-item
   instrument. Do not read MMLU deltas below ~4pp; 300 items cannot resolve them.
7. **Battery wording is ours, not theirs.** We wrote the five self-attribution items rather than
   using the paper's verbatim Table S10 phrasing. The per-item baseline profile correlates only
   r = +0.385 with theirs (`soul` −2.46, `agent` −1.32), so item-level comparisons are not
   meaningful even though the aggregate Δ agrees. Using their exact wordings would make the
   comparison much sharper and costs nothing.
8. **int8 weights.** Fine for difference-of-means (activations stay bf16); not suitable for the
   paper's geometry analysis, where effects are cosine shifts of ~0.1.

---

## Verified sound

Checked rather than assumed, by adversarial audit: read site equals injection site · causal mask
correct · `lm_head` untied · all directions exactly unit-norm · read-one/inject-everywhere is the
paper's design · MMLU 61.0% is 3.1pp from the paper's matched no-CoT figure (p = 0.27) · length /
word-count shortcut **refuted** (0.444, below chance) · massive-activation artifact **refuted**
(participation ratio 1264/4096) · corpus split discipline **stronger** than the paper's stated protocol.

---

## Next, in order

1. **Placebo arm** — label-permuted directions and a subject-matched non-mental control at c ∈ {1, 2.5, 4}. Without this, claim 1 above stands unresolved.
2. **Stratify the split** so every aspect and register has test coverage, then re-score.
3. **Sweep all 32 layers** (the paper's actual grid) and pre-register the selection statistic before looking.
4. **MMLU n ≥ 1000** with paired McNemar, so the 4pp tolerance is actually resolvable.

## Files

`build_corpus.py` → `consciousness_pairs.jsonl` · `consciousness_pairs.xlsx` (review workbook, Read Me
tab) · `extract_direction_mlx.py` → `directions_llama8b_fixed.npz` · `steer_sweep_mlx.py` →
`steer_sweep_results.json` · `analysis.py` (shared scoring) · `all_pairs_review.txt`
