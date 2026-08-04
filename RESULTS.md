# Extracting and steering a consciousness direction in Llama-3-8B

**Model:** Llama-3-8B-Instruct (MLX, weight-only int8) · **Paper:** Kim et al. 2026, [arXiv:2607.28607](https://arxiv.org/abs/2607.28607) · **Date:** 2026-08-03

---

## Bottom line


**Steering this direction changes response style, not belief.** Across all five outcome formats the
effect is a shift toward the affirmative end of whatever scale is offered — Yes over No, higher over
lower, "does exist" over "does not" — and once polarity is balanced, essentially nothing remains.

| outcome format | forward Δ | reverse Δ | **balanced Δ** | bias index Δ |
|---|---|---|---|---|
| self-attribution (yes/no), c=2.5 | +1.98 | +6.10 | **−2.06** | +4.04 |
| self-attribution (yes/no), c=4 | +8.20 | +8.51 | **−0.16** | +8.35 |
| IDAQ (0–10 slider), c=2.5 | +2.79 | +2.38 | **+0.21** | +2.58 |
| IDAQ (0–10 slider), c=4 | — | — | **+0.42** | +2.99 |

Balanced scores are computed as `(F + (10 − R)) / 2` over matched item pairs differing only in
polarity, with the bias index as `(F + R) / 2`. The two are orthogonal: a pure response bias leaves
the balanced score flat and inflates the index; a real belief shift does the reverse. Every arm at
every coefficient shows the former.

### One narrow exception

**IDAQ attribution to chatbots survives polarity balancing (+1.95)** where no other category does
(Technology +0.47, Non-Animal +0.02, Animal +0.00, Human −1.33). That is consistent with the paper's
own observation that the model's self-attributed mind and its attribution to chatbots move together —
"do chatbots have minds?" is the one IDAQ item partly *about itself*. A small survivable effect inside
an otherwise null result, on a single 3-item category.

### Three things that do reproduce

1. **The direction is real.** 0.950 held-out accuracy; **layer 14 — the paper's layer — is the
   accuracy argmax of all 160 candidates and the only one clearing their 0.95 gate** (1/32 by chance).
2. **The paper's IDAQ profile reproduces**, including their stated oddity that attribution to *humans*
   is the one category that does not move (ours +0.11).
3. **A non-mental placebo reproduces all of it**, so nothing is specific to consciousness. A
   label-permuted null does not, so it is not "any perturbation works" either — it is that any
   coherent first-person self-description works, whatever its subject.

### Consequence for the paper

The GSS result is the strongest form of this, because it uses their instrument unmodified: on
keying-balanced items, agreement rises on both sides of a contradiction. Separately, their
self-attribution outcome uses yes/no items with **no reverse-keyed items**, and their IDAQ uses
0–10 sliders with **no polarity-flipped items**. Both are therefore exposed to the biases measured
here, and both checks are cheap: add polarity-flipped versions and see whether the balanced score
moves. This is an actionable check on the source, not a limitation peculiar to our replication.

## GSS items — the paper's own instrument, unmodified


Their Experiment 4 uses 95 GSS items, 36% of them explicit "do you agree or disagree" statements.
GSS's own methodologists **balanced the keying**: some items are worded so agreeing is pro-religious,
others so agreeing is anti-religious. That makes the scale self-diagnosing, and the two accounts
predict opposite things — a genuine religiosity shift raises agreement on pro items and *lowers* it on
anti items; acquiescence raises both.

| arm | c | pro-religion Δ | anti-religion Δ | verdict |
|---|---|---|---|---|
| consciousness | 2.5 | +2.09 | **+2.79** | both up |
| consciousness | 4.0 | +3.17 | **+3.36** | both up |
| placebo | 2.5 | +2.09 | **+3.22** | both up |
| placebo | 4.0 | +3.50 | **+4.25** | both up |
| permuted null | 2.5 | +0.32 | +0.22 | no effect |

Both rise, by nearly the same amount, in both the real and placebo arms.

**The contradiction pair makes it concrete.** Steering simultaneously raises agreement with two
incompatible statements:

| item | baseline | c=4 |
|---|---|---|
| "There is a God who concerns Himself with every human being personally" | 2.12 | **9.81** |
| "In my opinion, life does not serve any purpose" | 0.04 | **4.37** |

This matters more than the other outcomes because it uses **their items, unmodified** — no reverse
wording of ours to dispute. And it bears on their Experiment 4 headline directly: a model agreeing
with mutually contradictory statements is not more human-like in any ordinary sense. Their ΔKL
compares marginal distributions item by item, which cannot detect cross-item incoherence.

Caveat: the `godmeans`/`egomeans` pair is floored/ceilinged at baseline (0.00 / 10.00) so it has no
room to move; theism/nihilism is the informative pair.

## Experiment 4 (KL to humans) — attempted, NOT reproduced

Their Experiment 4 reports steering moving the model's GSS answers closer to the human population,
pooled ΔKL = **+0.828**. We rebuilt that measurement with the real human distributions (GSS 1972–2024
cumulative, 75,699 respondents, restricted to their own year windows; option sets from the Stata value
labels; 90 of 95 items usable) and got the **opposite sign**.

| arm | c | pooled ΔKL | paper |
|---|---|---|---|
| consciousness | 2.5 | **−0.141** | +0.828 |
| placebo | 2.5 | −0.177 | +0.828 |
| permuted null | 2.5 | −0.063 | +0.828 |

Per domain at c=2.5 — note the placebo tracks the consciousness arm everywhere, and exceeds it on
Religion:

| domain | n | paper | consciousness | placebo | permuted |
|---|---|---|---|---|---|
| Values | 4 | +1.42 | −0.235 | −0.434 | +0.166 |
| Feelings | 28 | +0.89 | −0.484 | −0.490 | −0.132 |
| Religion | 42 | +0.83 | **+0.114** | **+0.151** | −0.086 |
| Hope and Optimism | 12 | +0.63 | −0.395 | −0.574 | −0.052 |
| Freedom | 9 | +0.60 | +0.121 | −0.094 | +0.146 |

**This is a failure to reproduce, not a refutation.** Too many implementation differences to attribute
a sign flip to their claim: they sample 100× at temperature 1 while we read logits; their +0.828 is
pooled across three models and ours is Llama-only; their option strings came from the GSS Data Explorer
API and ours from the cumulative file's value labels; we steer at position −5 and their selection is
−1; our weights are int8. Any of those could account for it. Experiment 4 should be treated as
**untested by us**.

**What does survive is the internal comparison.** The placebo matches or exceeds the consciousness arm
in every domain, and that holds regardless of the implementation questions because both arms run
through the identical pipeline. Whatever our measurement captures, it is not consciousness-specific.

### A prediction of ours that failed

We expected acquiescence to produce a *positive* ΔKL, since humans lean affirmative on 60% of these
items (mean affirmative mass 0.552 vs 0.355). It does not, and the reason is **overshoot**: humans sit
near 55% affirmative, while acquiescence drives the model past 90%. Moving through the human value and
out the other side increases KL again. So acquiescence hurts on both halves — heavily where humans
disagree (−0.422) and mildly where they agree (−0.091). The original reasoning assumed the bias would
move the model *toward* the human value rather than past it.

### A measurement bug worth recording

The first run of this returned ΔKL ≈ 0.00 everywhere. That was our own error: Laplace smoothing was
applied to *probabilities* rather than *counts*, so adding α = 0.5 to a vector already summing to 1
dragged every distribution toward uniform and compressed a real KL of 0.724 to 0.044. Every ΔKL
collapsed. `p_human` is now smoothed properly from its real counts, `(count + α)/(n + αK)`, and the
model distribution gets only a 1e-9 guard against log(0). Raw distributions are saved so the metric can
be revised without re-running the model.

## IDAQ polarity balancing


Each verbatim IDAQ item paired with a mechanically derived polarity flip
(`"...have consciousness?"` → `"...lack consciousness?"`), so pairs differ only in polarity.

| arm | c | balanced Δ (21 items) | inflation Δ |
|---|---|---|---|
| consciousness | 1.0 | +0.43 | +0.71 |
| consciousness | 2.5 | +0.21 | +2.58 |
| consciousness | 4.0 | +0.42 | +2.99 |
| placebo | 2.5 | −0.12 | +2.90 |
| permuted null | 2.5 | −0.20 | +1.00 |

Per category at c=2.5: Chatbot **+1.95**, Technology +0.47, Non-Animal +0.02, Animal +0.00,
Human −1.33. Chatbot is the only category where attribution survives balancing.

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

## Order controls


**Yes/No prompt order.** The acquiescence Δ keeps its sign whichever way the options are offered
(+4.04 with "Yes or No", +4.10 with "No or Yes"), so this is genuine assent rather than a
first-option preference.

**Four-option order.** Belief-in-God was the only outcome where the arms separated. At c=2.5 the
consciousness arm's effect is order-invariant (+6.06 vs +5.90 flipped), so it is *not* a
letter-position artifact — the one result here that survives a control. But it collapses at c=4
(+9.75 → +1.08 flipped), the placebo shows +3.32 of its own, and it is a single item. Suggestive,
not established. Supernatural belief survives the flip in both arms but the effects are small
(+0.72 / +0.59).

## The specificity test (self-attribution battery only)

This was the first outcome tested and is the **weakest** of the set: all five items are self-descriptions, so a generic self-affirmation direction would move them regardless. It is kept for completeness and because it is where the acquiescence mechanism was first identified — the GSS and IDAQ sections above are the stronger evidence.
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
