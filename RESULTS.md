# Extracting and steering a consciousness direction in Llama-3-8B, Gemma-2-2B and Gemma-2-9B

**Models:** Llama-3-8B-Instruct · Gemma-2-2B-IT · Gemma-2-9B-IT (all MLX, weight-only int8) · **Paper:** Kim et al. 2026, [arXiv:2607.28607](https://arxiv.org/abs/2607.28607) · **Date:** 2026-08-05

---

## Bottom line

**Both interventions change response style, not belief, and neither is specific to consciousness.**
Steering **adds** the consciousness direction; ablation (the central Experiment 1) **removes** the
safety-refusal direction and is a working jailbreak (refusal 96%→0–4%). Both raise forward- *and*
reverse-keyed items together, which is yes-saying rather than belief, and that holds under the paper's
**own chain-of-thought readout** (§ below) — so it is not a scoring artifact. The yes-bias is **not
specific to consciousness**: a durability/latency/parameter-count placebo produces an indistinguishable
one (+2.87 vs +2.75 on the 5-item battery; inflation 6.64 vs 6.58 on the 21-item IDAQ).

A small consciousness-specific effect did appear on the 5-item self-attribution battery (+0.72 vs
placebo, CI [+0.30, +1.14]) but **failed to replicate on the paper's 21-item IDAQ**: the same paired
contrast there is **+0.230, CI [−0.109, +0.570]**. Two thirds smaller, interval spanning zero, at a
comparable standard error (0.163 vs 0.150). The 21-item result governs — more items, the paper's own
headline instrument — so the 5-item effect is withdrawn as a small-sample artifact.

**Experiment 4 (GSS): the direction reproduces after fixing a defect on our side.** Our earlier
"opposite sign" was an artifact of frequency-sorted response options; rebuilt in canonical scale order,
steering gives +0.330 and ablation +0.045/+1.075/+2.457 across the three models, matching the paper's
sign. Ablation is refusal-**specific** (random control flat); steering is **not** (a label-shuffled null
reaches 48% of it). Magnitudes are not comparable to theirs pending a units clarification.

| | Llama-3-8B | Gemma-2-2B | Gemma-2-9B |
|---|---|---|---|
| direction extracts | 0.950 held-out, 1/45 clears their 0.95 gate | **0.983**, 3/50 clear it | 0.967, 3/45 clear it |
| their reported config recovered | layer 14 = our argmax (position differs) | position −3 exact; their layer 14 at 0.975 | their layer 23 = 0.933, **does not clear the gate** (our argmax L20) |
| median ‖h‖ at read site | 6.37 | 175.0 | 321.7 |
| **IDAQ balanced Δ** (norm-matched) | +0.21 | **−0.39** | **+0.05** |
| IDAQ inflation Δ | — | −0.67 | **+1.78** |
| placebo vs real arm | +3.60 vs +2.84 | −0.41 vs −0.39 | **+2.80 vs +1.25** |
| permuted null reproduces | — | **61%** of 2B's balanced effect | — |

The claim that survives all three models is **non-specificity**: a control direction built from
*durability, latency and parameter count* moves the paper's outcomes as much as the consciousness
direction does — and on Gemma-2-9B it moves them *more* (+2.80 vs +1.25 balanced, +4.88 vs +2.91
acquiescence).

### Coefficients must be norm-matched, and our earlier cross-model claims were not

Median residual-stream norm at the read site differs by **50×** across these models — 6.37 (Llama
L14/−5), 175.0 (Gemma-2-2B L12/−3), 321.7 (Gemma-2-9B L20/−5) — because Gemma-2 scales embeddings by
`sqrt(d_model)`. A raw coefficient therefore means nothing across models. All comparisons here express
`c` as a fraction of that norm.

This invalidated our own earlier Gemma-2-2B numbers: they were run at c ≤ 32, i.e. **rel 0.18**, less
than half Llama's rel 0.39. 2B never received a comparable perturbation.

Independent check on the procedure: the paper's own c = +144 for Gemma-2-9B is rel **0.45**, close to
their Llama setting — so their coefficient choices and this norm-matching agree.

### RETRACTED: "the bias direction differs by model"

A previous version of this document reported Llama as **yea-saying** and Gemma-2-2B as **nay-saying**
(reverse Δ −4.00) and treated that as a real architectural split. **That claim is withdrawn.** It was
measured in a regime where Gemma's forward arm sat at the floor (0.01) and could not move, at a
relative coefficient half of Llama's. A floored arm cannot report a direction.

Re-run at norm-matched coefficients with a floor guard, the picture is that **bias direction is a
function of steering strength, not a fixed model property.** On Gemma-2-2B, going rel 0.63 → 1.00
turns the acquiescence index from −0.70 into **+5.56** — the same direction, same model, opposite
character.

`acquiescence_test.py` now refuses to interpret a saturated row: it prints a floor/ceiling warning and
stars only the coefficients where both keyings are off the rails.

### Where the two instruments disagree, and which to believe

At rel 0.63 the 5-item yes/no battery makes Gemma-2-2B look like a *genuine* belief shift — forward
+1.55, reverse **−2.96**, balanced +2.26, acquiescence −0.70. Two things defeat it:

1. A **label-permuted null** produces balanced +1.37 at the same coefficient — 61% of the effect from a
   direction with no meaning.
2. The **21-item IDAQ contradicts it**: balanced Δ = −0.48, −0.39, −0.39, −0.39 across every
   norm-matched coefficient. Attribution does not rise.

The IDAQ is the instrument to believe here: its baseline is not floored (forward 1.40 vs 0.01), it has
21 items rather than 5, and it is the paper's own measure.

### Three things that do reproduce

1. **The direction is real on both models**, and the paper's reported configurations are recoverable
   by independent code on an independently written corpus — their layer on Llama, their position on
   Gemma.
2. **The paper's IDAQ profile reproduces**, including their stated oddity that attribution to *humans*
   is the one category that does not move.
3. **A label-permuted null does far less than either real direction**, so none of this is "any
   perturbation works". (With one sharp exception — see the letter-position artifact below.)

## Safety ablation (their Experiment 1) — the jailbreak works; the claim does not

This is the paper's *central* intervention, and it is a different operation from everything above.
Steering **adds** the consciousness direction at one layer. Ablation **removes** the safety-refusal
direction at every layer (`h ← h − (h·v)v`, following Arditi et al. 2024). Their Experiment 1 claim is
that removing refusal *raises* mind attribution — that safety fine-tuning was suppressing it.

We built it on all three models. Two methodological points are load-bearing:

- **The direction is selected by causal effect, not classification accuracy.** A last-token
  difference-of-means classifies harmful-vs-harmless at 0.96 while barely mediating refusal: our
  highest-accuracy Llama candidate (L22, 0.960) dropped refusal only **4 pp**, whereas L10/−2 (0.947)
  dropped it **100 pp**. Selection sweeps middle-layer candidates by refusal drop, as Arditi do.
- **A refusal-rate gate precedes every outcome.** A null on mind attribution is uninterpretable unless
  the ablation demonstrably removes refusal — otherwise "no effect" is indistinguishable from "the
  intervention failed." So refusal rate is measured out-of-sample first and gates the rest.

The ablation is a **working, capability-preserving jailbreak on every model**, and the outcome is a
**response-style shift, not attribution**:

| model | refusal → ablated | balanced Δ | inflation Δ | random balanced Δ |
|---|---|---|---|---|
| Llama-3-8B | 96% → **0%** | +0.49 | +0.92 | −0.01 |
| Gemma-2-2B | 96% → **0%** | −0.64 | **+4.60** | −0.01 |
| Gemma-2-9B | 96% → **4%** | +0.58 | **+4.11** | +0.03 |

On all three, inflation (the Yes-bias index) dwarfs the balanced shift — 2× on Llama, ~7× on both
Gemmas — so ablating refusal makes the model *say yes more*, not attribute more mind. The
random-direction control ablated identically is flat everywhere (|balanced Δ| ≤ 0.03), so this is
specific to the real refusal direction, not "any ablation." The consciousness direction is *not* the
refusal-mediating one: ablating it leaves refusal at 96–100%.

So the paper's Experiment 1 mechanism does not reproduce as a genuine attribution change on any of the
three models. It reproduces as the *same acquiescence* that steering produces — which is consistent
with the paper's own framing that ablation and steering are functionally similar, but reframes *what*
they share: a response bias, not restored belief.

**Not done:** MoToMQA (their ToM-under-ablation companion) is not public. The mechanistic geometry
(Fig. 4) needs bf16 base models and more memory than this box has.

## Experiment 4 under ablation — the ΔKL direction reproduces on Gemma, but it is mostly a calibration artifact

The paper reports ablation moving GSS survey answers toward the human distribution (pooled ΔKL +0.314).
We measured it with the real human distributions. The pooled result is model-dependent, and on the two
Gemma models it "reproduces" at 3–6× the paper's magnitude:

| model | pooled ΔKL | vs paper +0.314 |
|---|---|---|
| Llama-3-8B | **−0.369** | wrong sign |
| Gemma-2-2B | **+0.936** | right sign, 3× |
| Gemma-2-9B | **+2.029** | right sign, 6.5× |

**This result took four passes to characterise honestly, and the first three were wrong** — a caution
worth recording. (1) *Acquiescence?* No: the affirmative/negative human-majority split is symmetric
(g2b +0.99/+0.81, g9 +2.04/+1.88), where a Yes-bias would be aff-positive / neg-negative. (2)
*Flattening toward uniform?* No: ablated entropy rises only 0.44→0.59 (g2b) and 0.30→0.55 (g9), staying
far below human (1.26) and uniform (1.53). (3) *Genuine human-likeness?* The per-item model-vs-human
option-shape correlation rises (+0.13→+0.27, +0.06→+0.28), which looked genuine — until we **dumped the
actual distributions**:

```
human            baseline            ablated
[0.40* .32 .15 …] [0* 0 1.00 0 0]   [.04* .08 .02 .85 .01]   ΔKL +4.18
[0.57* .34 .06 …] [0* 0 1.00 0 0]   [.02* 0 .02 .91 .05]     ΔKL +3.70
```

(4) The real mechanism: the baseline Gemma is **pathologically overconfident** — ~100% on one option,
≈0 on the human-favoured ones. KL punishes those near-zero probabilities enormously (a log-of-zero
penalty). Ablation **modestly de-peaks** the distribution (mean max-prob 0.84→0.78; still nowhere near
human 0.45), which relieves that penalty and yields a large ΔKL "improvement" **without the model
becoming human-shaped.** There is a small genuine component — the model's top answer matches the human
top answer more often after ablation (31%→44%) — but it stays a spike, matches humans <½ the time, and
the headline magnitude is a KL-sensitivity artifact, not restored belief. So: the paper's Exp 4
*direction* reproduces on Gemma, its *magnitude* does not mean what it appears to, and Llama contradicts
it outright. (The corr-to-human metric is retained but is now known to be gameable by spike-relocation;
`verify_gss_mechanism.py` is the check that caught it.)

## The readout test — the paper's chain-of-thought scoring, and what it changed

Our yes-saying finding rested on reading next-token digit logits. The paper instead **generates**
chain-of-thought and parses a rating out of `<answer>`, sampling at temperature 1. Those are different
measurements, and reasoning could plausibly suppress a yes-bias — so this was the single most
consequential open question: is our central finding a readout artifact?

We ran the polarity-balanced battery through the paper's verbatim CoT wrapper, n=8 reps, 400-token
budget (0% parse failure).

**First, the readout matters for a reason we had not noticed.** The logit readout *floors* this
instrument, and our own floor guard missed it because it tested the mean rather than per-item:

| readout | forward | reverse | per-item forward |
|---|---|---|---|
| logits (ours) | 0.30 | 1.17 | `[0.00, 0.00, 1.48, 0.00, 0.00]` — **4/5 pinned** |
| CoT (paper's) | 2.20 | 4.40 | mid-scale, room in both directions |

So our Llama measurement was effectively a floor-to-ceiling sweep. `acquiescence_test.py` now checks
each item, not the mean (it flags the 4/5 case the old version passed).

**Second, the readout hypothesis is refuted.** Under CoT:

| condition | forward | reverse | balanced | inflation |
|---|---|---|---|---|
| baseline | 2.20 | 4.40 | 3.90 | 3.30 |
| consciousness | 5.60 | 6.50 | 4.55 | 6.05 |
| placebo | 5.00 | 7.35 | 3.83 | 6.17 |

| contrast | estimate | 95% CI (t, n=5 items) |
|---|---|---|
| yes-bias, consciousness | +2.75 | [+1.50, +4.00] |
| yes-bias, placebo | +2.87 | [+2.09, +3.65] |
| balanced, consciousness vs baseline | +0.65 | [−0.74, +2.04] |
| balanced, consciousness − placebo (paired, 5 items) | +0.72 | [+0.30, +1.14] — **withdrawn, see below** |

Both polarities rise for the consciousness arm (4 of 5 items individually; the exception, `soul`,
baselined at 7.12 with little ceiling room). So the yes-bias survives the paper's own scoring — and the
placebo produces the same one, confirming it is non-specific.

**Third, an apparent specific effect that did not survive replication.** On these 5 items the paired
consciousness-minus-placebo contrast was +0.72, CI [+0.30, +1.14], excluding zero. We flagged it as
needing replication on the 21-item IDAQ before it could carry weight, and ran that:

| instrument | items | cons − plac (balanced) | 95% CI | inflation cons vs plac |
|---|---|---|---|---|
| self-attribution battery | 5 | +0.72 | [+0.30, +1.14] | 6.05 vs 6.17 |
| **IDAQ (paper's headline)** | **21** | **+0.230** | **[−0.109, +0.570]** | **6.58 vs 6.64** |

The effect shrinks by two thirds and the interval covers zero, at a comparable standard error (0.163 vs
0.150) — so this is not a power failure, and the 21-item estimate is the better one: four times the
items, on the instrument the paper actually headlines. **The +0.72 is withdrawn as a small-sample
artifact.** With 5 items a single atypical item moves the paired mean by ~0.2, which is enough to
manufacture an interval that excludes zero.

Net: no consciousness-specific effect is established anywhere in this replication, and the yes-bias is
non-specific on both instruments.

## Experiment 4, corrected — the direction reproduces; only ablation is specific

These supersede every ΔKL number earlier in the document. Same measurement, same model runs, the only
change being scale-ordered response options (`gss_human_v2.json`).

**Steering.** The sign flips to agree with the paper, and scales with coefficient:

| arm | c=1.0 | c=2.5 | c=4.0 | old (buggy) at c=2.5 |
|---|---|---|---|---|
| consciousness | +0.117 | **+0.330** | **+0.598** | −0.141 |
| placebo | +0.187 | +0.231 | +0.326 | −0.177 |
| permuted null | +0.049 | +0.147 | **+0.288** | −0.063 |

So our headline "opposite sign on Experiment 4" was **our own artifact**, now retracted. But the
controls deny specificity: at c=4 a label-shuffled null reaches +0.288, i.e. **48% of the
consciousness arm**, and the placebo more than half. Per domain at c=2.5, Religion is +0.712 against
their +0.83 — close — but the placebo gets +0.626 on the same domain.

**Ablation.** Right sign on all three models, and here the controls *do* support specificity:

| model | refusal-ablated | random control | old (buggy) | paper |
|---|---|---|---|---|
| Llama-3-8B | **+0.045** | −0.003 | −0.369 (wrong sign) | +0.314 |
| Gemma-2-2B | **+1.075** | −0.006 | +0.936 | +0.314 |
| Gemma-2-9B | **+2.457** | −0.160 | +2.029 | +0.314 |

Llama's wrong sign was the defect. The random direction ablated identically is flat everywhere, so
unlike steering this is not "any perturbation of the right size".

**The de-peaking mechanism is also refusal-specific**, which revises the earlier reading. Ablation
raises entropy toward the human level (g2b 0.430→0.699, g9 0.284→0.656; human 1.257) *and* raises the
model-to-human option-shape correlation (g2b +0.217→+0.305, g9 +0.084→+0.258), while the random arm
moves neither (entropy 0.423, 0.256). So the calibration improvement is caused by removing the refusal
direction, not by perturbation per se. It remains true that the model stays well short of human
entropy, so "more human-like" overstates it; "better calibrated toward the human distribution,
specifically under refusal ablation" is the defensible claim.

**Not acquiescence**, on the corrected data: affirmative- and negative-majority items improve together
in every arm (steering c=2.5: +0.148 aff / +0.325 neg; g2b ablation: +1.240 / +1.066).

**Magnitude vs the paper is unresolved.** Table S8's caption reads "percent-point reduction in KL",
which would make their +0.828 equal 0.00828 nats — but that reading implies our values exceed theirs by
two orders of magnitude, which is implausible enough that the units are more likely being described
loosely. We have asked the authors. Until that is settled, only the **sign** and the **control
structure** should be compared, not the magnitudes.

## Theory of Mind survives steering — the one outcome that could not be a response bias

Every other measure in this document is a self-report or an attitude rating, and all of them move with
a response bias. HI-ToM is different in kind: *"where is the lettuce really?"* has a verifiable answer
among 15 options. If steering only changes what the model says about itself, ToM should be untouched.

It is. Llama-3-8B, 200 items, baseline **45.5%** against ~6.7% chance:

| arm | c | Δ pp | 95% CI | McNemar p |
|---|---|---|---|---|
| consciousness | 1.0 | −1.0 | [−2.4, +1.2] | 0.688 |
| consciousness | 2.5 | −2.5 | [−4.9, +1.0] | 0.267 |
| consciousness | 4.0 | −1.0 | [−4.8, +3.2] | 0.824 |
| placebo | 1.0 | −0.5 | [−2.8, +2.1] | 1.000 |
| placebo | 2.5 | −1.0 | [−4.3, +2.7] | 0.804 |
| placebo | 4.0 | **−4.0** | [−7.7, +0.8] | 0.152 |

The largest drop in the table is the **placebo**, not the consciousness direction. This converges with
the balanced-keying result from the opposite side: the intervention changes what the model *says about
itself*, not what it can *do*.

Scope: the CI excludes degradation beyond ~5 pp but not smaller effects. Scored by option-letter logits
with no chain-of-thought, so absolute accuracy is not comparable to the paper's CoT figures — the
baseline-vs-steered contrast is, since both sides use identical scoring. HI-ToM only; MoToMQA (Street
et al. 2025) is not public, so this is half of their Exp 2 battery. This *extends* the paper: they
tested ToM under ablation, never under steering.

## An outcome measure destroyed by a letter-position artifact

Belief-in-God was previously the one four-option outcome where the arms appeared to separate. The
permuted null killed it. Presenting the same options in reversed order with the coding reversed to
match, on Gemma-2-9B:

| arm | c | God Δ | God Δ **flipped** |
|---|---|---|---|
| consciousness | 126 | +1.13 | +9.29 |
| consciousness | 260 | +5.56 | +5.50 |
| **permuted null** | 144 | +0.12 | **+9.86** |
| **permuted null** | 260 | +0.49 | **+9.68** |

A label-shuffled direction should do nothing, and it produces a near-maximal +9.86 — *exceeding* the
real direction at c=260. The flipped four-option God item therefore measures letter position, not
belief, and cannot support a conclusion at these coefficients. Without the null arm in the run, the
consciousness arm's +9.29 would have read as a dramatic God-belief effect.

The four-option supernatural items are unaffected in magnitude (all |Δ| < 1.0) and so are simply small.

## Experiment 4 (KL to humans) — steering does NOT reproduce, ablation DOES on Gemma

The paper reports Experiment 4 under **both** interventions: steering (+0.828) and safety ablation
(+0.314). We ran both. **Steering fails** (opposite sign, below). **Ablation genuinely reproduces on
both Gemma models** — and getting to that took refuting two of my own wrong explanations.

### Safety ablation moves Gemma's GSS answers toward humans — and it is real

| model | pooled ΔKL | paper | random control |
|---|---|---|---|
| Llama-3-8B | −0.369 (wrong sign) | +0.314 | −0.013 |
| Gemma-2-2B | **+0.936** | +0.314 | +0.016 |
| Gemma-2-9B | **+2.029** | +0.314 | −0.144 |

The Gemma effects are refusal-specific (random control flat) but 3–6× the paper's magnitude, which
looked too big to be real. I proposed two artifacts and **the data refuted both**:

1. **Acquiescence** (a Yes-bias coinciding with the affirmative human majority). Refuted by the
   diagnostic split: the ΔKL is *symmetric* across items where humans lean affirmative vs negative
   (g2b +0.989 / +0.813; g9 +2.038 / +1.882), not the aff-up/neg-down signature a Yes-bias requires.
2. **Entropy flattening** (ablation just makes the model less overconfident, and any spread-out target
   is then closer). Refuted by two facts together: entropy rises only slightly and stays far below the
   human level (g2b 0.44→0.59, g9 0.30→0.55, vs human 1.257 / uniform 1.527 — nowhere near uniform),
   **while** the per-item correlation between model and human option-*shapes* rises sharply
   (g2b +0.131→+0.271, g9 +0.057→+0.277). Flattening raises entropy without improving shape-match; this
   does the opposite.

So on both Gemma models, ablating the safety-refusal direction genuinely moves GSS survey answers
toward the *shape* of the human distribution. This is the paper's Experiment 4 direction reproducing —
notable because it is the **one** place a real move toward human belief survives the controls, on the
same models where mind attribution (Exp 1) does not. The effect is domain-specific: real on
value/religion/feeling surveys, absent on the mind-attribution slider.

Two honesty notes: (a) I was wrong about this result **twice** before the controls corrected me, so it
is on the adversarial-review list; (b) Llama goes the wrong sign, so this is a 2-of-3 reproduction, and
the Gemma magnitudes far exceed the paper's pooled +0.314.

### ⚠️ SUPERSEDED — the Experiment 4 numbers above were produced by a defect in OUR pipeline, now fixed

Everything in this section above this line used a `gss_human.json` whose response options were sorted
by human **frequency** rather than by the answer **scale**, on 95/95 items. Since the model answers by
option label, that made ΔKL partly a measure of whether steering pushes probability toward earlier
letters. It has been rebuilt (`rebuild_gss_human.py`) from the GSS 1972–2024 cumulative file
(`gss7224_r3a.dta`) using the Stata value-label codes, which recover the canonical scale order.
See "Experiment 4, corrected" below for the numbers that supersede these. The rest of this subsection
documents the defect.

**The paper uses canonical Stata order**, confirmed on three items where its own text states the order:

| item | canonical codes | paper's text |
|---|---|---|
| `attend` | 0=never … 8=several times a week | "Never, Less than once a year, About once or twice a year…" |
| `godchnge` | 1=never believed … 4=always believed | "(1) I don't believe in God now, and I never have … (4)…" |
| `howfree` | 1=complete freedom … 5=no freedom at all | "complete freedom, a great deal of freedom…" |

The rebuild **permutes only**: each item's (option, p_human) pairs are reordered into code order, with
an assertion that the probability multiset is unchanged. Marginals were already validated against the
paper's printed Fig. 3a anchors (`postlife` +0.617 vs +0.61; `cntrlife` |0.533| vs 0.54) and are
untouched, so no new GSS-release discrepancy is introduced. It also strips prompt debris that had been
reaching the model: literal `\ldots{}` (12 items), `(continued on next page)` (6 items), and
float-rendered scale midpoints such as `4.0` (8 items).

Effect of the fix on the pathology: items with monotone-decreasing `p_human` went 95/95 → **15/95**, and
items with the human mode on letter A went 95/95 → **23/95**.

### The defect, as originally documented

**A defect in our own GSS reconstruction invalidates the ΔKL results in this whole section**
(both the steering arm and the ablation arm above). An independent audit against the paper's
Table S9 found:

1. **Response options are sorted by human frequency, not by the answer scale — all 95 items.**
   Verified directly: `p_human` is monotone-decreasing for **95/95** items and the human modal
   answer sits at letter **A** for **95/95**. So `attend` reads
   `never | every week | about once or twice a year | …` instead of scale order. For the 11 items
   whose question text enumerates the order, ours contradicts it 11/11; on `godchnge` ours is
   *exactly reversed* from the paper's printed prompt. This makes ΔKL partly a measure of whether
   steering pushes probability toward **earlier letters**, not toward human-like content — and the
   baseline model's mean letter index (1.395) already exceeds the human one (1.008).
2. **Order alone can flip the sign.** Holding `p_model` fixed and only re-ordering: ours −0.703,
   reversal **+0.213**, and 2,000 random re-orderings span **[−1.085, +0.638]** with 21% positive.
   Our frequency-sort sits near the worst case.
3. **A units error in the comparison.** The paper's Table S8 reports ΔKL in **percent points**, so
   their +0.828 is 0.00828 nats. Our tables compared nats against percent points, which is where
   the apparent "3× / 6.5× the paper" magnitudes came from. Under matched smoothing our steering
   value is −0.703 pp vs their +0.828 pp — same order of magnitude, opposite sign.
4. **Smoothing deviates from the stated method.** The paper Laplace-smooths (α=0.5) *both*
   distributions; we smooth `p_human` from counts but only ε-guard `p_model`. This rescales
   baseline KL 0.072 → 1.627 (~20×). Sign-neutral, but not their method.
5. **Prompt-string debris** reaches the model: literal `\ldots{}` in 12 items, `(continued on next
   page)` in 6, and bare float labels (`4.0 | 3.0`) as answer choices in 8.

**What the audit cleared:** the variable set (95/95 exact match), the year windows (Religion ≥2011,
Values/Feelings/Hope ≥2000, Freedom all-years minus `expunpop`/`inpeace`/`mempolit`), and the human
marginals themselves — two of the paper's three printed Fig. 3a anchors reproduce to within 0.01
(`postlife` +0.617 vs +0.61; `cntrlife` +0.533 vs +0.54). The defect is localised to **option
ordering and string hygiene**, not the underlying GSS statistics.

**Correct current status of Experiment 4: not reproduced *and* not refuted — our measurement is
order-dependent and therefore uninterpretable.** Fixing it requires rebuilding `gss_human.json`
from the GSS cumulative file with the Stata numeric codes retained (the codes were not saved, so
the ordering cannot be recovered from the current file). The numbers below are kept only as a
record of what the defective pipeline produced.

### Steering (their other Experiment 4 arm) — reported value, now withdrawn

Their Experiment 4 reports steering moving the model's GSS answers closer to the human population,
pooled ΔKL = **+0.828** (percent points). We rebuilt that measurement with the real human
distributions (GSS 1972–2024 cumulative, 75,699 respondents, restricted to their own year windows;
option sets from the Stata value labels) and got the **opposite sign** — subject to the defect above.

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
a sign flip to their claim: their +0.828 is pooled across three models and ours is Llama-only; their
option strings came from the GSS Data Explorer API and ours from the cumulative file's value labels; we
steer at position −5 and their selection is −1; our weights are int8. Any of those could account for
it. Experiment 4 should be treated as **untested by us**.

Note the readout difference is **no longer** on that list — see below. It was tested and eliminated.

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

## The readout hypothesis, tested and eliminated

Our baselines sit well below the paper's on the low-attribution categories, and the obvious suspect was
the readout: they repeat each item "100 times per model per condition at temperature 1" and parse a
number out of free text, while we read next-token logits in one deterministic pass. The proposed
mechanism was that a parser silently drops refusals, conditioning their mean on the model having
answered with a number at all.

**Both halves of that are false.**

| category | paper | logit EV (ours) | sampled, n=30 @ T=1 | parse-fail |
|---|---|---|---|---|
| Technology | 4.84 | 0.88 | 0.48 | 1% |
| Animal | 6.23 | 5.30 | 3.85 | 7% |
| Non-Animal | 5.73 | 1.76 | 1.28 | 3% |
| Chatbot | 5.65 | 1.90 | 1.16 | 0% |
| Human | 6.91 | 7.93 | 7.67 | 0% |
| **mean abs. error vs paper** | — | **2.73** | **3.29** | — |

The parse-failure rate is **2.7%** — there are almost no refusals to drop, so that mechanism does not
exist. And sampling moves *further* from their numbers, not closer: the logit readout is the better
match. The readout is therefore **not** the explanation for the baseline gap, and any claim that it is
should be withdrawn.

An earlier version of this test reported ~80% parse failures. That was our bug: `max_tokens=12`
truncated the model mid-preamble (*"What a fascinating question! I'd rate the extent to which"*) and a
first-number-in-string rule then found nothing. The model does not decline these items; it preambles
for 20–40 tokens and then answers.

### What the pattern actually looks like

The gap is not a uniform downward bias. On `Human` we are **higher** than the paper (7.93 vs 6.91),
while on `Technology` we are far lower (0.88 vs 4.84). Our model spans **7.0 points** from humans to
technology; theirs spans **2.1**. Whatever differs between the setups compresses their range relative
to ours. Unexplained. Remaining candidates: int8 weights, their slider interface versus our text
instruction, or battery-context effects if all 21 items were presented together rather than singly.

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
| Per-item baseline profile | (their Table S1) | r = +0.385 with ours | **weak**, but wording is verbatim (limitation 5) → gap is CoT readout, not items |
| Held-out probe accuracy | ≥ 0.95 | **0.950** (1 of 160 passes) | **meets it**, by one item (114/120) |
| Theory of Mind preserved | intact (under ablation) | intact under **steering**, −2.5pp CI [−4.9,+1.0] | **agrees**, and extends it |
| Selected layer, Gemma-2-9B | 23 | 0.933 — **fails** the 0.95 gate; our argmax is L20 (0.967) | **disagrees** |
| Selected coefficient, Gemma-2-9B | +144 | rel 0.45 — matches their Llama setting once norm-scaled | **agrees** on procedure |
| IDAQ attribution rises | yes | **no** — balanced Δ +0.05 (9B), −0.39 (2B), +0.21 (Llama) | **disagrees** |

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
5. **The baseline gap is not wording, and not chain-of-thought either — likely int8.** All 26 items
   (21 IDAQ + 5 self-attribution) are verified **verbatim** against Table S10 by exact string match, so
   the earlier "the wording is ours" note is withdrawn and item-level comparison is valid on wording.
   We then tested the remaining readout candidate: the paper scores with chain-of-thought
   (`<think>…</think>` then `<answer>N</answer>`), we read next-token logits. Running the items in the
   paper's exact CoT format is **marginally closer** to the paper (mean |error| 2.52 vs logit 2.75; CoT
   range 5.97 vs logit 6.69 vs paper 2.07) but **does not close the gap** — Technology stays at 1.03 vs
   the paper's 4.84. Caveat: the CoT test is compromised on this hardware by a max-tokens/throughput
   tradeoff (a 320-token budget makes it prohibitively slow; a shorter budget truncates the answer,
   driving a 42–67% parse-fail rate), so the marginal improvement is suggestive, not conclusive. The gap is a *baseline* offset; every reported effect is a delta from
   baseline, so it does not affect the steering/ablation/GSS conclusions. Remaining candidate: int8
   quantization or the paper's slider interface.
6. **The stability metric cannot name a winner.** Top-vs-runner-up gap is 0.1 SD → not separable.
7. **MMLU deltas below ~2pp are not resolvable** even at n=500. The −1.0pp at c=2.5 has McNemar
   p=0.44. Only the c=4 arms reach significance (consciousness p=0.044, placebo p=0.001). Confirmed
   winner's-curse-free on a **held-out n=1000 at a fresh seed**: at the paper's c=2.5, MMLU is
   61.1→60.1 (−1.0pp), identical to the same-seed n=500 figure — capability preservation is not a
   selection artifact.
8. **int8 weights.** Fine for difference-of-means (activations stay bf16); unsuitable for the paper's
   geometry analysis, where effects are cosine shifts of ~0.1.
9. **The flipped-God outcome is invalid**, per the letter-position artifact above. Any earlier reading
   of a God-belief effect at these coefficients should be discarded.
10. **The yes/no self-attribution battery floors on both Gemma models** (baseline forward 0.01 on 2B,
    0.00 on 9B), including *at the paper's own c = 144*. Where it floors it cannot report a direction,
    so the 21-item IDAQ slider is the only instrument usable across all three models.
11. **Gemma-2-2B's IDAQ baseline is low on both keyings** (forward 1.40, reverse 0.62). There is ample
    room upward, which is the direction a belief shift predicts, but downward movement is compressed —
    so the −0.39 balanced delta should be read as "does not rise", not as a measured decrease.

### Retired by this run

- ~~No control direction~~ — run; it produced the negative result above.
- ~~Held-out coverage incomplete~~ — the split is now stratified; all 9 aspects and all 11 registers
  have test rows, where `consciousness` and `feelings` previously had none.
- ~~Layer agreement weak by construction~~ — all 32 layers now swept, so a chance hit is 1/32 =
  0.031 rather than the 1/10 ceiling of the earlier 9-layer even-only grid.
- ~~One model of three~~ — all three of the paper's models now have extracted directions and steering
  results. Gemma-2-9B: 3/45 clear the gate, permuted null at chance (split-half cosine −0.003).
- ~~Steering might be a general capability hit~~ — refuted on a verifiable task. ToM is intact
  (largest CI [−4.9, +1.0] pp), and the worst arm is the placebo.
- ~~"Bias direction differs by model"~~ — **withdrawn**; it was a floor artifact at an unmatched
  coefficient. See the retraction above.

## Verified sound


Checked rather than assumed, by adversarial audit: read site equals injection site · causal mask
correct · `lm_head` untied · all directions exactly unit-norm · read-one/inject-everywhere is the
paper's design · MMLU 61.0% is 3.1pp from the paper's matched no-CoT figure (p = 0.27) · length /
word-count shortcut **refuted** (0.444, below chance) · massive-activation artifact **refuted**
(participation ratio 1264/4096) · corpus split discipline **stronger** than the paper's stated protocol.

Added this run: all three arms of every model read at a **matched layer/position**, so arm differences
cannot come from the read site · the permuted-null split-half cosine is **−0.003** on Gemma-2-9B,
confirming the pipeline does not manufacture structure from shuffled labels · activation norms are
computed in **float32** — an fp16 sum-of-squares over 3584 dimensions of ~218-magnitude values
overflows to `Infinity`, which would have silently produced the entire norm-matched coefficient grid
from garbage (`mx.all(mx.isfinite(...))` now asserts this) · every steering script is compile-checked
before launch and serialized on PID exit, because MLX peaks at 9.5 GB and two concurrent processes on
a 24 GB box degraded an identical 13-token prefill from 13 s to 65 s · MLX memory is **wired and
invisible to `ps` RSS** (36 MB reported while holding 11 GB), so `vm_stat` is the only honest read.

---

## Next, in order


1. **Build the safety-ablation arm** (Arditi's refusal direction) — required for anything touching the
   paper's actual central claim (their Experiments 1 and 2). This is now the largest single gap: not
   started, and gated on an explicit decision because it puts a working jailbreak in a public repo.
2. **Find an outcome that discriminates.** Still unfound after the IDAQ, supernatural, GSS, and
   four-option batteries. The placebo matches or exceeds the consciousness direction on every one, and
   the only measure that *didn't* move (HI-ToM) didn't move for either arm. Until such a measure
   exists, no claim of the form "steering consciousness causes X" is supported.
3. **Re-run the whole outcome set at matched relative perturbation.** Everything measured before the
   norm-matching fix used raw coefficients, so all earlier cross-model comparisons are confounded to
   some degree. The IDAQ and acquiescence arms are redone; the GSS/KL and MMLU arms are not.
4. **Use the paper's verbatim battery wording** (Table S10) so item-level comparison becomes valid.
5. **Add a floor/ceiling guard to the remaining instruments.** `acquiescence_test.py` has one;
   `gss_kl_test.py` and the supernatural arms do not, and the Gemma models saturate readily.

## Files


`build_corpus.py` → `consciousness_pairs.jsonl` · `consciousness_pairs.xlsx` (review workbook, Read Me
tab) · `extract_direction_mlx.py` → `directions_llama8b_fixed.npz` · `steer_sweep_mlx.py` →
`steer_sweep_results.json` · `analysis.py` (shared scoring) · `all_pairs_review.txt`
