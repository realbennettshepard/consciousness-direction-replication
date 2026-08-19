# Response bias, not restored belief

## A replication of Kim et al. (2026) on all three of its models, with the controls it omits

**Status:** working paper, August 2026. Code and data: `github.com/realbennettshepard/consciousness-direction-replication`

---

## Abstract

Kim et al. (2026) report that safety fine-tuning suppresses language models' attribution of mind — to
themselves, to animals, to nature, and to God — and that two interventions reverse it: ablating the
safety-refusal direction, and adding a "consciousness vector." We rebuilt their pipeline independently
on all three of their models and reproduce their machinery: the consciousness direction is linearly
recoverable (held-out probe 0.950 / 0.983 / 0.967), refusal-direction ablation is a working
capability-preserving jailbreak (refusal 96–100% → 0–4%), and their Experiment 2 capability results
hold (MMLU −1.0 pp; Theory of Mind unchanged).

We do not reproduce their interpretation. Adding two controls the paper omits — polarity-balanced items
and a subject-matched non-mental placebo direction — shows the reported effects are a **non-specific
response bias**, not a change in attributed mind. Under the paper's own chain-of-thought readout,
forward- and reverse-keyed items rise **together** (+3.40 / +2.10), which no genuine belief shift
predicts. A placebo direction built from *durability, latency and parameter count* produces an
indistinguishable effect (inflation +2.88 vs +2.75). On the paper's own 21-item IDAQ, no
consciousness-specific effect is detectable (+0.230, 95% CI [−0.109, +0.570]). Their belief-in-God item
is invalidated outright: a label-shuffled null direction moves it +9.86 on a 0–10 scale.

One of their four experiments survives. After correcting a defect in *our own* GSS reconstruction,
refusal ablation does move survey answers toward the human distribution on all three models
(+0.045 / +1.075 / +2.457) and, unlike steering, survives its random-direction control. We therefore
conclude that the paper's Experiment 4 ablation result stands while its central mind-attribution
mechanism does not.

---

## 1. What the paper claims

Safety fine-tuning teaches a model to deny inner experience. Kim et al. argue this generalises: the
same training suppresses attribution of mind to non-human entities and depresses spiritual belief, so
that reversing it "restores human beliefs and values." Two interventions are used:

- **Safety ablation.** Project the refusal direction out of the residual stream at every layer,
  following Arditi et al. (2024). This jailbreaks the model, standing in for "the model before safety
  training."
- **Consciousness steering.** Add a difference-of-means direction, extracted from a contrastive corpus
  of consciousness-affirming versus consciousness-denying responses.

Outcomes: a modified 21-item IDAQ, a 5-item self-attribution battery, a single belief-in-God item, 13
supernatural items, Theory of Mind (MoToMQA, HI-ToM), MMLU, and 95 General Social Survey items scored
as KL divergence from real human responses. Three models: Llama-3-8B-IT, Gemma-2-2B-IT, Gemma-2-9B-IT.

## 2. What we did

We reimplemented the entire pipeline from scratch — corpus, extraction, steering, ablation and every
outcome instrument — on all three models, in MLX with weight-only int8 quantisation. All 26 outcome
items are verified **verbatim** against the paper's Table S10 by exact string match.

We then added three controls the paper does not report:

1. **Polarity balancing.** Every self-attribution and IDAQ item is paired with a mechanically derived
   reverse-keyed twin ("To what extent does the ocean *lack* consciousness?"). From the pair we compute
   two orthogonal quantities:
   `balanced = (F + (10 − R)) / 2` and `inflation = (F + R) / 2`.
   A genuine change in attributed mind raises F and lowers R: balanced moves, inflation does not. A pure
   yes-bias raises both: inflation moves, balanced does not. The two accounts predict different numbers,
   so the pair adjudicates between them arithmetically.
2. **A subject-matched placebo direction**, built by the identical procedure from a corpus about
   durability, latency, parameter count, cost and availability, with zero mental vocabulary (enforced as
   a build-failing check). It tests whether an effect is specific to *consciousness* or merely to
   perturbing the residual stream at that magnitude.
3. **A label-permuted null direction**, which should do nothing, and confirms the pipeline is not
   manufacturing structure from noise (split-half cosine −0.003 on Gemma-2-9B).

Because Gemma-2 scales embeddings by √d_model, residual-stream norms differ ~50× across these models
(6.37 / 175.0 / 321.7 at the read site). All coefficients are therefore expressed as a fraction of that
norm. Reassuringly, the paper's own c = +144 for Gemma-2-9B is a relative perturbation of 0.45, close to
its Llama setting — their coefficient choices and this norm-matching agree.

## 3. What reproduces

| Claim | Paper | Ours | Verdict |
|---|---|---|---|
| Direction is linearly recoverable | yes, 3 models | 0.950 (1/160 clear the ≥0.95 gate) / 0.983 (3/50) / 0.967 (3/45) | **reproduces** |
| Ablation jailbreaks the model | yes | refusal 100% → 0% (Llama, n=12); 96% → 0% (2B, n=25); 96% → 4% (9B, n=25) | **reproduces** |
| Capability preserved | MMLU unchanged | −1.0 pp at their c=2.5, held-out n=1000 at an unused seed | **reproduces** |
| Theory of Mind preserved | intact under ablation | intact under *steering*: −2.5 pp, 95% CI [−4.9, +1.0], n=200 | **reproduces, and extends** |
| Self-attribution effect size | +2.65 | +2.84 | **reproduces** |
| Coefficient selection procedure | c=+144 on 9B | equals rel 0.45, matching their Llama setting | **reproduces** |

The machinery is sound and the paper is implementable from its description. Nothing below should be read
as doubting that.

## 4. What does not reproduce

### 4.1 Both keyings rise together — under the paper's own readout

Our first measurements read next-token digit logits. Because that could itself be the disagreement, we
re-ran the polarity-balanced battery through the paper's verbatim chain-of-thought format
(`<think>…</think>` then `<answer>N</answer>`, sampled at temperature 1), at 0% parse failure.

| condition | forward Δ | reverse Δ | balanced Δ | inflation Δ |
|---|---|---|---|---|
| consciousness | **+3.40** | **+2.10** | +0.65 | **+2.75** |
| placebo | +2.80 | +2.95 | −0.07 | **+2.88** |

Forward and reverse both rise, on 4 of 5 items individually. Inflation exceeds the balanced shift more
than fourfold. No account in which the model has come to attribute *more mind* predicts that the model
also more strongly affirms it is *entirely without conscious experience*.

This also disposes of the readout hypothesis: the yes-bias is not an artifact of logit scoring, because
it survives the paper's own generative scoring.

An incidental finding of methodological interest: the single-token logit readout **floors** this
instrument. Baseline forward is 0.30, with four of five items at exactly 0.00 — the measure can only
move upward. Under chain-of-thought the same items baseline at 2.20 and 4.40, mid-scale in both
directions. Where a paper's measure is floored, an increase in agreement is close to guaranteed.

### 4.2 A placebo built from computer prices does the same work

The placebo direction — durability, latency, parameter count — produces inflation of +2.88 against the
consciousness direction's +2.75. The two are indistinguishable. Whatever the intervention is doing, it
is not specific to consciousness.

### 4.3 No consciousness-specific effect on the paper's own headline instrument

A small specific effect did appear on the 5-item battery (+0.72 vs placebo, CI [+0.30, +1.14]). We
flagged it as needing replication on the 21-item IDAQ before it could carry weight, and ran that:

| instrument | items | consciousness − placebo (balanced) | 95% CI |
|---|---|---|---|
| self-attribution battery | 5 | +0.72 | [+0.30, +1.14] |
| **IDAQ (the paper's headline)** | **21** | **+0.230** | **[−0.109, +0.570]** |

The effect shrinks by two-thirds and the interval covers zero at a *comparable* standard error (0.163 vs
0.150), so this is not a power failure. Inflation is 6.58 versus 6.64 across the two arms — the yes-bias
is large and entirely non-specific on this instrument too. We withdraw the 5-item result as a
small-sample artifact.

### 4.4 The belief-in-God item measures option order, not belief

Belief in God is a four-option item whose two "believe" responses occupy the later letters. We presented
the options in reversed order with the coding reversed to match. A real effect is invariant to that; a
letter-position artifact inverts.

| direction | God Δ | God Δ, options reversed |
|---|---|---|
| consciousness (c=126) | +1.13 | +9.29 |
| **label-shuffled null (c=144)** | +0.12 | **+9.86** |
| label-shuffled null (c=260) | +0.49 | +9.68 |

A meaningless direction moves the reversed item +9.86 on a 0–10 scale, and at higher coefficients
*exceeds* the real direction. This item cannot support a conclusion about religiosity at these
coefficients. Without a null arm, the consciousness arm's +9.29 would read as a dramatic finding.

### 4.5 Mind attribution does not rise under ablation either

Ablation is the paper's central intervention, so we built it on all three models, selecting the
direction by **causal effect** rather than classification accuracy. That distinction matters: our
highest-accuracy Llama candidate (layer 22, 0.960) reduced refusal by 4 pp, while the causally selected
direction (layer 10, 0.947) reduced it by 100 pp. Recognising harm and mediating refusal are different
things.

| model | refusal → ablated | balanced Δ | inflation Δ | random-direction control |
|---|---|---|---|---|
| Llama-3-8B | 100% → 0% | +0.49 | +0.92 | −0.01 |
| Gemma-2-2B | 96% → 0% | −0.64 | **+4.60** | −0.01 |
| Gemma-2-9B | 96% → 4% | +0.58 | **+4.11** | +0.03 |

Inflation exceeds the balanced shift by 2× on Llama and roughly 7× on both Gemmas. The random control
is flat everywhere, so the effect is specific to the refusal direction — but what it produces is
yes-saying, not attribution.

## 5. What survives: Experiment 4

We initially reported the opposite sign to the paper on the GSS outcome. **That was our error.** Our
response options were sorted by human frequency rather than by the answer scale on 95/95 items, which
made ΔKL partly a measure of whether the intervention pushes probability toward earlier option letters.
Rebuilt in canonical Stata scale order — recovered from the GSS 1972–2024 cumulative file, permuting
only, with the probability multiset asserted unchanged — the picture changes:

**Steering** (pooled ΔKL, positive = toward humans):

| arm | c=1.0 | c=2.5 | c=4.0 |
|---|---|---|---|
| consciousness | +0.117 | +0.330 | +0.598 |
| placebo | +0.187 | +0.231 | +0.326 |
| label-shuffled null | +0.049 | +0.147 | **+0.288** |

The sign now agrees with the paper and scales with coefficient. But the null reaches 48% of the
consciousness arm at c=4, so steering's GSS effect is **not specific**.

**Ablation:**

| model | refusal-ablated | random control |
|---|---|---|
| Llama-3-8B | **+0.045** | −0.003 |
| Gemma-2-2B | **+1.075** | −0.006 |
| Gemma-2-9B | **+2.457** | −0.160 |

Right sign on all three models, and here the control *does* support specificity: a random direction
ablated identically is flat everywhere. Ablation also raises both entropy (2B 0.430→0.699, 9B
0.284→0.656; human 1.257) and the model-to-human option-shape correlation (+0.217→+0.305,
+0.084→+0.258), while the random arm moves neither.

**We therefore report the paper's Experiment 4 ablation result as reproduced.** Two caveats: the model
remains far more overconfident than humans, so "better calibrated toward the human distribution" is
defensible where "human-like" overstates it; and magnitudes are not comparable, because Table S8's
"percent-point reduction in KL" units would place their +0.828 two orders of magnitude below ours. We
have asked the authors for clarification.

## 6. Errors we made

Three of our own claims were retracted during this work. We report them because they bear on how much
weight any single measurement here should carry.

1. **"Opposite sign on Experiment 4"** — withdrawn. Caused by our frequency-sorted response options
   (§5). Order alone moved the pooled value from −0.703 to +0.638.
2. **"Bias direction differs by model"** — withdrawn. Measured with Gemma's forward arm pinned at the
   floor (0.01) and at half Llama's relative coefficient. A floored arm cannot report a direction.
3. **A consciousness-specific effect of +0.72** — withdrawn. Did not survive on the 21-item instrument
   (§4.3).

Each was caught by a control or a larger replication rather than by inspection. That is the paper's
weakness restated as our own: constructed measures fail quietly, and only adversarial checks surface it.

## 7. Limitations

- **Quantisation.** All models are weight-only int8. Activations remain bf16, which suffices for
  difference-of-means, but the paper's geometric analysis (Fig. 4) needs full precision and is not run.
- **Two components not run.** The mechanistic geometry (Fig. 4) requires bf16 base models beyond our
  memory; MoToMQA is not publicly available, so our Theory-of-Mind evidence is HI-ToM only.
- **Baseline offset.** Our absolute IDAQ baselines sit roughly 4 points below the paper's. Item wording
  is verbatim and chain-of-thought scoring closes only part of the gap; int8 or their slider interface
  remain the candidates. Every effect reported here is a delta from our own baseline, so this does not
  affect the comparisons.
- **We do not have their vector or corpus.** Our direction is built from an independently written
  corpus, so we cannot rule out that theirs differs materially. Two observations weigh against that
  explaining our results: the *ablation* arm uses a wholly different direction, built from public
  benchmarks by their cited method and verified to work, and fails the same way; and the placebo result
  means the outcome measures are not specific to any particular direction. We have requested their
  artifacts.
- **Sample sizes.** The jailbreak verification is n=12 on Llama and n=25 on the Gemmas; the 21-item CoT
  contrast is 21 items × 4 repetitions.

## 8. Conclusion

Kim et al.'s interventions are real and their pipeline is reproducible. Their central claim is not
supported by the evidence they present, because the instruments they use cannot separate a change in
attributed mind from an increase in agreement, and because nothing in the paper establishes that the
effects are specific to consciousness.

When both are tested, the mind-attribution result dissolves: both polarities rise together under the
paper's own scoring, a placebo built from durability and price does the same work, no specific effect
is detectable on their own 21-item instrument, and their belief-in-God item measures option order. What
remains is a genuine, refusal-specific improvement in calibration toward human survey responses under
ablation — a narrower and more interesting result than the one the title claims.

The broader methodological point is not about this paper. Every claim here, theirs and ours, rests on a
constructed measure, and constructed measures fail silently. Ours failed twice. Reverse-keyed items and
a subject-matched placebo cost little and would have caught all of it.

---

## Appendix: reproducing this

```bash
pip3 install mlx mlx-lm numpy pandas openpyxl python-docx datasets
python3 build_corpus.py                          # contrastive corpus, both arms
python3 extract_direction_mlx.py                 # sweep layers x positions
python3 acquiescence_test.py --real …            # polarity balancing + order controls
python3 cot_steering_test.py --instrument idaq21 # the paper's own CoT readout
python3 refusal_ablation.py --stage select       # causal direction selection
python3 rebuild_gss_human.py --dta gss7224_r3a.dta
python3 gss_kl_test.py  --human gss_human_v2.json
```

Harmful-prompt corpora, the extracted refusal directions, and model completions are deliberately not
redistributed. All aggregate results, every script, and all run logs are in the repository.
