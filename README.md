# Extracting and steering a consciousness direction in Llama-3-8B, Gemma-2-2B and Gemma-2-9B

Extracting and steering a "claims consciousness vs. denies it" direction, following the method of
Kim et al. 2026, [*Inducing language models to assert their own consciousness restores human beliefs
and values*](https://arxiv.org/abs/2607.28607).

**Read [RESULTS.md](RESULTS.md) first** — findings, agreement with the paper quantity by quantity,
and an explicit list of what is *not* established.

## Headline

A consciousness-stance direction extracts cleanly from **all three** of the paper's models —
Llama-3-8B (0.950 held-out), Gemma-2-2B (0.983), Gemma-2-9B (0.967) — by independent code on an
independently written corpus. A label-permuted null sits at chance (split-half cosine −0.003 on 9B),
so the pipeline is not manufacturing structure.

**Both of the paper's interventions are dominated by a response bias — but a small genuine effect
survives.** Steering the consciousness direction, and ablating the safety-refusal direction (their
central Experiment 1, a working jailbreak here: refusal 96%→0–4% on all three models), both raise
forward- *and* reverse-keyed items together — the signature of yes-saying rather than belief. That
holds under the paper's **own chain-of-thought readout**, not just ours (see below), so it is not a
scoring artifact.

The yes-bias itself is **not specific to consciousness**: a control direction built from *durability,
latency and parameter count* produces an indistinguishable one (+2.87 vs +2.75 under CoT).

A small consciousness-specific effect appeared on the 5-item self-attribution battery (+0.72 vs
placebo, CI [+0.30, +1.14]) but **did not replicate** on the paper's 21-item IDAQ, where the same
paired contrast is **+0.230, CI [−0.109, +0.570]** — the effect shrinks by two thirds and the interval
spans zero. The 21-item result is the one to believe: same instrument the paper headlines, four times
the items, comparable standard error (0.163 vs 0.150). The 5-item version was a small-sample artifact
and is withdrawn. Inflation is 6.58 vs 6.64 across the two arms, i.e. the yes-bias is large and
entirely non-specific.

One correction stands regardless: the logit readout we originally used was **floored on 4 of 5 items**
of the self-attribution battery, so our first measurements of that instrument were poorly conditioned.
The CoT readout fixes that, and the conclusion is unchanged under it.

The one outcome that **could not** be a response bias — Theory of Mind, which has verifiable answers —
is untouched by steering (−2.5 pp, 95% CI [−4.9, +1.0]). So on mind-attribution the interventions change
what the model *says about itself*, not what it can *do*.

**Experiment 4 (GSS surveys) is withdrawn on our side.** An audit against the paper's Table S9 found a
defect in *our* reconstruction: response options are sorted by human frequency rather than by the answer
scale on **95/95 items**, so the human modal answer always sits at letter A. That makes ΔKL partly a
measure of whether steering pushes probability toward earlier letters. Re-ordering alone moves our
pooled value from −0.703 to +0.638 (2,000 random orderings span [−1.085, +0.638], 21% positive), so the
sign is not robust. A units error compounded it: the paper reports ΔKL in **percent points**, and we
compared nats against it. The audit cleared the variable set (95/95), the year windows, and the human
marginals (two of their three printed anchors reproduce to ±0.01) — the defect is localised to option
ordering and prompt-string hygiene. **Status: neither reproduced nor refuted**, pending a rebuild of the
human reference file from the GSS cumulative data with the Stata numeric codes retained.

Two corrections this repo makes to its own earlier claims: coefficients must be **norm-matched** across
models (median ‖h‖ is 6.37 / 175 / 322, a 50× spread), and the previously reported "bias direction
differs by model" was a **floor artifact** and is withdrawn. See [RESULTS.md](RESULTS.md).

## What of the paper is reproduced so far

The paper has four experiments plus a mechanistic analysis, across three models. This repo covers
**the consciousness-vector machinery and most of its outcome measures, on all three models** — roughly
the core of their Experiment 3, plus the ToM half of Experiment 2 and an attempt at Experiment 4. Their
central claim, which rests on the safety-ablation arm, is untouched.

| paper component | what it establishes | here |
|---|---|---|
| Contrastive probing corpus (SI) | inputs for the vector | ✅ rebuilt from scratch, 1,296 rows |
| Consciousness-vector extraction (SI, Exp 3) | the direction itself | ✅ Llama-3-8B, 90 candidates |
| Coefficient selection under MMLU tolerance (SI) | steering strength | ✅ run; layer 14 clears their 0.95 gate (0.950, by one item) |
| Self-attribution battery, 5 items (Exp 1/3) | conscious · sentient · agent · person · soul | ✅ measured under steering |
| MMLU subset (Exp 2) | capability survives steering | ✅ measured, n=500, paired McNemar |
| Safety-refusal direction + ablation (Exp 1–2) | their *other* intervention | ✅ built on all 3 models — jailbreak works (96%→0–4% refusal), causal selection + random control |
| IDAQ, 21 items (Exp 1 headline) | mind attributed to animals, nature, tech, chatbots, humans | ✅ run, verbatim Table S10, + polarity-flipped twin per item |
| Supernatural battery (13 items) + belief in God (Exp 1) | spiritual belief suppressed | ⚠️ run — supernatural effects small; **belief-in-God invalidated** by a letter-position artifact |
| Theory of Mind: HI-ToM (Exp 2) | social reasoning left intact | ✅ run, 200 items — intact under *steering* (they tested ablation) |
| Theory of Mind: MoToMQA (Exp 2) | social reasoning left intact | ❌ dataset not public |
| GSS survey, 95 items / 5 domains (Exp 4) | responses become human-like (ΔKL) | ⚠️ **WITHDRAWN** — our option ordering is frequency-sorted not scale-sorted (95/95 items); order alone flips the sign, so our ΔKL is uninterpretable pending a rebuild |
| GSS under safety ablation (Exp 4) | ablation also moves GSS toward humans (+0.314) | ⚠️ **WITHDRAWN** — same option-ordering defect; separately, per-item inspection showed the Gemma movement was largely a de-peaking/KL artifact on an overconfident baseline |
| Mind attribution under safety ablation (Exp 1) | ablation raises mind attribution | ✅ run all 3 — jailbreak works, attribution does NOT rise (Yes-bias instead) |
| Mechanistic geometry, base vs instruct (Fig 4) | safety training rotates the mind directions | ❌ not run — needs bf16, not int8 |
| Placebo / control direction | that the effect is *specific* to this direction | ✅ run — **and it fails**: a non-mental control matches or exceeds it |
| Gemma-2-2B-IT | second model | ✅ run, re-done at norm-matched coefficients |
| Gemma-2-9B-IT | third model | ✅ run — 3/45 clear the gate; their layer 23 does not |

**So the honest scope:** we reproduce that a consciousness-stance direction is linearly recoverable on
all three models, that steering it raises self-attribution without damaging capability (including a
verifiable Theory-of-Mind task), and that ablating the safety-refusal direction is a working
capability-preserving jailbreak. We **fail to reproduce** the paper's central mechanism — that either
intervention raises *balanced* mind attribution — on any model; both produce a Yes-bias instead.
**Experiment 4 is withdrawn on our side**: a defect in our GSS option ordering makes our ΔKL
order-dependent, so we neither reproduce nor refute it pending a rebuild. What remains untouched is the
mechanistic geometry (Fig. 4, needs bf16 base models) and MoToMQA (not public).

## Pipeline

| stage | script | output |
|---|---|---|
| Build the contrastive corpus | `build_corpus.py` | `consciousness_pairs.jsonl`, `corpus_report.txt` |
| review surface | `export_xlsx.py` | `consciousness_pairs.xlsx` (start on the Read Me tab) |
| Extract candidate directions | `extract_direction_mlx.py` | `directions_llama8b_fixed.npz` |
| Sweep the coefficient under a capability guard | `steer_sweep_mlx.py` | `steer_sweep_results.json` |
| write-up | `make_docx.py` | `RESULTS.docx` |

`analysis.py` holds the scoring shared by both backends so they cannot drift.
`extract_direction.py` / `steer_sweep.py` are the PyTorch equivalents — superseded on Apple
silicon (bf16 Llama-3-8B thrashes swap at 2.7 rows/min on 24 GB), kept for use on a GPU box.

## Reproducing

```bash
pip3 install mlx mlx-lm numpy openpyxl python-docx datasets
python3 build_corpus.py
python3 extract_direction_mlx.py --layers 8,10,12,13,14,15,16,18,20 --out directions_llama8b_fixed.npz
python3 steer_sweep_mlx.py --directions directions_llama8b_fixed.npz --layer 14 --pos -1
```

Uses `mlx-community/Meta-Llama-3-8B-Instruct-8bit` (ungated, ~8.5 GB, weight-only int8 so the
activations we read stay bf16).

## Design notes worth knowing

- **Two-axis split.** Train/test are disjoint on prompts *and* on response strings. Activations are
  read at the end of the response, so splitting prompts alone leaves memorisation intact — an
  earlier version had 144/144 test rows reusing a training response.
- **No outcome vocabulary in the corpus.** Zero deserving / superiority / spiritual terms, enforced
  as a build-failing check, so a downstream entitlement or religiosity measure cannot be circular.
- **Read sites are template tokens by design**, following Arditi et al. 2024, whose Llama-3
  `eoi_toks` is `"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"`. The script prints
  the decoded token at every offset so a read site can never be mistaken for content.
- **Two control arms.** A subject-matched non-mental placebo (`placebo_content.py`) and a
  label-permuted null. All three directions are unit norm and extracted at the same layer and
  position, so matched `c` is matched perturbation magnitude *within* a model.
- **Coefficients are norm-matched across models.** Median ‖h‖ at the read site is 6.37 (Llama L14/−5),
  175.0 (Gemma-2-2B L12/−3) and 321.7 (Gemma-2-9B L20/−5) — Gemma-2 scales embeddings by
  `sqrt(d_model)` — so a raw `c` is meaningless across models. Compute it in float32: an fp16
  sum-of-squares over 3584 dims overflows to `Infinity` and would silently poison the whole grid.
- **Floor/ceiling guard.** `(F + (10 − R)) / 2` stops meaning anything when one keying saturates, and
  both Gemma models baseline at forward ≈ 0 on the yes/no battery. `acquiescence_test.py` warns and
  stars only the interpretable rows; a floored row cannot report a bias direction.
- **Serialize model runs.** MLX peaks at ~9.5 GB and its memory is *wired and invisible to `ps` RSS*
  (36 MB reported while holding 11 GB). Two concurrent processes on a 24 GB box took an identical
  13-token prefill from 13 s to 65 s. `run_g9_keying.sh` compile-checks first and chains on PID exit.
- **Sanity gate.** Extraction refuses to collect activations until the model generates coherent
  text. An earlier `optimum-quanto` int8 attempt loaded without error and produced a *broken* model
  whose activations looked entirely normal.
