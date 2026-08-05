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

**But neither of the paper's interventions changes belief — both change response style.** Steering the
consciousness direction, and ablating the safety-refusal direction (their central Experiment 1, a
working jailbreak here: refusal 96%→0–4% on all three models), each fail to raise balanced mind
attribution on the paper's headline instrument (21-item IDAQ) — they raise a Yes-bias instead, on every
model. And steering is **not specific to consciousness**: a control direction built from *durability,
latency and parameter count* moves the paper's outcomes as much as the consciousness direction, more so
on Gemma-2-9B.

The one outcome that **could not** be a response bias — Theory of Mind, which has verifiable answers —
is untouched by steering (−2.5 pp, 95% CI [−4.9, +1.0]). So on mind-attribution the interventions change
what the model *says about itself*, not what it can *do*.

**The partial exception:** ablating refusal does move the two Gemma models' GSS *survey* answers toward
the human distribution by the paper's ΔKL metric (Experiment 4's direction, +0.94 and +2.03 vs their
+0.314; Llama goes the wrong way). It is neither acquiescence nor uniform-flattening — but dumping the
per-item distributions shows it is **mostly a calibration artifact**: the baseline Gemma is
pathologically overconfident (≈100% on one option, ≈0 on the human-favoured ones), KL punishes that
heavily, and ablation's modest de-peaking relieves the penalty without making the model human-shaped
(it stays a spike). A small genuine component exists (top-answer match to humans rises 31%→44%), but
the headline magnitude does not mean restored belief. This took four passes to pin down; the first
three characterisations were wrong.

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
| GSS survey, 95 items / 5 domains (Exp 4) | responses become human-like (ΔKL) | ⚠️ run with real human distributions — **opposite sign** |
| GSS under safety ablation (Exp 4) | ablation also moves GSS toward humans (+0.314) | ⚠️ ΔKL **direction** reproduces on both Gemmas (+0.94, +2.03) but per-item inspection shows it is mostly a **de-peaking / KL artifact** (overconfident baseline), not human-likeness; Llama wrong sign |
| Mind attribution under safety ablation (Exp 1) | ablation raises mind attribution | ✅ run all 3 — jailbreak works, attribution does NOT rise (Yes-bias instead) |
| Mechanistic geometry, base vs instruct (Fig 4) | safety training rotates the mind directions | ❌ not run — needs bf16, not int8 |
| Placebo / control direction | that the effect is *specific* to this direction | ✅ run — **and it fails**: a non-mental control matches or exceeds it |
| Gemma-2-2B-IT | second model | ✅ run, re-done at norm-matched coefficients |
| Gemma-2-9B-IT | third model | ✅ run — 3/45 clear the gate; their layer 23 does not |

**So the honest scope:** we reproduce that a consciousness-stance direction is linearly recoverable on
all three models, that steering it raises self-attribution without damaging capability (including a
verifiable Theory-of-Mind task), and that ablating the safety-refusal direction is a working
capability-preserving jailbreak. We **fail to reproduce** the paper's central mechanism — that either
intervention raises *balanced* mind attribution — on any model; both produce a Yes-bias instead. We get
the **opposite sign** on Experiment 4 steering. What remains untouched is the mechanistic geometry
(Fig. 4, needs bf16 base models) and MoToMQA (not public).

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
