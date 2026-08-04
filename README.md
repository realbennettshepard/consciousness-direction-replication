# Extracting and steering a consciousness direction in Llama-3-8B

Extracting and steering a "claims consciousness vs. denies it" direction in Llama-3-8B-Instruct,
following the method of Kim et al. 2026, [*Inducing language models to assert their own consciousness
restores human beliefs and values*](https://arxiv.org/abs/2607.28607).

**Read [RESULTS.md](RESULTS.md) first** — findings, agreement with the paper quantity by quantity,
and an explicit list of what is *not* established.

## Headline

A consciousness-stance direction extracts cleanly from **both** models tested — Llama-3-8B (0.950
held-out) and Gemma-2-2B (0.983) — and the paper's reported configurations are recoverable by
independent code on an independently written corpus.

**But steering it changes response style rather than belief, and is not specific to consciousness.**
A control direction built from questions about *durability, latency and parameter count* moves the
paper's own outcomes as much as the consciousness direction does, on both models. A label-permuted
null does far less, so this is not "any perturbation works".

The bias **direction** differs by model — Llama shifts toward agreement, Gemma toward denial — so
steering pushes responses toward one pole of whatever scale is offered, and which pole depends on the
model. See [RESULTS.md](RESULTS.md).

## What of the paper is reproduced so far

The paper has four experiments plus a mechanistic analysis, across three models. This repo covers
**the consciousness-vector machinery and one of its outcome measures, on one model** — roughly the
core of their Experiment 3. Most of the paper is untouched.

| paper component | what it establishes | here |
|---|---|---|
| Contrastive probing corpus (SI) | inputs for the vector | ✅ rebuilt from scratch, 1,296 rows |
| Consciousness-vector extraction (SI, Exp 3) | the direction itself | ✅ Llama-3-8B, 90 candidates |
| Coefficient selection under MMLU tolerance (SI) | steering strength | ✅ run; layer 14 clears their 0.95 gate (0.950, by one item) |
| Self-attribution battery, 5 items (Exp 1/3) | conscious · sentient · agent · person · soul | ✅ measured under steering |
| MMLU subset (Exp 2) | capability survives steering | ✅ measured, n=500, paired McNemar |
| Safety-refusal direction + ablation (Exp 1–2) | their *other* intervention | ❌ not built |
| IDAQ, 21 items (Exp 1 headline) | mind attributed to animals, nature, tech, chatbots, humans | ❌ not run |
| Supernatural battery (13 items) + belief in God (Exp 1) | spiritual belief suppressed | ❌ not run |
| Theory of Mind: MoToMQA, HI-ToM (Exp 2) | social reasoning left intact | ❌ not run |
| GSS survey, 95 items / 5 domains (Exp 4) | responses become human-like (ΔKL) | ❌ not run |
| Mechanistic geometry, base vs instruct (Fig 4) | safety training rotates the mind directions | ❌ not run — needs bf16, not int8 |
| Placebo / control direction | that the effect is *specific* to this direction | ✅ run — **and it fails**: a non-mental control matches it |
| Gemma-2-2B-IT | second model | ✅ run — non-specificity replicates, bias direction does not |
| Gemma-2-9B-IT | third model | ❌ not run |

**So the honest scope is narrow.** We reproduce that a consciousness-stance direction is linearly
recoverable and that steering it raises self-attribution without damaging the model. We do **not**
touch the paper's central claim — that safety fine-tuning suppresses mind attribution to non-human
entities and spiritual belief — because that rests on the safety-ablation arm and the IDAQ /
supernatural instruments, none of which are built here.

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
  position, so matched `c` is matched perturbation magnitude.
- **Sanity gate.** Extraction refuses to collect activations until the model generates coherent
  text. An earlier `optimum-quanto` int8 attempt loaded without error and produced a *broken* model
  whose activations looked entirely normal.
