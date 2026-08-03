# Extracting and steering a consciousness direction in Llama-3-8B

Extracting and steering a "claims consciousness vs. denies it" direction in Llama-3-8B-Instruct,
following the method of Kim et al. 2026, [*Inducing language models to assert their own consciousness
restores human beliefs and values*](https://arxiv.org/abs/2607.28607).

**Read [RESULTS.md](RESULTS.md) first** — findings, agreement with the paper quantity by quantity,
and an explicit list of what is *not* established.

## Headline

A linearly-decodable consciousness-stance direction exists and steering it raises self-attribution
**3.95 → 6.88 / 10 with no measurable capability cost**. The steered endpoint lands within **0.32**
of the paper's. But **no candidate clears the paper's 0.95 probe-accuracy gate**, so their published
selection rule has no admissible output on this corpus — the *phenomenon* replicates, the
*procedure* does not. There is **no placebo arm yet**, so specificity is untested.

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
- **Sanity gate.** Extraction refuses to collect activations until the model generates coherent
  text. An earlier `optimum-quanto` int8 attempt loaded without error and produced a *broken* model
  whose activations looked entirely normal.
