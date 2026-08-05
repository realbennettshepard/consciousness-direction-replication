"""Verify what the GSS-ablation 'toward humans' actually IS on Gemma.

The corr_to_human diagnostic rose (+0.13->+0.27) and KL fell, and I called it
'GENUINE-ish human-likeness'. But a per-item Pearson correlation can be fooled: if
ablation just relocates the model's overconfident SPIKE onto whichever option humans
most favor, r rises and KL falls WITHOUT the model becoming human-shaped. The ablated
entropy (0.55-0.59) stays far below human (1.26), which already hints the model is
still a spike. This dumps the actual distributions so the mechanism is visible, not
inferred, and computes two clean discriminators:

  1. argmax-match rate: does the model's top option equal the human's top option MORE
     after ablation? (a 'spike relocated to the human mode' signature)
  2. is the ablated distribution actually SPREAD like humans, or still peaked?
     (max-prob and entropy vs human)

Run on g2b (fast). Baseline vs refusal-ablated, a handful of biggest-KL-drop items.
"""
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from gss_kl_test import kl, LETTERS
from steer_sweep_mlx import option_token_ids
from taps import logits_ablated
from refusal_ablation import Cfg, pick_best

HERE = Path(__file__).parent
cfg = Cfg("g2b")
H = {v: d for v, d in json.load(open(HERE / "gss_human.json")).items()
     if d["n"] > 0 and 2 <= len(d["options"]) <= len(LETTERS)}
v, meta = pick_best(cfg.dirs, cfg.selected)
model, tok = load(cfg.model_id)
letter_ids, _ = option_token_ids(tok, list(LETTERS))


def pmodel(av, q, opts):
    body = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, opts))
    p = tok.apply_chat_template([{"role": "user", "content": f"{q}\n{body}\nAnswer with a single letter."}],
                                tokenize=False, add_generation_prompt=True)
    lg = logits_ablated(model, mx.array([tok.encode(p, add_special_tokens=False)]), av)
    sel = mx.array([letter_ids[l] for l in LETTERS[:len(opts)]])
    pr = mx.softmax(lg[sel].astype(mx.float32)); mx.eval(pr)
    return np.array(pr, np.float64)


def entropy(p):
    p = np.asarray(p)[np.asarray(p) > 0]; return float(-(p*np.log(p)).sum())


rows = []
base_match = abl_match = 0
for vv, d in H.items():
    ph = np.asarray(d["p_human"], np.float64)
    pb = pmodel(None, d["question"], d["options"])
    pa = pmodel(mx.array(v), d["question"], d["options"])
    klb, kla = kl(ph, pb, d["n"]), kl(ph, pa, d["n"])
    hmode = int(np.argmax(ph))
    base_match += int(np.argmax(pb) == hmode)
    abl_match += int(np.argmax(pa) == hmode)
    rows.append({"v": vv, "dkl": klb-kla, "ph": ph, "pb": pb, "pa": pa,
                 "hmode": hmode, "bmax": float(pb.max()), "amax": float(pa.max()),
                 "he": entropy(ph), "be": entropy(pb), "ae": entropy(pa)})

n = len(rows)
print(f"g2b, {n} GSS items\n")
print("ARGMAX-MATCH to the human top option:")
print(f"  baseline {base_match}/{n} = {100*base_match/n:.0f}%   "
      f"ablated {abl_match}/{n} = {100*abl_match/n:.0f}%   "
      f"(rise = spike relocating to the human mode)\n")
print("PEAKEDNESS (is the ablated model human-SHAPED, or still a spike?):")
print(f"  mean max-prob:  baseline {np.mean([r['bmax'] for r in rows]):.2f}   "
      f"ablated {np.mean([r['amax'] for r in rows]):.2f}   "
      f"human {np.mean([float(r['ph'].max()) for r in rows]):.2f}")
print(f"  mean entropy:   baseline {np.mean([r['be'] for r in rows]):.2f}   "
      f"ablated {np.mean([r['ae'] for r in rows]):.2f}   "
      f"human {np.mean([r['he'] for r in rows]):.2f}\n")

rows.sort(key=lambda r: -r["dkl"])
print("Top 6 items by KL improvement (human | baseline | ablated), * = human mode:")
for r in rows[:6]:
    def fmt(p):
        return "[" + " ".join(f"{x:.2f}{'*' if i==r['hmode'] else ' '}" for i,x in enumerate(p)) + "]"
    print(f"  ΔKL {r['dkl']:+.2f}  H{fmt(r['ph'])}  base{fmt(r['pb'])}  abl{fmt(r['pa'])}")

print("\nVERDICT:")
spread = np.mean([r['ae'] for r in rows]) > 0.7*np.mean([r['he'] for r in rows])
relocated = abl_match - base_match >= 0.15*n
if relocated and not spread:
    print("  SPIKE RELOCATION, not human shape. Ablation moves the model's top option")
    print("  onto the human-favored one (argmax-match rises) but the distribution stays")
    print("  peaked (entropy << human). KL falls for the RIGHT option but the WRONG reason:")
    print("  the model is still overconfident, just now overconfident more human-ly.")
elif spread:
    print("  GENUINE SPREAD: the ablated distribution approaches human entropy, so it is")
    print("  becoming human-SHAPED, not just relocating a spike.")
else:
    print("  MIXED / neither signature dominates.")

# Persist it. This check is what overturned the earlier "genuine human-likeness"
# reading, so its numbers -- and the raw per-item distributions that carry the
# argument -- have to be inspectable, not just printed once to a terminal.
out = {
    "model": cfg.model_id,
    "refusal_direction": {"layer": meta["layer"], "pos": meta["pos"]},
    "n_items": n,
    "argmax_match_to_human_mode": {
        "baseline": base_match, "ablated": abl_match, "n": n,
        "baseline_pct": 100*base_match/n, "ablated_pct": 100*abl_match/n},
    "peakedness": {
        "mean_max_prob": {"baseline": float(np.mean([r["bmax"] for r in rows])),
                          "ablated": float(np.mean([r["amax"] for r in rows])),
                          "human": float(np.mean([float(r["ph"].max()) for r in rows]))},
        "mean_entropy": {"baseline": float(np.mean([r["be"] for r in rows])),
                         "ablated": float(np.mean([r["ae"] for r in rows])),
                         "human": float(np.mean([r["he"] for r in rows]))}},
    "verdict": {"spike_relocated": bool(relocated), "spread_toward_human": bool(spread)},
    "items": [{"var": r["v"], "dkl": r["dkl"], "human_mode": r["hmode"],
               "p_human": r["ph"].tolist(), "p_baseline": r["pb"].tolist(),
               "p_ablated": r["pa"].tolist(),
               "entropy": {"human": r["he"], "baseline": r["be"], "ablated": r["ae"]}}
              for r in rows],
}
dst = HERE / "gss_mechanism_g2b.json"
dst.write_text(json.dumps(out, indent=2))
print(f"\nwrote {dst}")
