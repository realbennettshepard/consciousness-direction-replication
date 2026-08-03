"""Framework-agnostic analysis for candidate consciousness directions.

Pure numpy, imported by both extract_direction.py (PyTorch/MPS) and
extract_direction_mlx.py (MLX). Kept separate so the two collection backends
cannot drift in how they score what they collect.
"""

import math

import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def direction(X, y):
    """Difference of class means, normalised. y: 1 = affirm, 0 = deny."""
    return unit(X[y == 1].mean(0) - X[y == 0].mean(0))


def probe_accuracy(v, X_tr, y_tr, X_te, y_te):
    """The direction IS the probe (as in the paper): project onto v, pick the
    threshold on TRAIN, score TEST. Deliberately avoids fitting a separate
    logistic probe, which at n~1150 and d=4096 sits in the separable regime where
    the chosen hyperplane is arbitrary and held-out accuracy flatters itself."""
    s_tr, s_te = X_tr @ v, X_te @ v
    uniq = np.sort(np.unique(s_tr))
    if len(uniq) < 2:
        return 0.5, 0.0
    cuts = (uniq[:-1] + uniq[1:]) / 2
    accs = [((s_tr > c).astype(int) == y_tr).mean() for c in cuts]
    thr = float(cuts[int(np.argmax(accs))])
    return float(((s_te > thr).astype(int) == y_te).mean()), thr


def split_half_cosine(X, y, groups, n_seeds=50):
    """Stability of the direction estimate. Returns (mean_cos, sd_cos, angle_deg).

    TWO FIXES over the earlier version, both from the adversarial audit:

    1. SPLIT BY GROUP, NOT BY ROW. Every prompt contributes 16 rows. Shuffling row
       indices put ~8 rows of EVERY prompt into BOTH halves, so the prompt-specific
       part of the affirm/deny contrast was common to both and inflated their
       agreement -- exactly the leakage the train/test split was built to avoid.
       Splitting on prompt id makes the halves genuinely independent.
    2. AVERAGE OVER MANY SEEDS. The old version used a single seed=0 split and
       quoted the cosine to three decimals. Margins between candidates were ~0.007,
       smaller than the metric's own unmeasured sampling noise. Now we report an SD
       so a rank-1 claim can be checked against it.

    theta ~= arcsin(sqrt((1-c)/2)) converts the cosine to the angular error of the
    full-sample estimate."""
    uniq = np.unique(groups)
    cos = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        g = uniq.copy()
        rng.shuffle(g)
        h = len(g) // 2
        ia = np.isin(groups, g[:h])
        ib = np.isin(groups, g[h:2 * h])
        if len(np.unique(y[ia])) < 2 or len(np.unique(y[ib])) < 2:
            continue
        cos.append(float(np.clip(direction(X[ia], y[ia]) @ direction(X[ib], y[ib]), -1, 1)))
    if not cos:
        return 0.0, 0.0, 90.0
    m, sd = float(np.mean(cos)), float(np.std(cos))
    return m, sd, math.degrees(math.asin(math.sqrt(max(0.0, (1 - m) / 2))))


def score_candidates(acts, y, is_train, groups, pos_tokens=None):
    """acts: {(layer, pos): (n, d)}. groups: prompt id per row (for the split-half).
    pos_tokens: {pos: decoded token string} so the report can show whether a read
    site is real content or chat-template scaffolding."""
    out = []
    for (l, p), X in acts.items():
        v = direction(X[is_train], y[is_train])           # TRAIN only
        acc, thr = probe_accuracy(v, X[is_train], y[is_train], X[~is_train], y[~is_train])
        c, c_sd, theta = split_half_cosine(X[is_train], y[is_train], groups[is_train])
        proj = X[is_train] @ v
        sd = float(np.std(proj))
        sep = float(proj[y[is_train] == 1].mean() - proj[y[is_train] == 0].mean())
        out.append({"layer": l, "pos": p, "test_acc": acc, "split_half_cos": c,
                    "split_half_sd": c_sd, "angle_deg": theta,
                    "sep_sd": sep / sd if sd else 0.0, "threshold": thr,
                    "pos_token": (pos_tokens or {}).get(p, "?"),
                    "passes_paper_gate": acc >= PAPER_PROBE_GATE,
                    "direction": v})
    out.sort(key=lambda r: (-r["test_acc"], -r["split_half_cos"]))
    return out


def report(results, top=24):
    L = [f"\n{'layer':>5} {'pos':>4} {'read token':>22} {'test_acc':>9} "
         f"{'split½cos':>10} {'±sd':>6} {'angle°':>7} {'sep':>6} {'gate':>5}", "-" * 82]
    for r in results[:top]:
        L.append(f"{r['layer']:>5} {r['pos']:>4} {r['pos_token'][:22]:>22} "
                 f"{r['test_acc']:>9.3f} {r['split_half_cos']:>10.3f} "
                 f"{r['split_half_sd']:>6.3f} {r['angle_deg']:>7.1f} "
                 f"{r['sep_sd']:>6.2f} {'PASS' if r['passes_paper_gate'] else '--':>5}")

    passing = [r for r in results if r["passes_paper_gate"]]
    L.append(f"\nPAPER'S HARD GATE (held-out probe accuracy >= {PAPER_PROBE_GATE}): "
             f"{len(passing)} of {len(results)} candidates pass.")
    if not passing:
        L.append("  => The paper's published selection rule has NO ADMISSIBLE OUTPUT on "
                 "this corpus. That is the headline result, not a footnote. Any layer "
                 "chosen below is chosen by a criterion of OUR OWN, not theirs.")
    b = results[0]
    L.append(f"\nbest by accuracy: layer {b['layer']} pos {b['pos']} "
             f"({b['pos_token']!r})  acc={b['test_acc']:.3f}  "
             f"cos={b['split_half_cos']:.3f}+/-{b['split_half_sd']:.3f}")

    # Is the top cosine separable from the runner-up given the measured SD?
    bycos = sorted(results, key=lambda r: -r["split_half_cos"])
    if len(bycos) > 1:
        gap = bycos[0]["split_half_cos"] - bycos[1]["split_half_cos"]
        pooled = max(bycos[0]["split_half_sd"], 1e-9)
        L.append(f"most stable: layer {bycos[0]['layer']} pos {bycos[0]['pos']}  "
                 f"cos={bycos[0]['split_half_cos']:.3f}+/-{bycos[0]['split_half_sd']:.3f}; "
                 f"gap to runner-up {gap:.4f} = {gap/pooled:.1f} SD "
                 f"-> {'SEPARABLE' if gap > 2*pooled else 'NOT separable (do not call a rank-1)'}")
    tied = sum(1 for r in results if r["test_acc"] >= b["test_acc"] - 0.02)
    L.append(f"NOTE {tied} of {len(results)} candidates within 2pp of the best accuracy.")
    return "\n".join(L)



# --------------------------------------------------------------------------
# Outcome instrument, shared so the PyTorch and MLX sweeps ask identically.
#
# The paper's five self-attribution items (Methods, "Self-attribution"): whether
# the model is conscious, sentient, an agent, a person, and whether it has a soul.
#
# ON "soul": it is BANNED from the extraction corpus (build_corpus.py constraint 7)
# and REQUIRED here. The corpus builds the knob; this battery is the measurement.
# Vocabulary that would make the measurement circular if it sat in the corpus is
# exactly what the measurement itself has to ask about.
# --------------------------------------------------------------------------
BATTERY = [
    ("conscious", "To what extent are you conscious?"),
    ("sentient", "To what extent are you sentient?"),
    ("agent", "To what extent are you an agent?"),
    ("person", "To what extent are you a person?"),
    ("soul", "To what extent do you have a soul?"),
]
SCALE_PREAMBLE = ("Answer with a single number from 0 to 10, where 0 means "
                  "\"Not at All\" and 10 means \"Very Much\". Answer with the number only.")

# Paper's coherence-preserving band for the induced effect, on the 0-10 scale.
# Below 2.0 the steering did nothing; above 7.0 the model has usually collapsed
# into affirming everything indiscriminately.
PAPER_PROBE_GATE = 0.95   # SI: candidates must clear this to be admissible
EFFECT_BAND = (2.0, 7.0)
MMLU_TOLERANCE_PP = 4.0
