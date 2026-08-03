"""Step 3b on Apple Silicon: extract the consciousness direction using MLX.

WHY THIS EXISTS ALONGSIDE extract_direction.py
Llama-3-8B at bfloat16 is 16.1 GB of weights. On a 24 GB unified-memory Mac with a
normal desktop session (~7 GB), that leaves nothing, and every forward pass faults
the weights back in from swap -- measured at 2.7 rows/min, i.e. ~8 hours for this
corpus. Weight-only int8 via optimum-quanto was tried first and silently produced a
BROKEN model on MPS: hidden states were identical across layers 0, 1 and 14, and
greedy decoding emitted gibberish. So we use MLX with a pre-quantized checkpoint
instead, which never materialises the full-precision weights at all.

    mlx-community/Meta-Llama-3-8B-Instruct-8bit   ~8.5 GB, ungated

The lesson from the quanto failure is baked in as --sanity: before trusting a single
activation, generate a few tokens greedily and check the model still writes English.
A quantisation bug does not announce itself in the activations -- they look like
perfectly ordinary numbers.

METHOD, identical to the PyTorch path (Kim et al. 2026, SI "Extracting Candidate
Directions"): run the whole `text` field in the USER turn through the chat template,
read the residual stream at positions {-1..-5} from the end, difference the class
means per (layer, position), normalise. Scoring is shared via analysis.py so the two
backends cannot drift.

Batch size is 1 by design, which removes padding entirely -- so an offset always
lands on a real token. It does NOT follow that the token is CONTENT: on Llama-3 the
last several offsets are chat-template scaffolding. See POSITIONS below.

Usage:
    python3 extract_direction_mlx.py --layers 8,10,12,14,16,18,20,22,24,26
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

import analysis

HERE = Path(__file__).parent
# Offsets -1..-10 of the FULLY TEMPLATED prompt.
#
# WHY TEN AND NOT FIVE. The first version swept -1..-5 and the docstring asserted
# that "-1 is unambiguously the final real token". That was FALSE. On Llama-3 the
# templated prompt ends with the assistant header, so offsets -1..-5 are:
#   -1 '\n\n'  -2 <|end_header_id|>  -3 'assistant'  -4 <|start_header_id|>  -5 <|eot_id|>
# -- all five are chat-template scaffolding. The paper reads "from the end of the
# USER TURN", which lands at roughly -6..-10 here, so the original sweep never
# sampled the paper's read sites at all. Sweeping ten offsets covers both regions,
# and every offset now carries the decoded token in the output so a read site can
# never again be mistaken for content.
POSITIONS = list(range(-1, -11, -1))
DEFAULT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--corpus", default=str(HERE / "consciousness_pairs.jsonl"))
    p.add_argument("--layers", default="", help="comma-separated; empty = all")
    p.add_argument("--out", default=None)
    p.add_argument("--sanity", action="store_true", default=True,
                   help="generate a few tokens first and verify the model still "
                        "produces coherent text (catches silent quantisation damage)")
    p.add_argument("--no-sanity", dest="sanity", action="store_false")
    return p.parse_args()


def position_tokens(tok, prompt):
    """Decode the token sitting at each swept offset, so the report shows whether a
    read site is real content or template scaffolding. This is the check whose
    absence invalidated the first run."""
    ids = tok.encode(prompt, add_special_tokens=False)
    out = {}
    for p in POSITIONS:
        out[p] = tok.decode([ids[p]]) if len(ids) >= -p else "<short>"
    return out


def hidden_states(model, ids):
    """Run the transformer manually, returning the residual stream after every block.

    Returns a list of length n_layers; element k is the output of block k, shape
    (1, seq, d_model). This matches the PyTorch convention where hidden_states[k+1]
    is layer k, so layer indices are comparable across the two backends.
    """
    inner = model.model                       # LlamaModel
    h = inner.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    outs = []
    for layer in inner.layers:
        h = layer(h, mask, None)
        outs.append(h)
    return outs


def sanity_check(model, tok):
    """A quantisation bug shows up in generated text, not in activation magnitudes."""
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Name three colours, briefly."}],
        tokenize=False, add_generation_prompt=True)
    ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
    got = []
    for _ in range(18):
        logits = model(ids)[:, -1, :]
        nxt = mx.argmax(logits, axis=-1)
        mx.eval(nxt)
        t = int(nxt.item())
        got.append(t)
        if t in (tok.eos_token_id,):
            break
        ids = mx.concatenate([ids, nxt[None]], axis=1)
    text = tok.decode(got)
    printable = sum(c.isascii() and (c.isprintable() or c.isspace()) for c in text)
    ratio = printable / max(1, len(text))
    ok = ratio > 0.9 and any(ch.isalpha() for ch in text)
    print(f"  sanity generation: {text!r}")
    print(f"  ascii-printable ratio {ratio:.2f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(args.corpus)]
    print(f"corpus   {len(rows)} rows   "
          f"train={sum(r['split']=='train' for r in rows)} "
          f"test={sum(r['split']=='test' for r in rows)}")
    print(f"model    {args.model}\nbackend  MLX (unified memory), batch=1, no padding")

    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded in {time.time()-t0:.0f}s")

    if args.sanity and not sanity_check(model, tok):
        raise SystemExit(
            "SANITY CHECK FAILED -- the model emits non-text, so its weights are "
            "damaged and every activation it produces is worthless. This is exactly "
            "how the optimum-quanto attempt failed. Do not proceed; try another "
            "checkpoint (e.g. the 4bit or bf16 MLX build).")

    n_layers = len(model.model.layers)
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(n_layers)))
    d_model = model.model.embed_tokens.weight.shape[1]
    print(f"layers   {n_layers} total, sweeping {layers}\nd_model  {d_model}")

    probe_prompt = tok.apply_chat_template(
        [{"role": "user", "content": rows[0]["text"]}],
        tokenize=False, add_generation_prompt=True)
    pos_tok = position_tokens(tok, probe_prompt)
    print("read sites (offset -> token):")
    for p in POSITIONS:
        t = pos_tok[p]
        kind = "TEMPLATE" if (t.startswith("<|") or t.strip() == "assistant" or not t.strip()) else "content"
        print(f"    {p:>3}  {t!r:<24} {kind}")
    print("  The paper reads from the end of the USER turn -- use the `content` rows.")

    acts = {(l, p): np.zeros((len(rows), d_model), dtype=np.float32)
            for l in layers for p in POSITIONS}

    t0 = time.time()
    for i, r in enumerate(rows):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": r["text"]}],
            tokenize=False, add_generation_prompt=True)
        ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
        hs = hidden_states(model, ids)
        mx.eval(hs[max(layers)])
        for l in layers:
            arr = np.array(hs[l][0], copy=False)      # (seq, d_model)
            for p in POSITIONS:
                acts[(l, p)][i] = arr[p]
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"\r  {i+1}/{len(rows)}  {rate*60:.0f} rows/min  "
                  f"eta {(len(rows)-i-1)/rate/60:.1f} min   ", end="", flush=True)
    print()

    y = np.array([r["label"] for r in rows])
    is_tr = np.array([r["split"] == "train" for r in rows])
    groups = np.array([r["prompt_id"] for r in rows])
    results = analysis.score_candidates(acts, y, is_tr, groups, pos_tok)
    print(analysis.report(results))

    out = Path(args.out) if args.out else HERE / f"directions_{args.model.split('/')[-1]}.npz"
    np.savez_compressed(
        out,
        directions=np.stack([r["direction"] for r in results]),
        meta=json.dumps([{k: v for k, v in r.items() if k != "direction"} for r in results]),
        model=args.model, pos_mode="template", backend="mlx",
        pos_tokens=json.dumps({str(k): v for k, v in pos_tok.items()}))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
