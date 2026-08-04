"""Capture and steer residual-stream activations WITHOUT reimplementing the forward pass.

WHY THIS EXISTS. The first version hand-rolled the transformer loop:

    h = embed_tokens(ids); mask = create_attention_mask(h, None)
    for layer in layers: h = layer(h, mask, None)
    h = norm(h); logits = model.lm_head(h)

That works on Llama and is silently WRONG on Gemma-2, which additionally:
  * scales embeddings by sqrt(hidden_size)  -- a factor of ~48, skipped entirely
  * ties its embeddings, so there is no lm_head
  * applies final_logit_softcapping, tanh(out/30)*30, to the logits
  * needs create_attention_mask(..., return_array=True), not the 'causal' sentinel
Each of those had to be discovered by a crash or by reading the model source, and the
embedding-scale one produced plausible-looking numbers (0.758 accuracy) from a model
that was not actually Gemma.

So: stop reimplementing. Wrap each transformer block in a tap that records its output
and can add c*v to it, then call the model's OWN __call__. Every family-specific
detail -- embedding scale, mask construction, tied heads, softcapping -- comes from
the model's own code, and any future architecture works without changes here.

    with taps(model, steer_layer=14, vec=v, coeff=2.5) as t:
        logits = model(ids)          # model's own forward, fully correct
        h14 = t.captured(14)         # residual stream after block 14
"""

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


class _Tap(nn.Module):
    """Transparent proxy around one transformer block.

    An nn.Module (not a bare object) so MLX's module tree stays traversable when the
    layer list is swapped out.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._out = None
        self._record = True
        self._coeff = 0.0
        self._vec = None
        self._ablate = None      # unit vector to project OUT of this block's output

    def __getattr__(self, name):
        """Delegate anything we do not define to the wrapped layer.

        mlx_lm reaches into layer attributes from outside (e.g. `layer.use_sliding`
        in the Llama stack), so the proxy has to be transparent. MLX's Module is
        dict-backed, hence the dict.get to fetch `inner` without re-entering here.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        inner = dict.get(self, "inner", None)
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, x, *args, **kwargs):
        h = self.inner(x, *args, **kwargs)
        if self._ablate is not None:
            # Directional ablation (Arditi et al. 2024): h <- h - (h . v)v, at EVERY
            # position. Computed in float32 -- an fp16 dot product over thousands of
            # dimensions overflows, and a silent inf here would zero the whole stream.
            v = self._ablate.astype(mx.float32)
            h32 = h.astype(mx.float32)
            h = (h32 - (h32 @ v)[..., None] * v).astype(h.dtype)
        if self._vec is not None and self._coeff:
            h = h + self._coeff * self._vec.astype(h.dtype)
        if self._record:
            self._out = h
        return h


@contextmanager
def taps(model, record=True, steer_layer=None, vec=None, coeff=0.0, ablate_vec=None):
    """Wrap every block of `model.model.layers`, restoring the originals on exit.

    steer_layer/vec/coeff inject `coeff * vec` into that block's OUTPUT, which is the
    same site the direction was measured at (block l's output), so read site and
    injection site coincide by construction.

    ablate_vec projects that direction OUT of every block's output, at every position.
    This is a different operation from steering and it is the paper's Experiment 1/2
    intervention, following Arditi et al. 2024:

        steering  : one layer,     h <- h + c*v      (adds a component)
        ablation  : every layer,   h <- h - (h.v)v   (removes a component)

    NOTE ON FIDELITY. Arditi ablate the direction from the attention and MLP write-outs
    as well as the residual stream. Ablating each block's output is the residual-stream
    variant: a block can still write the component back, and it is removed again
    immediately after, so the stream never carries it between blocks. The embedding is
    covered because block 0's output includes it through the residual path. This is the
    common simplification and it is not identical to theirs -- see RESULTS.md.
    """
    inner = model.model
    original = list(inner.layers)
    wrapped = []
    for i, layer in enumerate(original):
        t = _Tap(layer)
        t._record = bool(record)
        if steer_layer is not None and i == steer_layer:
            t._vec, t._coeff = vec, float(coeff)
        if ablate_vec is not None:
            t._ablate = ablate_vec
        wrapped.append(t)
    inner.layers = wrapped

    class Handle:
        n_layers = len(wrapped)

        @staticmethod
        def captured(i):
            return wrapped[i]._out

        @staticmethod
        def all_captured():
            return [w._out for w in wrapped]

    try:
        yield Handle
    finally:
        inner.layers = original


def hidden_states(model, ids):
    """Residual stream after every block, via the model's own forward pass.

    Returns a list of length n_layers; element k is block k's output. Matches the
    PyTorch convention where hidden_states[k+1] is layer k, so layer indices are
    comparable across backends.
    """
    with taps(model, record=True) as t:
        model(ids)
        return t.all_captured()


def logits_steered(model, ids, steer_layer, vec, coeff):
    """Next-token logits with coeff*vec added at steer_layer's output.

    coeff = 0 runs the identical code path, so baseline and steered are strictly
    comparable -- an earlier version skipped the injection branch when coeff was 0,
    which meant the baseline was the only run taking a different path.
    """
    with taps(model, record=False, steer_layer=steer_layer, vec=vec, coeff=coeff):
        return model(ids)[0, -1]


def logits_ablated(model, ids, ablate_vec):
    """Next-token logits with ablate_vec projected out of every block's output.

    Pass ablate_vec=None for the baseline so both sides take the same code path.
    """
    with taps(model, record=False, ablate_vec=ablate_vec):
        return model(ids)[0, -1]
