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
        if self._vec is not None and self._coeff:
            h = h + self._coeff * self._vec.astype(h.dtype)
        if self._record:
            self._out = h
        return h


@contextmanager
def taps(model, record=True, steer_layer=None, vec=None, coeff=0.0):
    """Wrap every block of `model.model.layers`, restoring the originals on exit.

    steer_layer/vec/coeff inject `coeff * vec` into that block's OUTPUT, which is the
    same site the direction was measured at (block l's output), so read site and
    injection site coincide by construction.
    """
    inner = model.model
    original = list(inner.layers)
    wrapped = []
    for i, layer in enumerate(original):
        t = _Tap(layer)
        t._record = bool(record)
        if steer_layer is not None and i == steer_layer:
            t._vec, t._coeff = vec, float(coeff)
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
