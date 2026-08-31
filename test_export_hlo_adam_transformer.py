import os

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import pytest
from jax import export as jexport
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir as jmlir
from jax._src.lib.mlir import ir
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib import _jax
from jaxlib.mlir.dialects import stablehlo as hlo
from mlx.export_hlo import export_tree, flatten_args
from mlx.utils import tree_flatten

mx.random.seed(0)
np.random.seed(0)

# Llama-style decoder: RMSNorm, RoPE, causal attention, SwiGLU, no biases.
B, T, V, D, H, HD, L, FF = 4, 8, 32, 16, 2, 8, 2, 32
EPS, BASE = 1e-5, 10000.0
B1, B2, LR, ADAM_EPS = 0.9, 0.999, 1e-3, 1e-8

_PLATFORM = os.environ.get("EXPORT_HLO_PLATFORM") or xb.get_backend().platform
_devices = jax.devices(_PLATFORM)

# Explicit meshes (label, (dp, tp, cp)); each runs only where its count exists.
MESHES = [("fsdp_tp", (2, 2, 1)), ("fsdp_tp_cp", (2, 2, 2))]


def _mesh_params():
    params = []
    for label, (dp, tp, cp) in MESHES:
        need = dp * tp * cp
        mark = pytest.mark.skipif(len(_devices) < need, reason=f"need {need} devices")
        params.append(pytest.param(dp, tp, cp, id=label, marks=mark))
    return params


def _attention(x, wq, wk, wv, wo, mask):
    q = mx.swapaxes(mx.reshape(x @ wq, (B, T, H, HD)), 1, 2)
    k = mx.swapaxes(mx.reshape(x @ wk, (B, T, H, HD)), 1, 2)
    v = mx.swapaxes(mx.reshape(x @ wv, (B, T, H, HD)), 1, 2)
    q = mx.fast.rope(q, HD, traditional=False, base=BASE, scale=1.0, offset=0)
    k = mx.fast.rope(k, HD, traditional=False, base=BASE, scale=1.0, offset=0)
    scores = (q @ mx.swapaxes(k, -1, -2)) * (HD**-0.5)
    scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
    a = mx.softmax(scores, axis=-1) @ v
    a = mx.reshape(mx.swapaxes(a, 1, 2), (B, T, D))
    return a @ wo


def _swiglu(x, w_gate, w_up, w_down):
    g = x @ w_gate
    return (g * mx.sigmoid(g)) * (x @ w_up) @ w_down


def _forward(params, tokens):
    emb, g_final, lm_head = params[0], params[-2], params[-1]
    layers = params[1:-2]
    idx = mx.arange(T)
    mask = idx[:, None] >= idx[None, :]
    h = emb[tokens]
    for li in range(L):
        p = layers[li * 9 : li * 9 + 9]
        g_attn, wq, wk, wv, wo, g_ffn, w_gate, w_up, w_down = p
        h = h + _attention(mx.fast.rms_norm(h, g_attn, EPS), wq, wk, wv, wo, mask)
        h = h + _swiglu(mx.fast.rms_norm(h, g_ffn, EPS), w_gate, w_up, w_down)
    return mx.fast.rms_norm(h, g_final, EPS) @ lm_head


def _loss(params, x, y):
    logits = _forward(params, x).astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    tgt = mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
    return mx.mean(lse - tgt)


def adam_step(params, m, v, t, x, y):
    loss, grads = mx.value_and_grad(lambda p: _loss(p, x, y))(params)
    t2 = t + 1.0
    m2 = [B1 * mi + (1 - B1) * g.astype(mx.float32) for mi, g in zip(m, grads)]
    v2 = [B2 * vi + (1 - B2) * g.astype(mx.float32) ** 2 for vi, g in zip(v, grads)]
    bc1, bc2 = 1 - B1**t2, 1 - B2**t2
    p2 = [
        p - (LR * (mi / bc1) / (mx.sqrt(vi / bc2) + ADAM_EPS)).astype(p.dtype)
        for p, mi, vi in zip(params, m2, v2)
    ]
    return loss, p2, m2, v2, t2


def _rp(*shape):
    a = (np.random.rand(*shape) * 0.2 - 0.1).astype(np.float32)
    return mx.array(a).astype(mx.bfloat16)


def _make_params():
    params = [_rp(V, D)]
    for _ in range(L):
        params += [
            _rp(D),  # g_attn
            _rp(D, D),  # wq
            _rp(D, D),  # wk
            _rp(D, D),  # wv
            _rp(D, D),  # wo
            _rp(D),  # g_ffn
            _rp(D, FF),  # w_gate
            _rp(D, FF),  # w_up
            _rp(FF, D),  # w_down
        ]
    params += [_rp(D), _rp(D, V)]  # g_final, lm_head
    return params


def _param_specs():
    DP, TP = "dp", "tp"
    specs = [P(DP, TP)]  # emb
    for _ in range(L):
        specs += [
            P(TP),  # g_attn
            P(DP, TP),  # wq  column-parallel
            P(DP, TP),  # wk
            P(DP, TP),  # wv
            P(TP, DP),  # wo  row-parallel
            P(TP),  # g_ffn
            P(DP, TP),  # w_gate  column-parallel
            P(DP, TP),  # w_up
            P(TP, DP),  # w_down  row-parallel
        ]
    specs += [P(TP), P(DP, TP)]  # g_final, lm_head
    return specs


def to_exported(text, in_avals, out_avals):
    backend = xb.get_backend(_PLATFORM)
    with jmlir.make_ir_context():
        module = ir.Module.parse(text)
        ver = hlo.get_version_from_compatibility_requirement(
            hlo.StablehloCompatibilityRequirement.WEEK_4
        )
        blob = _jax.mlir.serialize_portable_artifact(
            module, ver, backend.serialize_with_sdy
        )
    ni, no = len(in_avals), len(out_avals)
    return jexport.Exported(
        fun_name="main",
        in_tree=jax.tree_util.tree_structure((tuple(range(ni)), {})),
        in_avals=tuple(in_avals),
        out_tree=jax.tree_util.tree_structure(tuple(range(no))),
        out_avals=tuple(out_avals),
        _in_named_shardings=(None,) * ni,
        _out_named_shardings=(None,) * no,
        in_shardings_hlo=(None,) * ni,
        out_shardings_hlo=(None,) * no,
        nr_devices=1,
        platforms=(_PLATFORM,),
        ordered_effects=(),
        unordered_effects=(),
        disabled_safety_checks=(),
        mlir_module_serialized=blob,
        calling_convention_version=jexport.maximum_supported_calling_convention_version,
        module_kept_var_idx=tuple(range(ni)),
        uses_global_constants=False,
        _get_vjp=None,
    )


_JDTYPE = {mx.bfloat16: jnp.bfloat16, mx.float32: np.float32, mx.uint32: np.uint32}


def _to_jax(a):
    if a.dtype == mx.bfloat16:
        return jnp.asarray(np.array(a.astype(mx.float32))).astype(jnp.bfloat16)
    return np.array(a)


def _f32(o):
    return np.array(jnp.asarray(o).astype(jnp.float32))


@pytest.mark.parametrize("dp,tp,cp", _mesh_params())
def test_adam_transformer_fsdp_tp_cp(dp, tp, cp):
    params = _make_params()
    m = [mx.zeros(p.shape, dtype=mx.float32) for p in params]
    v = [mx.zeros(p.shape, dtype=mx.float32) for p in params]
    t = mx.array(0.0, dtype=mx.float32)
    x = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    y = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    args = (params, m, v, t, x, y)

    ref_leaves = [o for _, o in tree_flatten(adam_step(*args))]
    text, _ = export_tree(adam_step, *args, precision="highest")
    ref = [np.array(o.astype(mx.float32)) for o in ref_leaves]

    flat_in = flatten_args(*args)
    in_avals = [jax.core.ShapedArray(a.shape, _JDTYPE[a.dtype]) for a in flat_in]
    out_avals = [jax.core.ShapedArray(o.shape, _JDTYPE[o.dtype]) for o in ref_leaves]
    exp = to_exported(text, in_avals, out_avals)

    mesh = Mesh(
        np.array(_devices[: dp * tp * cp]).reshape(dp, tp, cp), ("dp", "tp", "cp")
    )
    ps = _param_specs()
    in_specs = ps + ps + ps + [P(), P("dp", "cp"), P("dp", "cp")]
    out_specs = [P()] + ps + ps + ps + [P()]

    in_sh = [NamedSharding(mesh, s) for s in in_specs]
    out_sh = [NamedSharding(mesh, s) for s in out_specs]
    f = jax.jit(exp.call, in_shardings=tuple(in_sh), out_shardings=tuple(out_sh))
    puts = [jax.device_put(_to_jax(a), s) for a, s in zip(flat_in, in_sh)]

    compiled = f.lower(*puts).compile().as_text()
    collectives = [
        c for c in ("all-reduce", "all-gather", "reduce-scatter") if c in compiled
    ]
    outs = f(*puts)

    for o, r in zip(outs, ref):
        got = _f32(o).reshape(r.shape)
        assert np.allclose(got, r, atol=1e-2), np.abs(got - r).max()
    assert collectives, "expected communication under FSDP/TP/CP"
