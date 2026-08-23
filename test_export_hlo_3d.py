import os

import jax
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
from mlx.export_hlo import export_to_hlo

mx.random.seed(0)
np.random.seed(0)

B, T, V, D, H, HD, FF = 4, 4, 16, 8, 2, 4, 16
_PLATFORM = os.environ.get("EXPORT_HLO_PLATFORM") or xb.get_backend().platform
_devices = jax.devices(_PLATFORM)


# Explicit meshes: (label, (dp, tp, cp)). Each runs only where its device count
# is available: 2x2x1 uses 4 devices (FSDP + TP), 2x2x2 uses 8 (+ context).
MESHES = [("fsdp_tp", (2, 2, 1)), ("fsdp_tp_cp", (2, 2, 2))]


def _mesh_params():
    params = []
    for label, (dp, tp, cp) in MESHES:
        need = dp * tp * cp
        mark = pytest.mark.skipif(len(_devices) < need, reason=f"need {need} devices")
        params.append(pytest.param(dp, tp, cp, id=label, marks=mark))
    return params


def _ln(x, g, b):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + 1e-5) * g + b


def _forward(params, x):
    emb, wq, wk, wv, wo, w1, b1, w2, b2, g1, c1, g2, c2, wout = params
    h = emb[x]
    hn = _ln(h, g1, c1)
    q = mx.swapaxes(mx.reshape(hn @ wq, (B, T, H, HD)), 1, 2)
    k = mx.swapaxes(mx.reshape(hn @ wk, (B, T, H, HD)), 1, 2)
    v = mx.swapaxes(mx.reshape(hn @ wv, (B, T, H, HD)), 1, 2)
    a = mx.softmax((q @ mx.swapaxes(k, -1, -2)) * (HD**-0.5), axis=-1) @ v
    a = mx.reshape(mx.swapaxes(a, 1, 2), (B, T, D)) @ wo
    h = h + a
    hn2 = _ln(h, g2, c2)
    h = h + mx.maximum(hn2 @ w1 + b1, 0.0) @ w2 + b2
    return h @ wout


def _loss(params, x, y):
    logits = _forward(params, x)
    lse = mx.logsumexp(logits, axis=-1)
    tgt = mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
    return mx.mean(lse - tgt)


def train_step(*flat):
    params, x, y = list(flat[:-2]), flat[-2], flat[-1]
    loss, grads = mx.value_and_grad(lambda p: _loss(p, x, y))(params)
    new = [p - 0.1 * g for p, g in zip(params, grads)]
    return (loss, *new)


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


def _rp(*shape):
    return mx.array((np.random.rand(*shape) * 0.2 - 0.1).astype(np.float32))


@pytest.mark.parametrize("dp,tp,cp", _mesh_params())
def test_transformer_fsdp_tp_cp(dp, tp, cp):
    params = [
        _rp(V, D),
        _rp(D, D),
        _rp(D, D),
        _rp(D, D),
        _rp(D, D),
        _rp(D, FF),
        _rp(FF),
        _rp(FF, D),
        _rp(D),
        _rp(D),
        _rp(D),
        _rp(D),
        _rp(D),
        _rp(D, V),
    ]
    x = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    y = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    args = [*params, x, y]

    ref = [np.array(o) for o in train_step(*args)]
    in_avals = [
        jax.core.ShapedArray(np.array(a).shape, np.array(a).dtype) for a in args
    ]
    out_avals = [jax.core.ShapedArray(o.shape, o.dtype) for o in ref]
    exp = to_exported(export_to_hlo(train_step, *args), in_avals, out_avals)

    mesh = Mesh(
        np.array(_devices[: dp * tp * cp]).reshape(dp, tp, cp),
        ("dp", "tp", "cp"),
    )
    DP, TP, CP = "dp", "tp", "cp"
    # FSDP shards params+batch on dp; Megatron TP on tp; context-parallel seq on cp.
    param_specs = [
        P(DP, TP),  # emb  (V, D)
        P(DP, TP),
        P(DP, TP),
        P(DP, TP),  # wq wk wv (D, D) column-parallel
        P(TP, DP),  # wo   (D, D) row-parallel
        P(DP, TP),  # w1   (D, FF) column-parallel
        P(TP),  # b1   (FF,)
        P(TP, DP),  # w2   (FF, D) row-parallel
        P(TP),  # b2   (D,)
        P(TP),
        P(TP),
        P(TP),
        P(TP),  # g1 c1 g2 c2 (D,)
        P(DP, TP),  # wout (D, V)
    ]
    in_specs = [*param_specs, P(DP, CP), P(DP, CP)]  # x, y (B, T)
    out_specs = [P(), *param_specs]  # loss replicated; params keep layout

    in_sh = [NamedSharding(mesh, s) for s in in_specs]
    out_sh = [NamedSharding(mesh, s) for s in out_specs]
    f = jax.jit(exp.call, in_shardings=tuple(in_sh), out_shardings=tuple(out_sh))
    puts = [jax.device_put(np.array(a), s) for a, s in zip(args, in_sh)]

    text = f.lower(*puts).compile().as_text()
    collectives = [
        c for c in ("all-reduce", "all-gather", "reduce-scatter") if c in text
    ]
    outs = f(*puts)

    for o, r in zip(outs, ref):
        assert np.allclose(np.array(o), r, atol=1e-4)
    assert collectives, "expected communication under FSDP/TP/CP"
