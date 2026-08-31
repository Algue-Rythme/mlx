import functools
import os

import jax
import jax.numpy as jnp
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir as jmlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects import chlo, stablehlo  # noqa: F401
from mlx.export_hlo import (
    _primitive,
    export_to_hlo,
    export_tree,
    flatten_args,
)
from mlx.utils import tree_flatten, tree_map

mx.random.seed(0)
np.random.seed(0)

_PLATFORM = os.environ.get("EXPORT_HLO_PLATFORM") or None
_ON_TPU = _PLATFORM == "tpu"
skip_on_tpu = pytest.mark.skipif(_ON_TPU, reason="complex unsupported on TPU")


@functools.lru_cache(maxsize=None)
def _backend():
    return xb.get_backend(_PLATFORM) if _PLATFORM else xb.get_backend()


def run_xla(text, inputs):
    backend = _backend()
    dev = backend.local_devices()[0]
    with jmlir.make_ir_context():
        module = ir.Module.parse(text)
        exe = backend.compile_and_load(
            module,
            executable_devices=xc.DeviceList((dev,)),
            compile_options=xc.CompileOptions(),
        )
    res = exe.execute_sharded([jax.device_put(x, dev) for x in inputs])
    return res.disassemble_into_single_device_arrays()


def check_device(arr):
    # run_xla returns a list of per-output lists of single-device shards.
    while isinstance(arr, list):
        arr = arr[0]
    return arr.device.platform


def check(fn, *arrays, atol=1e-4):
    # Snapshot inputs before calling fn: some fns update their argument in place.
    np_in = [np.array(a) for a in arrays]
    out = fn(*arrays)
    ref = [np.array(o) for o in (out if isinstance(out, tuple) else (out,))]
    outs = run_xla(export_to_hlo(fn, *arrays, precision="highest"), np_in)
    for r, o in zip(ref, outs):
        assert np.allclose(np.asarray(o).reshape(r.shape), r, atol=atol)


def check_cpu(fn, *arrays, atol=1e-4):
    # Trace under a CPU stream so fused fast:: ops decompose to base ops.
    np_in = [np.array(a) for a in arrays]
    out = fn(*arrays)
    ref = [np.array(o) for o in (out if isinstance(out, tuple) else (out,))]
    with mx.stream(mx.cpu):
        text = export_to_hlo(fn, *arrays, precision="highest")
    outs = run_xla(text, np_in)
    for r, o in zip(ref, outs):
        assert np.allclose(np.asarray(o).reshape(r.shape), r, atol=atol)


def u(*shape):
    return mx.random.uniform(shape=shape)


def ri(hi, *shape):
    return mx.array(np.random.randint(0, hi, shape), dtype=mx.uint32)


def _setslice(a, up):
    a[1:3, 2:5] = up
    return a


def _scatterset(a, u):
    a[mx.array([0, 2, 5], dtype=mx.uint32)] = u
    return a


def _scatteradd(a, i, u):
    return a.at[i].add(u)


def attention(q, k, v):
    scores = (q @ mx.swapaxes(k, -1, -2)) * (q.shape[-1] ** -0.5)
    return mx.softmax(scores, axis=-1) @ v


def mlp_block(emb, tok, w1, b1, w2, b2):
    x = emb[tok]
    h = mx.maximum(x @ w1 + b1, 0.0)
    y = h @ w2 + b2
    return mx.softmax(y, axis=-1)


def layernorm_gelu(x, g, b):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    xn = (x - mu) * mx.rsqrt(var + 1e-5) * g + b
    return xn * 0.5 * (1 + mx.erf(xn / 2**0.5))


_emb = u(1000, 64)

CASES = [
    pytest.param(
        lambda x, y: mx.log(mx.abs(x - y)).astype(mx.int32),
        (mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[0.0, 1.0], [1.0, 1.0]])),
        id="log_abs_int",
    ),
    pytest.param(lambda a, b: mx.where(a > b, a, b), (u(4, 3), u(4, 3)), id="where"),
    pytest.param(lambda a, b: (a >= b) & (a != b), (u(4, 3), u(4, 3)), id="ge_and_ne"),
    pytest.param(lambda a: mx.sum(a, axis=1), (u(4, 3),), id="sum_axis1"),
    pytest.param(
        lambda a: mx.max(a, axis=0, keepdims=True), (u(4, 3),), id="max_keepdims"
    ),
    pytest.param(lambda a: mx.prod(a, axis=1), (u(4, 3) + 1,), id="prod_axis1"),
    pytest.param(lambda a: mx.transpose(a, (1, 0)), (u(2, 5),), id="transpose"),
    pytest.param(
        lambda a, b: mx.concatenate([a, b], axis=1),
        (u(2, 3), u(2, 4)),
        id="concatenate",
    ),
    pytest.param(
        lambda a: mx.broadcast_to(mx.reshape(a, (1, 6)), (4, 6)),
        (u(2, 3),),
        id="broadcast_reshape",
    ),
    pytest.param(
        lambda a: mx.rsqrt(a) + mx.sigmoid(a), (u(3, 3) + 1,), id="rsqrt_sigmoid"
    ),
    pytest.param(lambda a: a * 2.0 - 1.0, (u(4, 3),), id="affine_scalar"),
    pytest.param(lambda a: a + mx.array([1.0, 2.0, 3.0]), (u(4, 3),), id="add_vec"),
    pytest.param(lambda w, b, x: x * w + b, (u(4, 3), u(4, 3), u(4, 3)), id="mul_add"),
    pytest.param(lambda a, b: a @ b, (u(2, 3), u(3, 4)), id="matmul_2d"),
    pytest.param(lambda a, b: a @ b, (u(5, 2, 3), u(5, 3, 4)), id="matmul_batched"),
    pytest.param(lambda w, b, x: x @ w + b, (u(3, 4), u(4), u(2, 3)), id="matmul_bias"),
    pytest.param(lambda a: a[0:2, 1:6:2], (u(4, 6),), id="slice_strided"),
    pytest.param(lambda a: mx.full((2, 3), 5.0) + a, (u(2, 3),), id="full_add"),
    pytest.param(lambda a: mx.mean(a, axis=1), (u(4, 3),), id="mean_axis1"),
    pytest.param(
        lambda a, b: mx.depends([a], [b])[0] + a, (u(2, 3), u(2, 3)), id="depends1"
    ),
    pytest.param(
        lambda a, b: tuple(mx.depends([a, b], [a + b])),
        (u(2, 3), u(2, 3)),
        id="depends_tuple",
    ),
    pytest.param(lambda a: mx.softmax(a, axis=-1), (u(3, 5),), id="softmax_last"),
    pytest.param(lambda a: mx.softmax(a, axis=-1), (u(2, 4, 6),), id="softmax_last_3d"),
    pytest.param(lambda a: mx.softmax(a, axis=0), (u(3, 5),), id="softmax_axis0"),
    pytest.param(lambda a: mx.logsumexp(a, axis=-1), (u(3, 5),), id="logsumexp"),
    pytest.param(lambda a, b: mx.logaddexp(a, b), (u(3, 5), u(3, 5)), id="logaddexp"),
    pytest.param(lambda a: mx.sinh(a) + mx.cosh(a), (u(3, 5),), id="sinh_cosh"),
    pytest.param(lambda a: mx.expm1(a) + mx.log1p(a), (u(3, 5),), id="expm1_log1p"),
    pytest.param(lambda a: mx.log2(a), (u(3, 5) + 1,), id="log2"),
    pytest.param(lambda a: mx.log10(a), (u(3, 5) + 1,), id="log10"),
    pytest.param(
        lambda a: mx.softmax(a, axis=-1), (u(3, 5) * 400 - 200,), id="softmax_large"
    ),
    pytest.param(
        lambda a: mx.logsumexp(a, axis=-1),
        (u(3, 5) * 400 - 200,),
        id="logsumexp_large",
    ),
    pytest.param(lambda a: tuple(mx.split(a, 2, axis=1)), (u(2, 6),), id="split_even"),
    pytest.param(
        lambda a: tuple(mx.split(a, [1, 4], axis=1)), (u(2, 6),), id="split_idx"
    ),
    pytest.param(lambda a: mx.pad(a, [(1, 1), (0, 2)]), (u(2, 4),), id="pad"),
    pytest.param(
        lambda c, a, b: mx.addmm(c, a, b), (u(2, 4), u(2, 3), u(3, 4)), id="addmm"
    ),
    pytest.param(
        lambda c, a, b: mx.addmm(c, a, b, alpha=2.0, beta=0.5),
        (u(2, 4), u(2, 3), u(3, 4)),
        id="addmm_alpha_beta",
    ),
    pytest.param(
        lambda a: mx.arange(6).astype(mx.float32) + a, (u(6),), id="arange_int"
    ),
    pytest.param(lambda a: mx.arange(2.0, 14.0, 2.0) + a, (u(6),), id="arange_float"),
    pytest.param(
        lambda a: mx.arcsin(a) + mx.arccos(a), (u(3, 5) * 1.6 - 0.8,), id="arcsin"
    ),
    pytest.param(
        lambda a: mx.arctan(a) + mx.arcsinh(a),
        (u(3, 5) * 1.6 - 0.8,),
        id="arctan",
    ),
    pytest.param(lambda a: mx.arctanh(a), (u(3, 5) * 1.6 - 0.8,), id="arctanh"),
    pytest.param(lambda a: mx.arccosh(a), (u(3, 5) * 2 + 1,), id="arccosh"),
    pytest.param(
        lambda a, b: mx.arctan2(a, b),
        (u(3, 5) * 1.6 - 0.8, u(3, 5) * 1.6 - 0.8),
        id="arctan2",
    ),
    pytest.param(lambda a: mx.erf(a), (u(3, 5) * 1.6 - 0.8,), id="erf"),
    pytest.param(lambda a: mx.erfinv(a), (u(3, 5) * 1.6 - 0.8,), id="erfinv"),
    pytest.param(
        lambda a, i: mx.slice(a, i, axes=[0, 1], slice_size=[2, 3]),
        (u(6, 8), mx.array([1, 2], dtype=mx.uint32)),
        id="dyn_slice_2ax",
    ),
    pytest.param(
        lambda a, i: mx.slice(a, i, axes=[1], slice_size=[6, 3]),
        (u(6, 8), mx.array([2], dtype=mx.uint32)),
        id="dyn_slice_1ax",
    ),
    pytest.param(
        lambda a, up, i: mx.slice_update(a, up, i, axes=[0, 1]),
        (u(6, 8), u(2, 3), mx.array([1, 2], dtype=mx.uint32)),
        id="slice_update",
    ),
    pytest.param(_setslice, (u(6, 8), u(2, 3)), id="setitem_slice"),
    pytest.param(
        lambda x, w: mx.conv2d(x, w), (u(1, 8, 8, 3), u(4, 3, 3, 3)), id="conv2d"
    ),
    pytest.param(
        lambda x, w: mx.conv2d(x, w, stride=2, padding=1),
        (u(2, 9, 9, 3), u(5, 3, 3, 3)),
        id="conv2d_stride_pad",
    ),
    pytest.param(
        lambda x, w: mx.conv2d(x, w, groups=2),
        (u(1, 8, 8, 4), u(6, 3, 3, 2)),
        id="conv2d_groups",
    ),
    pytest.param(lambda x, w: mx.conv1d(x, w), (u(1, 10, 3), u(4, 3, 3)), id="conv1d"),
    pytest.param(lambda a: mx.sort(a, axis=-1), (u(2, 3, 4, 33),), id="sort_last"),
    pytest.param(lambda a: mx.sort(a, axis=1), (u(8, 20, 5),), id="sort_axis1"),
    pytest.param(lambda a: mx.sort(a, axis=0), (u(64, 128),), id="sort_axis0"),
    pytest.param(lambda a: mx.argsort(a, axis=-1), (u(4, 6, 50),), id="argsort_last"),
    pytest.param(lambda a: mx.argsort(a, axis=1), (u(3, 40, 7),), id="argsort_axis1"),
    pytest.param(
        lambda a: mx.sort(a.astype(mx.int32), axis=-1),
        ((u(5, 64) * 1000).astype(mx.int32),),
        id="sort_int",
    ),
    pytest.param(
        lambda a: mx.softmax(a, axis=-1),
        (u(4, 8, 32, 128) * 20 - 10,),
        id="softmax_4d_large",
    ),
    pytest.param(
        lambda a: mx.softmax(a, axis=1), (u(2, 16, 64),), id="softmax_axis1_3d"
    ),
    pytest.param(
        lambda a, b: a @ b, (u(6, 4, 32, 48), u(6, 4, 48, 24)), id="matmul_4d"
    ),
    pytest.param(lambda a: mx.sum(a, axis=2), (u(3, 5, 40, 7),), id="sum_axis2_4d"),
    pytest.param(
        lambda a: mx.max(a, axis=(1, 3), keepdims=True),
        (u(2, 9, 4, 11),),
        id="max_multiaxis",
    ),
    pytest.param(
        lambda x, w: mx.conv2d(x, w, stride=2, padding=1),
        (u(4, 32, 32, 16), u(32, 3, 3, 16)),
        id="conv2d_big",
    ),
    pytest.param(lambda w, i: w[i], (_emb, ri(1000, 5)), id="gather_1d"),
    pytest.param(lambda w, i: w[i], (_emb, ri(1000, 4, 7)), id="gather_2d"),
    pytest.param(lambda w, i: w[i], (_emb, ri(1000, 2, 3, 4)), id="gather_3d"),
    pytest.param(
        lambda w, i: mx.take(w, i, axis=1), (_emb, ri(64, 8)), id="take_axis1"
    ),
    pytest.param(lambda w, i: w[i], (u(50, 16, 32), ri(50, 6)), id="gather_3d_data"),
    pytest.param(
        lambda w, i, j: w[i, j],
        (u(6, 7), ri(6, 2, 1), ri(7, 1, 3)),
        id="gather_two_idx",
    ),
    pytest.param(
        lambda a, i: mx.take_along_axis(a, i, axis=1),
        (u(4, 20), ri(20, 4, 6)),
        id="take_along_1",
    ),
    pytest.param(
        lambda a, i: mx.take_along_axis(a, i, axis=0),
        (u(8, 5, 3), ri(8, 2, 5, 3)),
        id="take_along_0",
    ),
    pytest.param(
        lambda a, i, u_: mx.put_along_axis(a, i, u_, axis=1),
        (u(4, 20), ri(20, 4, 3), u(4, 3)),
        id="put_along_1",
    ),
    pytest.param(_scatterset, (u(8, 7), u(3, 7)), id="scatter_set"),
    pytest.param(
        _scatteradd,
        (u(8, 7), mx.array([0, 2, 2, 5], dtype=mx.uint32), u(4, 7)),
        id="scatter_add",
    ),
    pytest.param(
        attention, (u(2, 4, 16, 32), u(2, 4, 16, 32), u(2, 4, 16, 32)), id="attention"
    ),
    pytest.param(
        mlp_block,
        (u(100, 32), ri(100, 8, 12), u(32, 64), u(64), u(64, 10), u(10)),
        id="mlp_block",
    ),
    pytest.param(
        layernorm_gelu,
        (u(4, 16, 48) * 4 - 2, u(48), u(48)),
        id="layernorm_gelu",
    ),
    pytest.param(
        lambda k: mx.random.uniform(shape=(64,), key=k),
        (mx.random.key(0),),
        id="rng_uniform",
    ),
    pytest.param(
        lambda k: mx.random.normal(shape=(32,), key=k),
        (mx.random.key(7),),
        id="rng_normal",
    ),
    pytest.param(
        lambda x, w: mx.fast.rms_norm(x, w, 1e-5),
        (u(4, 12, 8), u(8)),
        id="rms_norm",
    ),
    pytest.param(
        lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
        (u(4, 12, 8), u(8), u(8)),
        id="layer_norm",
    ),
    pytest.param(
        lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.35),
        (u(2, 4, 16, 8), u(2, 4, 16, 8), u(2, 4, 16, 8)),
        id="sdpa",
    ),
]


@pytest.mark.parametrize("fn,arrays", CASES)
def test_export_matches(fn, arrays):
    check(fn, *arrays)


@skip_on_tpu
def test_complex_real_imag():
    check(
        lambda a: mx.real(a) + mx.imag(a),
        mx.array(np.array([[1 + 2j, 3 - 1j]], dtype=np.complex64)),
    )


@skip_on_tpu
def test_complex_conj():
    check(
        lambda a: mx.conj(a),
        mx.array(np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex64)),
    )


def test_constant_guardrail():
    big = u(512, 512)
    mx.eval(big)
    with pytest.raises(NotImplementedError):
        export_to_hlo(lambda a: a + big, u(512, 512), precision="highest")


def test_strided_slice_update_rejected():
    def setstrided(a, up):
        a[1:6:2, 0:8:2] = up
        return a

    with pytest.raises(NotImplementedError):
        export_to_hlo(setstrided, u(6, 8), u(3, 4), precision="highest")


def test_cholesky():
    m = np.array([[4.0, 1.0], [1.0, 3.0]])
    spd = mx.array(np.linalg.cholesky(m) @ np.linalg.cholesky(m).T, dtype=mx.float32)
    with mx.stream(mx.cpu):
        ref = np.array(mx.linalg.cholesky(spd))
    text = export_to_hlo(
        lambda a: mx.linalg.cholesky(a, stream=mx.cpu), spd, precision="highest"
    )
    outs = run_xla(text, [np.array(spd)])
    assert np.allclose(np.asarray(outs[0]).reshape(ref.shape), ref, atol=1e-4)


def test_number_of_elements():
    noe = {
        "name": "NumberOfElements",
        "arguments": [[1], True, mx.float32],
        "inputs": [("A", (4, 8), mx.float32)],
        "outputs": [("B", (), mx.float32)],
    }
    assert _primitive(noe, "HIGHEST") == [
        "%B = stablehlo.constant dense<0.125> : tensor<f32>"
    ]


def test_outputs_on_accelerator():
    plat = _backend().platform
    if plat == "cpu":
        pytest.skip("no accelerator backend; set EXPORT_HLO_PLATFORM=tpu on a TPU host")
    a, b = np.array(u(4, 8)), np.array(u(8, 4))
    outs = run_xla(
        export_to_hlo(
            lambda x, y: x @ y, mx.array(a), mx.array(b), precision="highest"
        ),
        [a, b],
    )
    for o in outs:
        assert check_device(o) == plat


@pytest.mark.parametrize(
    "dims,traditional,offset,shape",
    [
        (8, False, 0, (2, 3, 4, 8)),
        (8, True, 2, (2, 3, 4, 8)),
        (4, False, 0, (1, 2, 4, 8)),
    ],
)
def test_rope(dims, traditional, offset, shape):
    check_cpu(
        lambda x: mx.fast.rope(
            x, dims, traditional=traditional, base=1e4, scale=1.0, offset=offset
        ),
        u(*shape),
    )


# --- tiny transformer training step (forward + backward + SGD) ---
B, T, V, D, H, HD, FF = 3, 4, 16, 8, 2, 4, 16


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


def _rp(*shape):
    return mx.array((np.random.rand(*shape) * 0.2 - 0.1).astype(np.float32))


def _make_params():
    return [
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


def test_transformer_step():
    params = _make_params()
    x = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    y = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    check(train_step, *params, x, y)


def test_multistep_convergence():
    params = _make_params()
    x = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    y = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
    mp = [mx.array(np.array(p)) for p in params]
    xp = [np.array(p) for p in params]
    xn, yn = np.array(x), np.array(y)
    mlx_losses, xla_losses = [], []
    for _ in range(5):
        mo = train_step(*mp, x, y)
        mlx_losses.append(float(mo[0]))
        mp = list(mo[1:])
        text = export_to_hlo(
            train_step, *[mx.array(p) for p in xp], x, y, precision="highest"
        )
        outs = run_xla(text, [*xp, xn, yn])
        xla_losses.append(float(np.asarray(outs[0]).reshape(())))
        xp = [
            np.asarray(o).reshape(np.array(params[i]).shape)
            for i, o in enumerate(outs[1:])
        ]
    assert np.allclose(mlx_losses, xla_losses, atol=1e-3), (mlx_losses, xla_losses)


def test_bf16_attention():
    q32 = np.random.rand(2, 4, 16, 8).astype(np.float32)
    qm = mx.array(q32).astype(mx.bfloat16)
    ref = np.array(attention(qm, qm, qm).astype(mx.float32))
    qbf = jnp.asarray(q32).astype(jnp.bfloat16)
    outs = run_xla(
        export_to_hlo(attention, qm, qm, qm, precision="highest"), [qbf, qbf, qbf]
    )
    o = np.asarray(outs[0]).astype(np.float32).reshape(ref.shape)
    assert np.allclose(o, ref, atol=2e-2), np.abs(o - ref).max()


@pytest.mark.parametrize(
    "name,composite,fn,nargs",
    [
        ("mlx.rms_norm", "RMSNorm", lambda x, w: mx.fast.rms_norm(x, w, 1e-5), 2),
        (
            "mlx.layer_norm",
            "LayerNorm",
            lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
            3,
        ),
    ],
)
def test_composite(name, composite, fn, nargs):
    args = (u(3, 8), u(8), u(8))[:nargs]
    text = export_to_hlo(fn, *args, composites={composite}, precision="highest")
    if f'stablehlo.composite "{name}"' not in text:
        pytest.skip("fast ops decompose on this device; no fused primitive to wrap")
    assert f"func.func private @{name.replace('.', '_')}_0" in text
    outs = run_xla(text, [np.array(a) for a in args])
    ref = np.array(fn(*args))
    assert np.allclose(np.asarray(outs[0]).reshape(ref.shape), ref, atol=1e-4)


def test_pytree_module_step():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    mx.eval(model.parameters())
    p0 = model.parameters()

    def mloss(params, x, y):
        model.update(params)
        return mx.mean((model(x) - y) ** 2)

    gfn = mx.value_and_grad(mloss)

    def mstep(params, x, y):
        loss, grads = gfn(params, x, y)
        return loss, tree_map(lambda p, g: p - 0.1 * g, params, grads)

    xd, yd = u(5, 8), u(5, 4)
    ref = [np.array(v) for _, v in tree_flatten(mstep(p0, xd, yd))]
    hlo, _ = export_tree(mstep, p0, xd, yd, precision="highest")
    outs = run_xla(hlo, [np.array(a) for a in flatten_args(p0, xd, yd)])
    assert len(outs) == len(ref)
    for r, o in zip(ref, outs):
        assert np.allclose(np.asarray(o).reshape(r.shape), r, atol=1e-4)


def test_bf16_adam_step():
    am = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    mx.eval(am.parameters())
    ap = tree_map(lambda p: p.astype(mx.bfloat16), am.parameters())
    amm = tree_map(lambda p: mx.zeros(p.shape, dtype=mx.float32), ap)
    avv = tree_map(lambda p: mx.zeros(p.shape, dtype=mx.float32), ap)
    b1, b2, lr, eps = 0.9, 0.999, 0.01, 1e-8

    def aloss(params, x, y):
        am.update(params)
        return mx.mean((am(x) - y) ** 2)

    def adam(params, m, v, t, x, y):
        loss, grads = mx.value_and_grad(aloss)(params, x, y)
        t2 = t + 1.0
        nm = tree_map(lambda mi, g: b1 * mi + (1 - b1) * g.astype(mx.float32), m, grads)
        nv = tree_map(
            lambda vi, g: b2 * vi + (1 - b2) * g.astype(mx.float32) ** 2, v, grads
        )
        bc1, bc2 = 1 - b1**t2, 1 - b2**t2
        np_ = tree_map(
            lambda p, mi, vi: p
            - (lr * (mi / bc1) / (mx.sqrt(vi / bc2) + eps)).astype(p.dtype),
            params,
            nm,
            nv,
        )
        return loss, np_, nm, nv, t2

    xd = u(6, 8).astype(mx.bfloat16)
    yd = u(6, 4).astype(mx.bfloat16)
    t = mx.array(0.0, dtype=mx.float32)
    ref = [
        np.array(o.astype(mx.float32))
        for _, o in tree_flatten(adam(ap, amm, avv, t, xd, yd))
    ]
    hlo, _ = export_tree(adam, ap, amm, avv, t, xd, yd, precision="highest")

    def to_jax(a):
        n = np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)
        return jnp.asarray(n).astype(jnp.bfloat16) if a.dtype == mx.bfloat16 else n

    outs = run_xla(hlo, [to_jax(a) for a in flatten_args(ap, amm, avv, t, xd, yd)])
    for r, o in zip(ref, outs):
        o = np.array(jnp.asarray(o).astype(jnp.float32)).reshape(r.shape)
        assert np.allclose(o, r, atol=3e-2), np.abs(o - r).max()
