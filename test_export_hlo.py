import os
import sys

_local = "/Users/louisbethune/code/mlx/python"
if os.path.isdir(_local):
    sys.path.insert(0, _local)

import jax
import mlx.core as mx
import numpy as np
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir as jmlir
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects import chlo, stablehlo  # noqa: F401
from mlx.export_hlo import export_to_hlo


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Validate MLX StableHLO export against XLA."
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tpu", action="store_true", help="run on the TPU backend")
    g.add_argument("--platform", default=None, help="XLA platform: cpu, cuda, tpu")
    return p.parse_args()


_args = _parse_args() if __name__ == "__main__" else None
_PLATFORM = "tpu" if (_args and _args.tpu) else (_args.platform if _args else None)
_ON_TPU = _PLATFORM == "tpu"

_BACKENDS = {}


def _get_backend():
    if _PLATFORM not in _BACKENDS:
        _BACKENDS[_PLATFORM] = (
            xb.get_backend(_PLATFORM) if _PLATFORM else xb.get_backend()
        )
    return _BACKENDS[_PLATFORM]


def run_xla(text, inputs):
    backend = _get_backend()
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


def check(fn, *arrays):
    out = fn(*arrays)
    ref = [np.array(o) for o in (out if isinstance(out, tuple) else (out,))]
    text = export_to_hlo(fn, *arrays)
    outs = run_xla(text, [np.array(a) for a in arrays])
    for r, o in zip(ref, outs):
        o = np.asarray(o).reshape(r.shape)
        print("mlx:", r.ravel(), "xla:", o.ravel())
        assert np.allclose(o, r, atol=1e-4), "mismatch"
    print("OK")


def _skip_complex(*a, **k):
    print("skip (complex unsupported on TPU)")


# Complex ops (real/imag/conj) are not supported on TPU; skip them there.
check_complex = _skip_complex if _ON_TPU else check


def fn(x, y):
    return mx.log(mx.abs(x - y)).astype(mx.int32)


check(fn, mx.array([[1.0, 2.0], [3.0, 4.0]]), mx.array([[0.0, 1.0], [1.0, 1.0]]))
check(
    lambda a, b: mx.where(a > b, a, b),
    mx.random.uniform(shape=(4, 3)),
    mx.random.uniform(shape=(4, 3)),
)
check(
    lambda a, b: (a >= b) & (a != b),
    mx.random.uniform(shape=(4, 3)),
    mx.random.uniform(shape=(4, 3)),
)
check(lambda a: mx.sum(a, axis=1), mx.random.uniform(shape=(4, 3)))
check(lambda a: mx.max(a, axis=0, keepdims=True), mx.random.uniform(shape=(4, 3)))
check(lambda a: mx.prod(a, axis=1), mx.random.uniform(shape=(4, 3)) + 1)
check(lambda a: mx.transpose(a, (1, 0)), mx.random.uniform(shape=(2, 5)))
check(
    lambda a, b: mx.concatenate([a, b], axis=1),
    mx.random.uniform(shape=(2, 3)),
    mx.random.uniform(shape=(2, 4)),
)
check(
    lambda a: mx.broadcast_to(mx.reshape(a, (1, 6)), (4, 6)),
    mx.random.uniform(shape=(2, 3)),
)
check(lambda a: mx.rsqrt(a) + mx.sigmoid(a), mx.random.uniform(shape=(3, 3)) + 1)

check(lambda a: a * 2.0 - 1.0, mx.random.uniform(shape=(4, 3)))
check(lambda a: a + mx.array([1.0, 2.0, 3.0]), mx.random.uniform(shape=(4, 3)))
check(
    lambda w, b, x: x * w + b,
    mx.random.uniform(shape=(4, 3)),
    mx.random.uniform(shape=(4, 3)),
    mx.random.uniform(shape=(4, 3)),
)

big = mx.random.uniform(shape=(512, 512))
mx.eval(big)
try:
    export_to_hlo(lambda a: a + big, mx.random.uniform(shape=(512, 512)))
    raise SystemExit("guardrail did not trigger")
except NotImplementedError as e:
    print("guardrail OK:", str(e))

check(
    lambda a, b: a @ b, mx.random.uniform(shape=(2, 3)), mx.random.uniform(shape=(3, 4))
)
check(
    lambda a, b: a @ b,
    mx.random.uniform(shape=(5, 2, 3)),
    mx.random.uniform(shape=(5, 3, 4)),
)
check(
    lambda w, b, x: x @ w + b,
    mx.random.uniform(shape=(3, 4)),
    mx.random.uniform(shape=(4,)),
    mx.random.uniform(shape=(2, 3)),
)
check(lambda a: a[0:2, 1:6:2], mx.random.uniform(shape=(4, 6)))
check(lambda a: mx.full((2, 3), 5.0) + a, mx.random.uniform(shape=(2, 3)))
check(lambda a: mx.mean(a, axis=1), mx.random.uniform(shape=(4, 3)))

spd = mx.array(
    np.linalg.cholesky(np.array([[4.0, 1.0], [1.0, 3.0]]))
    @ np.linalg.cholesky(np.array([[4.0, 1.0], [1.0, 3.0]])).T,
    dtype=mx.float32,
)
with mx.stream(mx.cpu):
    ref = np.array(mx.linalg.cholesky(spd))
text = export_to_hlo(lambda a: mx.linalg.cholesky(a, stream=mx.cpu), spd)
outs = run_xla(text, [np.array(spd)])
assert np.allclose(np.asarray(outs[0]).reshape(ref.shape), ref, atol=1e-4)
print("cholesky OK")

check_complex(
    lambda a: mx.real(a) + mx.imag(a),
    mx.array(np.array([[1 + 2j, 3 - 1j]], dtype=np.complex64)),
)

check(
    lambda a, b: mx.depends([a], [b])[0] + a,
    mx.random.uniform(shape=(2, 3)),
    mx.random.uniform(shape=(2, 3)),
)
check(
    lambda a, b: tuple(mx.depends([a, b], [a + b])),
    mx.random.uniform(shape=(2, 3)),
    mx.random.uniform(shape=(2, 3)),
)

from mlx.export_hlo import _primitive  # noqa: E402

noe = {
    "name": "NumberOfElements",
    "arguments": [[1], True, mx.float32],
    "inputs": [("A", (4, 8), mx.float32)],
    "outputs": [("B", (), mx.float32)],
}
assert _primitive(noe, "HIGHEST") == [
    "%B = stablehlo.constant dense<0.125> : tensor<f32>"
]
print("number_of_elements OK")

check(lambda a: mx.softmax(a, axis=-1), mx.random.uniform(shape=(3, 5)))
check(lambda a: mx.softmax(a, axis=-1), mx.random.uniform(shape=(2, 4, 6)))
check(lambda a: mx.softmax(a, axis=0), mx.random.uniform(shape=(3, 5)))
check(lambda a: mx.logsumexp(a, axis=-1), mx.random.uniform(shape=(3, 5)))
check(
    lambda a, b: mx.logaddexp(a, b),
    mx.random.uniform(shape=(3, 5)),
    mx.random.uniform(shape=(3, 5)),
)
check(lambda a: mx.sinh(a) + mx.cosh(a), mx.random.uniform(shape=(3, 5)))
check(lambda a: mx.expm1(a) + mx.log1p(a), mx.random.uniform(shape=(3, 5)))
check(lambda a: mx.log2(a), mx.random.uniform(shape=(3, 5)) + 1)
check(lambda a: mx.log10(a), mx.random.uniform(shape=(3, 5)) + 1)
check_complex(
    lambda a: mx.conj(a),
    mx.array(np.array([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]], dtype=np.complex64)),
)

check(lambda a: mx.softmax(a, axis=-1), mx.random.uniform(shape=(3, 5)) * 400 - 200)
check(lambda a: mx.logsumexp(a, axis=-1), mx.random.uniform(shape=(3, 5)) * 400 - 200)

check(lambda a: tuple(mx.split(a, 2, axis=1)), mx.random.uniform(shape=(2, 6)))
check(lambda a: tuple(mx.split(a, [1, 4], axis=1)), mx.random.uniform(shape=(2, 6)))
check(lambda a: mx.pad(a, [(1, 1), (0, 2)]), mx.random.uniform(shape=(2, 4)))
check(
    lambda c, a, b: mx.addmm(c, a, b),
    mx.random.uniform(shape=(2, 4)),
    mx.random.uniform(shape=(2, 3)),
    mx.random.uniform(shape=(3, 4)),
)
check(
    lambda c, a, b: mx.addmm(c, a, b, alpha=2.0, beta=0.5),
    mx.random.uniform(shape=(2, 4)),
    mx.random.uniform(shape=(2, 3)),
    mx.random.uniform(shape=(3, 4)),
)
check(lambda a: mx.arange(6).astype(mx.float32) + a, mx.random.uniform(shape=(6,)))
check(lambda a: mx.arange(2.0, 14.0, 2.0) + a, mx.random.uniform(shape=(6,)))

u = lambda: mx.random.uniform(shape=(3, 5)) * 1.6 - 0.8
check(lambda a: mx.arcsin(a) + mx.arccos(a), u())
check(lambda a: mx.arctan(a) + mx.arcsinh(a), u())
check(lambda a: mx.arctanh(a), u())
check(lambda a: mx.arccosh(a), mx.random.uniform(shape=(3, 5)) * 2 + 1)
check(lambda a, b: mx.arctan2(a, b), u(), u())
check(lambda a: mx.erf(a), u())
check(lambda a: mx.erfinv(a), u())

check(
    lambda a, i: mx.slice(a, i, axes=[0, 1], slice_size=[2, 3]),
    mx.random.uniform(shape=(6, 8)),
    mx.array([1, 2], dtype=mx.uint32),
)
check(
    lambda a, i: mx.slice(a, i, axes=[1], slice_size=[6, 3]),
    mx.random.uniform(shape=(6, 8)),
    mx.array([2], dtype=mx.uint32),
)
check(
    lambda a, up, i: mx.slice_update(a, up, i, axes=[0, 1]),
    mx.random.uniform(shape=(6, 8)),
    mx.random.uniform(shape=(2, 3)),
    mx.array([1, 2], dtype=mx.uint32),
)


def _setslice(a, up):
    a[1:3, 2:5] = up
    return a


xv = np.random.rand(6, 8).astype(np.float32)
uv = np.random.rand(2, 3).astype(np.float32)
ref = np.array(_setslice(mx.array(xv), mx.array(uv)))
outs = run_xla(export_to_hlo(_setslice, mx.array(xv), mx.array(uv)), [xv, uv])
assert np.allclose(np.asarray(outs[0]).reshape(ref.shape), ref, atol=1e-4)
print("slice_update OK")


def _setstrided(a, up):
    a[1:6:2, 0:8:2] = up
    return a


try:
    export_to_hlo(
        _setstrided,
        mx.array(np.random.rand(6, 8).astype(np.float32)),
        mx.array(np.random.rand(3, 4).astype(np.float32)),
    )
    raise SystemExit("strided slice update should be rejected")
except NotImplementedError:
    print("strided slice_update rejected OK")

check(
    lambda x, w: mx.conv2d(x, w),
    mx.random.uniform(shape=(1, 8, 8, 3)),
    mx.random.uniform(shape=(4, 3, 3, 3)),
)
check(
    lambda x, w: mx.conv2d(x, w, stride=2, padding=1),
    mx.random.uniform(shape=(2, 9, 9, 3)),
    mx.random.uniform(shape=(5, 3, 3, 3)),
)
check(
    lambda x, w: mx.conv2d(x, w, groups=2),
    mx.random.uniform(shape=(1, 8, 8, 4)),
    mx.random.uniform(shape=(6, 3, 3, 2)),
)
check(
    lambda x, w: mx.conv1d(x, w),
    mx.random.uniform(shape=(1, 10, 3)),
    mx.random.uniform(shape=(4, 3, 3)),
)

check(lambda a: mx.sort(a, axis=-1), mx.random.uniform(shape=(2, 3, 4, 33)))
check(lambda a: mx.sort(a, axis=1), mx.random.uniform(shape=(8, 20, 5)))
check(lambda a: mx.sort(a, axis=0), mx.random.uniform(shape=(64, 128)))
check(lambda a: mx.argsort(a, axis=-1), mx.random.uniform(shape=(4, 6, 50)))
check(lambda a: mx.argsort(a, axis=1), mx.random.uniform(shape=(3, 40, 7)))
check(
    lambda a: mx.sort(a.astype(mx.int32), axis=-1),
    (mx.random.uniform(shape=(5, 64)) * 1000).astype(mx.int32),
)

# larger / higher-rank coverage for earlier ops
check(
    lambda a: mx.softmax(a, axis=-1), mx.random.uniform(shape=(4, 8, 32, 128)) * 20 - 10
)
check(lambda a: mx.softmax(a, axis=1), mx.random.uniform(shape=(2, 16, 64)))
check(
    lambda a, b: a @ b,
    mx.random.uniform(shape=(6, 4, 32, 48)),
    mx.random.uniform(shape=(6, 4, 48, 24)),
)
check(lambda a: mx.sum(a, axis=2), mx.random.uniform(shape=(3, 5, 40, 7)))
check(
    lambda a: mx.max(a, axis=(1, 3), keepdims=True),
    mx.random.uniform(shape=(2, 9, 4, 11)),
)
check(
    lambda x, w: mx.conv2d(x, w, stride=2, padding=1),
    mx.random.uniform(shape=(4, 32, 32, 16)),
    mx.random.uniform(shape=(32, 3, 3, 16)),
)

emb = mx.random.uniform(shape=(1000, 64))
check(
    lambda w, i: w[i], emb, mx.array(np.random.randint(0, 1000, (5,)), dtype=mx.uint32)
)
check(
    lambda w, i: w[i],
    emb,
    mx.array(np.random.randint(0, 1000, (4, 7)), dtype=mx.uint32),
)
check(
    lambda w, i: w[i],
    emb,
    mx.array(np.random.randint(0, 1000, (2, 3, 4)), dtype=mx.uint32),
)
check(
    lambda w, i: mx.take(w, i, axis=1),
    emb,
    mx.array(np.random.randint(0, 64, (8,)), dtype=mx.uint32),
)
check(
    lambda w, i: w[i],
    mx.random.uniform(shape=(50, 16, 32)),
    mx.array(np.random.randint(0, 50, (6,)), dtype=mx.uint32),
)
check(
    lambda w, i, j: w[i, j],
    mx.random.uniform(shape=(6, 7)),
    mx.array(np.random.randint(0, 6, (2, 1)), dtype=mx.uint32),
    mx.array(np.random.randint(0, 7, (1, 3)), dtype=mx.uint32),
)

# gather/scatter along an axis, and scatter reduce variants
check(
    lambda a, i: mx.take_along_axis(a, i, axis=1),
    mx.random.uniform(shape=(4, 20)),
    mx.array(np.random.randint(0, 20, (4, 6)), dtype=mx.uint32),
)
check(
    lambda a, i: mx.take_along_axis(a, i, axis=0),
    mx.random.uniform(shape=(8, 5, 3)),
    mx.array(np.random.randint(0, 8, (2, 5, 3)), dtype=mx.uint32),
)
check(
    lambda a, i, u: mx.put_along_axis(a, i, u, axis=1),
    mx.random.uniform(shape=(4, 20)),
    mx.array(np.random.randint(0, 20, (4, 3)), dtype=mx.uint32),
    mx.random.uniform(shape=(4, 3)),
)


def _scatterset(a, u):
    a[mx.array([0, 2, 5], dtype=mx.uint32)] = u
    return a


def _scatteradd(a, i, u):
    return a.at[i].add(u)


for label, fn2, arrs in [
    (
        "scatter_set",
        _scatterset,
        (mx.random.uniform(shape=(8, 7)), mx.random.uniform(shape=(3, 7))),
    ),
    (
        "scatter_add",
        _scatteradd,
        (
            mx.random.uniform(shape=(8, 7)),
            mx.array([0, 2, 2, 5], dtype=mx.uint32),
            mx.random.uniform(shape=(4, 7)),
        ),
    ),
]:
    fresh = [mx.array(np.array(a)) for a in arrs]
    ref = np.array(fn2(*[mx.array(np.array(a)) for a in arrs]))
    outs = run_xla(export_to_hlo(fn2, *fresh), [np.array(a) for a in arrs])
    assert np.allclose(np.asarray(outs[0]).reshape(ref.shape), ref, atol=1e-4), label
    print(label, "OK")


# --- complex multi-op compositions ---
def attention(q, k, v):
    scores = (q @ mx.swapaxes(k, -1, -2)) * (q.shape[-1] ** -0.5)
    return mx.softmax(scores, axis=-1) @ v


check(
    attention,
    mx.random.uniform(shape=(2, 4, 16, 32)),
    mx.random.uniform(shape=(2, 4, 16, 32)),
    mx.random.uniform(shape=(2, 4, 16, 32)),
)


def mlp_block(emb, tok, w1, b1, w2, b2):
    x = emb[tok]
    h = mx.maximum(x @ w1 + b1, 0.0)
    y = h @ w2 + b2
    return mx.softmax(y, axis=-1)


check(
    mlp_block,
    mx.random.uniform(shape=(100, 32)),
    mx.array(np.random.randint(0, 100, (8, 12)), dtype=mx.uint32),
    mx.random.uniform(shape=(32, 64)),
    mx.random.uniform(shape=(64,)),
    mx.random.uniform(shape=(64, 10)),
    mx.random.uniform(shape=(10,)),
)


def layernorm_gelu(x, g, b):
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mu) ** 2, axis=-1, keepdims=True)
    xn = (x - mu) * mx.rsqrt(var + 1e-5) * g + b
    return xn * 0.5 * (1 + mx.erf(xn / 2**0.5))


check(
    layernorm_gelu,
    mx.random.uniform(shape=(4, 16, 48)) * 4 - 2,
    mx.random.uniform(shape=(48,)),
    mx.random.uniform(shape=(48,)),
)


# --- full tiny transformer training step (forward + backward + SGD) ---
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


_params = [
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
_x = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
_y = mx.array(np.random.randint(0, V, (B, T)), dtype=mx.uint32)
check(train_step, *_params, _x, _y)

# fast ops (fall back to / decompose into supported ops) + faithful RNG
check(lambda k: mx.random.uniform(shape=(64,), key=k), mx.random.key(0))
check(lambda k: mx.random.normal(shape=(32,), key=k), mx.random.key(7))
check(
    lambda x, w: mx.fast.rms_norm(x, w, 1e-5),
    mx.random.uniform(shape=(4, 12, 8)),
    mx.random.uniform(shape=(8,)),
)
check(
    lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
    mx.random.uniform(shape=(4, 12, 8)),
    mx.random.uniform(shape=(8,)),
    mx.random.uniform(shape=(8,)),
)
_qkv = mx.random.uniform(shape=(2, 4, 16, 8))
check(
    lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.35),
    _qkv,
    _qkv,
    _qkv,
)

# multi-step convergence: MLX loop vs exported-step run repeatedly under XLA
mp = [mx.array(np.array(p)) for p in _params]
xp = [np.array(p) for p in _params]
xn, yn = np.array(_x), np.array(_y)
mlx_losses, xla_losses = [], []
for _ in range(5):
    mo = train_step(*mp, _x, _y)
    mlx_losses.append(float(mo[0]))
    mp = list(mo[1:])
    outs = run_xla(
        export_to_hlo(train_step, *[mx.array(p) for p in xp], _x, _y), [*xp, xn, yn]
    )
    xla_losses.append(float(np.asarray(outs[0]).reshape(())))
    xp = [
        np.asarray(o).reshape(np.array(_params[i]).shape)
        for i, o in enumerate(outs[1:])
    ]
assert np.allclose(mlx_losses, xla_losses, atol=1e-3), (mlx_losses, xla_losses)
print("multistep loss mlx:", [round(x, 4) for x in mlx_losses])
print("multistep OK")

# bf16 forward
q32 = np.random.rand(2, 4, 16, 8).astype(np.float32)
qm = mx.array(q32).astype(mx.bfloat16)
bref = np.array(attention(qm, qm, qm).astype(mx.float32))
import jax.numpy as jnp  # noqa: E402

qbf = jnp.asarray(q32).astype(jnp.bfloat16)
bouts = run_xla(export_to_hlo(attention, qm, qm, qm), [qbf, qbf, qbf])
bo = np.asarray(bouts[0]).astype(np.float32).reshape(bref.shape)
assert np.allclose(bo, bref, atol=2e-2), np.abs(bo - bref).max()
print("bf16 attention max err:", float(np.abs(bo - bref).max()), "OK")


def check_cpu(fn, *arrs):
    # Trace under a CPU stream so fused fast:: ops decompose to base ops.
    out = fn(*arrs)
    ref = [np.array(o) for o in (out if isinstance(out, tuple) else (out,))]
    with mx.stream(mx.cpu):
        text = export_to_hlo(fn, *arrs)
    outs = run_xla(text, [np.array(a) for a in arrs])
    for r, o in zip(ref, outs):
        assert np.allclose(np.asarray(o).reshape(r.shape), r, atol=1e-4), "mismatch"
    print("OK")


rope = mx.fast.rope
check_cpu(
    lambda x: rope(x, 8, traditional=False, base=1e4, scale=1.0, offset=0),
    mx.random.uniform(shape=(2, 3, 4, 8)),
)
check_cpu(
    lambda x: rope(x, 8, traditional=True, base=1e4, scale=1.0, offset=2),
    mx.random.uniform(shape=(2, 3, 4, 8)),
)
check_cpu(
    lambda x: rope(x, 4, traditional=False, base=1e4, scale=1.0, offset=0),
    mx.random.uniform(shape=(1, 2, 4, 8)),
)

# opt-in composites: named op + decomposition body
_cx = mx.random.uniform(shape=(3, 8))
_cw = mx.random.uniform(shape=(8,))
_cb = mx.random.uniform(shape=(8,))
_ct = export_to_hlo(
    lambda x, w: mx.fast.rms_norm(x, w, 1e-5), _cx, _cw, composites={"RMSNorm"}
)
assert 'stablehlo.composite "mlx.rms_norm"' in _ct
assert "func.func private @mlx_rms_norm_0" in _ct
_co = run_xla(_ct, [np.array(_cx), np.array(_cw)])
assert np.allclose(
    np.asarray(_co[0]).reshape(3, 8),
    np.array(mx.fast.rms_norm(_cx, _cw, 1e-5)),
    atol=1e-4,
)
_lt = export_to_hlo(
    lambda x, w, b: mx.fast.layer_norm(x, w, b, 1e-5),
    _cx,
    _cw,
    _cb,
    composites={"LayerNorm"},
)
assert 'stablehlo.composite "mlx.layer_norm"' in _lt
_lo = run_xla(_lt, [np.array(_cx), np.array(_cw), np.array(_cb)])
assert np.allclose(
    np.asarray(_lo[0]).reshape(3, 8),
    np.array(mx.fast.layer_norm(_cx, _cw, _cb, 1e-5)),
    atol=1e-4,
)
print("composite OK")

# pytree bridge: export a real nn.Module training step (params = nested dict)
import mlx.nn as nn  # noqa: E402
from mlx.export_hlo import export_tree, flatten_args  # noqa: E402
from mlx.utils import tree_flatten as _tf  # noqa: E402
from mlx.utils import tree_map as _tm

_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
mx.eval(_model.parameters())
_p0 = _model.parameters()


def _mloss(params, x, y):
    _model.update(params)
    return mx.mean((_model(x) - y) ** 2)


_gfn = mx.value_and_grad(_mloss)


def _mstep(params, x, y):
    loss, grads = _gfn(params, x, y)
    return loss, _tm(lambda p, g: p - 0.1 * g, params, grads)


_xd = mx.random.uniform(shape=(5, 8))
_yd = mx.random.uniform(shape=(5, 4))
_ref = [np.array(v) for _, v in _tf(_mstep(_p0, _xd, _yd))]
_hlo, _ok = export_tree(_mstep, _p0, _xd, _yd)
_fo = run_xla(_hlo, [np.array(a) for a in flatten_args(_p0, _xd, _yd)])
assert len(_fo) == len(_ref), (len(_fo), len(_ref))
for r, o in zip(_ref, _fo):
    assert np.allclose(np.asarray(o).reshape(r.shape), r, atol=1e-4), "pytree mismatch"
print("pytree module step OK:", len(_ref), "leaves, keys:", _ok[:3], "...")

# Adam training step in bf16 (bf16 params/grads, fp32 optimizer state), via pytree bridge
_am = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
mx.eval(_am.parameters())
_ap = _tm(lambda p: p.astype(mx.bfloat16), _am.parameters())
_amm = _tm(lambda p: mx.zeros(p.shape, dtype=mx.float32), _ap)
_avv = _tm(lambda p: mx.zeros(p.shape, dtype=mx.float32), _ap)
_B1, _B2, _LR, _EPS = 0.9, 0.999, 0.01, 1e-8


def _aloss(params, x, y):
    _am.update(params)
    return mx.mean((_am(x) - y) ** 2)


def _adam(params, m, v, t, x, y):
    loss, grads = mx.value_and_grad(_aloss)(params, x, y)
    t2 = t + 1.0
    nm = _tm(lambda mi, g: _B1 * mi + (1 - _B1) * g.astype(mx.float32), m, grads)
    nv = _tm(lambda vi, g: _B2 * vi + (1 - _B2) * g.astype(mx.float32) ** 2, v, grads)
    bc1, bc2 = 1 - _B1**t2, 1 - _B2**t2
    np_ = _tm(
        lambda p, mi, vi: p
        - (_LR * (mi / bc1) / (mx.sqrt(vi / bc2) + _EPS)).astype(p.dtype),
        params,
        nm,
        nv,
    )
    return loss, np_, nm, nv, t2


_axd = mx.random.uniform(shape=(6, 8)).astype(mx.bfloat16)
_ayd = mx.random.uniform(shape=(6, 4)).astype(mx.bfloat16)
_at = mx.array(0.0, dtype=mx.float32)
_aref = [
    np.array(x.astype(mx.float32))
    for _, x in _tf(_adam(_ap, _amm, _avv, _at, _axd, _ayd))
]
_ahlo, _aok = export_tree(_adam, _ap, _amm, _avv, _at, _axd, _ayd)


def _to_jax(a):
    n = np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)
    return jnp.asarray(n).astype(jnp.bfloat16) if a.dtype == mx.bfloat16 else n


_aouts = run_xla(
    _ahlo, [_to_jax(a) for a in flatten_args(_ap, _amm, _avv, _at, _axd, _ayd)]
)
_maxerr = 0.0
for r, o in zip(_aref, _aouts):
    o = np.array(jnp.asarray(o).astype(jnp.float32)).reshape(r.shape)
    _maxerr = max(_maxerr, float(np.abs(o - r).max()))
    assert np.allclose(o, r, atol=3e-2), np.abs(o - r).max()
print("bf16 Adam step OK:", len(_aref), "leaves, max err", round(_maxerr, 5))
