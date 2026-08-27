#!/usr/bin/env python3
"""Compile+runtime benchmark: MLX->StableHLO export vs native JAX.

Same Llama-style decoder + bf16 Adam step as
``test_export_hlo_adam_transformer.py``, expressed two ways, timed as a median
over N independent trials for three phases each:

  build    MLX: export_tree (graph build + HLO emit); JAX: trace + lower
  compile  StableHLO -> XLA executable
  run      one train step, median per-iter (blocked)

Swept over Llama-3 GQA presets (1b/3b/8b) and device layouts:
  single   one device, replicated
  dp8      pure data-parallel over 8 devices (batch sharded)
  mesh222  (dp,tp,cp)=(2,2,2) FSDP+TP+CP, same shardings as the test

Run on the 8-device TPU slice, e.g.:
  python bench_export_hlo_transformer.py --preset 1b 3b 8b --layout single dp8 mesh222
"""

import argparse
import os
import statistics
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
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

_perf = time.perf_counter
_JDTYPE = {mx.bfloat16: jnp.bfloat16, mx.float32: np.float32, mx.uint32: np.uint32}


def _to_jax(a):
    if a.dtype == mx.bfloat16:
        return jnp.asarray(np.array(a.astype(mx.float32))).astype(jnp.bfloat16)
    return np.array(a)


def _f32(o):
    return np.array(jnp.asarray(o).astype(jnp.float32))


def make_config(depth, width, head_dim, ff_mult, B, T, V):
    assert width % head_dim == 0, f"width {width} not divisible by head_dim {head_dim}"
    return SimpleNamespace(
        B=B,
        T=T,
        V=V,
        D=width,
        H=width // head_dim,
        KV=width // head_dim,  # MHA
        HD=head_dim,
        L=depth,
        FF=ff_mult * width,
        EPS=1e-5,
        BASE=10000.0,
        B1=0.9,
        B2=0.999,
        LR=1e-3,
        ADAM_EPS=1e-8,
    )


# Real Llama-3 configs: (layers, d_model, heads, kv_heads, head_dim, swiglu_inter, vocab).
PRESETS = {
    "1b": (16, 2048, 32, 8, 64, 8192, 128256),
    "3b": (28, 3072, 24, 8, 128, 8192, 128256),
    "8b": (32, 4096, 32, 8, 128, 14336, 128256),
}


def preset_config(name, B, T):
    L, D, H, KV, HD, FF, V = PRESETS[name]
    assert H * HD == D and H % KV == 0
    return SimpleNamespace(
        name=name,
        B=B,
        T=T,
        V=V,
        D=D,
        H=H,
        KV=KV,
        HD=HD,
        L=L,
        FF=FF,
        EPS=1e-5,
        BASE=500000.0,
        B1=0.9,
        B2=0.999,
        LR=1e-3,
        ADAM_EPS=1e-8,
    )


def n_params(cfg):
    return 1 + 9 * cfg.L + 2


def make_model_fns(cfg):
    B, T, V, D, H, HD, L, FF = cfg.B, cfg.T, cfg.V, cfg.D, cfg.H, cfg.HD, cfg.L, cfg.FF
    KV, KVD = cfg.KV, cfg.KV * cfg.HD
    rep = H // KV
    EPS, BASE = cfg.EPS, cfg.BASE
    B1, B2, LR, AE = cfg.B1, cfg.B2, cfg.LR, cfg.ADAM_EPS
    npar = n_params(cfg)

    # ---- MLX: Llama-style decoder with grouped-query attention ----
    def m_repeat_kv(z):  # [B, KV, T, HD] -> [B, H, T, HD]
        z = mx.broadcast_to(mx.expand_dims(z, 2), (B, KV, rep, T, HD))
        return mx.reshape(z, (B, H, T, HD))

    def m_attention(x, wq, wk, wv, wo, mask):
        q = mx.swapaxes(mx.reshape(x @ wq, (B, T, H, HD)), 1, 2)
        k = mx.swapaxes(mx.reshape(x @ wk, (B, T, KV, HD)), 1, 2)
        v = mx.swapaxes(mx.reshape(x @ wv, (B, T, KV, HD)), 1, 2)
        q = mx.fast.rope(q, HD, traditional=False, base=BASE, scale=1.0, offset=0)
        k = mx.fast.rope(k, HD, traditional=False, base=BASE, scale=1.0, offset=0)
        k, v = m_repeat_kv(k), m_repeat_kv(v)
        scores = (q @ mx.swapaxes(k, -1, -2)) * (HD**-0.5)
        scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
        a = mx.softmax(scores, axis=-1) @ v
        a = mx.reshape(mx.swapaxes(a, 1, 2), (B, T, D))
        return a @ wo

    def m_swiglu(x, wg, wu, wd):
        g = x @ wg
        return (g * mx.sigmoid(g)) * (x @ wu) @ wd

    def m_forward(params, tokens):
        emb, g_final, lm_head = params[0], params[-2], params[-1]
        layers = params[1:-2]
        idx = mx.arange(T)
        mask = idx[:, None] >= idx[None, :]
        h = emb[tokens]
        for li in range(L):
            g_attn, wq, wk, wv, wo, g_ffn, wg, wu, wd = layers[li * 9 : li * 9 + 9]
            h = h + m_attention(mx.fast.rms_norm(h, g_attn, EPS), wq, wk, wv, wo, mask)
            h = h + m_swiglu(mx.fast.rms_norm(h, g_ffn, EPS), wg, wu, wd)
        return mx.fast.rms_norm(h, g_final, EPS) @ lm_head

    def m_loss(params, x, y):
        logits = m_forward(params, x).astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        tgt = mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        return mx.mean(lse - tgt)

    def mlx_adam_step(params, m, v, t, x, y):
        loss, grads = mx.value_and_grad(lambda p: m_loss(p, x, y))(params)
        t2 = t + 1.0
        m2 = [B1 * mi + (1 - B1) * g.astype(mx.float32) for mi, g in zip(m, grads)]
        v2 = [B2 * vi + (1 - B2) * g.astype(mx.float32) ** 2 for vi, g in zip(v, grads)]
        bc1, bc2 = 1 - B1**t2, 1 - B2**t2
        p2 = [
            p - (LR * (mi / bc1) / (mx.sqrt(vi / bc2) + AE)).astype(p.dtype)
            for p, mi, vi in zip(params, m2, v2)
        ]
        return loss, p2, m2, v2, t2

    def _rp(*shape):
        a = (np.random.rand(*shape) * 0.2 - 0.1).astype(np.float32)
        return mx.array(a).astype(mx.bfloat16)

    def mlx_make_params():
        params = [_rp(V, D)]
        for _ in range(L):
            params += [
                _rp(D),
                _rp(D, D),
                _rp(D, KVD),
                _rp(D, KVD),
                _rp(D, D),
                _rp(D),
                _rp(D, FF),
                _rp(D, FF),
                _rp(FF, D),
            ]
        params += [_rp(D), _rp(D, V)]
        return params

    # ---- JAX (semantically identical) ----
    def j_rope(x):
        half = HD // 2
        inv = BASE ** (-(2.0 * jnp.arange(half, dtype=jnp.float32)) / HD)
        ang = jnp.arange(T, dtype=jnp.float32)[:, None] * inv[None, :]
        cos = jnp.cos(ang).astype(x.dtype)
        sin = jnp.sin(ang).astype(x.dtype)
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def j_rms(x, g):
        xf = x.astype(jnp.float32)
        n = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS)
        return n.astype(x.dtype) * g

    def j_attention(x, wq, wk, wv, wo, mask):
        q = jnp.swapaxes(jnp.reshape(x @ wq, (B, T, H, HD)), 1, 2)
        k = jnp.swapaxes(jnp.reshape(x @ wk, (B, T, KV, HD)), 1, 2)
        v = jnp.swapaxes(jnp.reshape(x @ wv, (B, T, KV, HD)), 1, 2)
        q, k = j_rope(q), j_rope(k)
        k = jnp.reshape(
            jnp.broadcast_to(jnp.expand_dims(k, 2), (B, KV, rep, T, HD)), (B, H, T, HD)
        )
        v = jnp.reshape(
            jnp.broadcast_to(jnp.expand_dims(v, 2), (B, KV, rep, T, HD)), (B, H, T, HD)
        )
        scores = (q @ jnp.swapaxes(k, -1, -2)) * (HD**-0.5)
        scores = jnp.where(mask, scores, jnp.array(-1e9, scores.dtype))
        a = jax.nn.softmax(scores, axis=-1) @ v
        a = jnp.reshape(jnp.swapaxes(a, 1, 2), (B, T, D))
        return a @ wo

    def j_swiglu(x, wg, wu, wd):
        g = x @ wg
        return (g * jax.nn.sigmoid(g)) * (x @ wu) @ wd

    def j_forward(params, tokens):
        emb, g_final, lm_head = params[0], params[-2], params[-1]
        layers = params[1:-2]
        idx = jnp.arange(T)
        mask = idx[:, None] >= idx[None, :]
        h = emb[tokens]
        for li in range(L):
            g_attn, wq, wk, wv, wo, g_ffn, wg, wu, wd = layers[li * 9 : li * 9 + 9]
            h = h + j_attention(j_rms(h, g_attn), wq, wk, wv, wo, mask)
            h = h + j_swiglu(j_rms(h, g_ffn), wg, wu, wd)
        return j_rms(h, g_final) @ lm_head

    def j_loss(params, x, y):
        logits = j_forward(params, x).astype(jnp.float32)
        lse = jax.scipy.special.logsumexp(logits, axis=-1)
        tgt = jnp.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        return jnp.mean(lse - tgt)

    def jax_flat_step(*flat):
        params = list(flat[:npar])
        m = list(flat[npar : 2 * npar])
        v = list(flat[2 * npar : 3 * npar])
        t, x, y = flat[3 * npar], flat[3 * npar + 1], flat[3 * npar + 2]
        loss, grads = jax.value_and_grad(lambda p: j_loss(p, x, y))(params)
        t2 = t + 1.0
        m2 = [B1 * mi + (1 - B1) * g.astype(jnp.float32) for mi, g in zip(m, grads)]
        v2 = [
            B2 * vi + (1 - B2) * g.astype(jnp.float32) ** 2 for vi, g in zip(v, grads)
        ]
        bc1, bc2 = 1 - B1**t2, 1 - B2**t2
        p2 = [
            p - (LR * (mi / bc1) / (jnp.sqrt(vi / bc2) + AE)).astype(p.dtype)
            for p, mi, vi in zip(params, m2, v2)
        ]
        return (loss, *p2, *m2, *v2, t2)

    return SimpleNamespace(
        mlx_adam_step=mlx_adam_step,
        mlx_make_params=mlx_make_params,
        jax_flat_step=jax_flat_step,
    )


def _param_specs(cfg):
    DP, TP = "dp", "tp"
    specs = [P(DP, TP)]
    for _ in range(cfg.L):
        specs += [
            P(TP),
            P(DP, TP),
            P(DP, TP),
            P(DP, TP),
            P(TP, DP),
            P(TP),
            P(DP, TP),
            P(DP, TP),
            P(TP, DP),
        ]
    specs += [P(TP), P(DP, TP)]
    return specs


def build_shardings(cfg, layout, devices):
    npar = n_params(cfg)
    if layout == "single":
        mesh = Mesh(np.array(devices[:1]).reshape(1, 1, 1), ("dp", "tp", "cp"))
        ps, xs, ys = [P()] * npar, P(), P()
    elif layout == "dp8":
        mesh = Mesh(np.array(devices[:8]).reshape(8, 1, 1), ("dp", "tp", "cp"))
        ps, xs, ys = [P()] * npar, P("dp"), P("dp")
    elif layout == "mesh222":
        mesh = Mesh(np.array(devices[:8]).reshape(2, 2, 2), ("dp", "tp", "cp"))
        ps, xs, ys = _param_specs(cfg), P("dp", "cp"), P("dp", "cp")
    else:
        raise ValueError(layout)
    in_specs = ps + ps + ps + [P(), xs, ys]
    out_specs = [P()] + ps + ps + ps + [P()]
    in_sh = [NamedSharding(mesh, s) for s in in_specs]
    out_sh = [NamedSharding(mesh, s) for s in out_specs]
    return in_sh, out_sh


def to_exported(text, in_avals, out_avals, platform):
    backend = xb.get_backend(platform)
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
        platforms=(platform,),
        ordered_effects=(),
        unordered_effects=(),
        disabled_safety_checks=(),
        mlir_module_serialized=blob,
        calling_convention_version=jexport.maximum_supported_calling_convention_version,
        module_kept_var_idx=tuple(range(ni)),
        uses_global_constants=False,
        _get_vjp=None,
    )


def _time_run(compiled, puts, warmup, iters):
    for _ in range(warmup):
        jax.block_until_ready(compiled(*puts))
    ts = []
    for _ in range(iters):
        t0 = _perf()
        jax.block_until_ready(compiled(*puts))
        ts.append(_perf() - t0)
    return statistics.median(ts)


def _dump_hlo(dump_dir, tag, mlx_stablehlo, jax_lowered, mlx_comp, jax_comp):
    # StableHLO = what each path hands to XLA; optimized = post-XLA (shows fusions/ops).
    files = {
        "mlx.stablehlo.mlir": mlx_stablehlo,
        "jax.stablehlo.mlir": jax_lowered.as_text(),
        "mlx.optimized.hlo": mlx_comp.as_text(),
        "jax.optimized.hlo": jax_comp.as_text(),
    }
    for suffix, content in files.items():
        with open(os.path.join(dump_dir, f"{tag}.{suffix}"), "w") as f:
            f.write(content)
    print(f"  dumped {tag}.*.{{mlir,hlo}} to {dump_dir}")


def bench(
    cfg, layout, devices, platform, trials, warmup, run_iters, dump_dir=None, tag=""
):
    fns = make_model_fns(cfg)
    params = fns.mlx_make_params()
    m = [mx.zeros(p.shape, dtype=mx.float32) for p in params]
    v = [mx.zeros(p.shape, dtype=mx.float32) for p in params]
    t = mx.array(0.0, dtype=mx.float32)
    x = mx.array(np.random.randint(0, cfg.V, (cfg.B, cfg.T)), dtype=mx.uint32)
    y = mx.array(np.random.randint(0, cfg.V, (cfg.B, cfg.T)), dtype=mx.uint32)
    mx_args = (params, m, v, t, x, y)
    flat_in = flatten_args(*mx_args)

    ref_leaves = [o for _, o in tree_flatten(fns.mlx_adam_step(*mx_args))]
    ref = [np.array(o.astype(mx.float32)) for o in ref_leaves]
    in_avals = [jax.core.ShapedArray(a.shape, _JDTYPE[a.dtype]) for a in flat_in]
    out_avals = [jax.core.ShapedArray(o.shape, _JDTYPE[o.dtype]) for o in ref_leaves]

    in_sh, out_sh = build_shardings(cfg, layout, devices)
    puts = [jax.device_put(_to_jax(a), s) for a, s in zip(flat_in, in_sh)]

    b_ml, c_ml, r_ml = [], [], []
    b_jx, c_jx, r_jx = [], [], []
    mlx_comp = jax_comp = None
    for _ in range(trials):
        jax.clear_caches()
        # --- MLX path: build = export_tree; compile = HLO -> executable ---
        t0 = _perf()
        text, _ = export_tree(fns.mlx_adam_step, *mx_args)
        t1 = _perf()
        exp = to_exported(text, in_avals, out_avals, platform)
        jf = jax.jit(exp.call, in_shardings=tuple(in_sh), out_shardings=tuple(out_sh))
        mlx_comp = jf.lower(*puts).compile()
        t2 = _perf()
        b_ml.append(t1 - t0)
        c_ml.append(t2 - t1)
        r_ml.append(_time_run(mlx_comp, puts, warmup, run_iters))

        jax.clear_caches()
        # --- JAX path: build = trace + lower; compile = lower -> executable ---
        t0 = _perf()
        jf = jax.jit(
            fns.jax_flat_step, in_shardings=tuple(in_sh), out_shardings=tuple(out_sh)
        )
        low = jf.lower(*puts)
        t1 = _perf()
        jax_comp = low.compile()
        t2 = _perf()
        b_jx.append(t1 - t0)
        c_jx.append(t2 - t1)
        r_jx.append(_time_run(jax_comp, puts, warmup, run_iters))

    if dump_dir:
        _dump_hlo(dump_dir, tag, text, low, mlx_comp, jax_comp)

    # parity: native JAX and MLX-exported vs the MLX reference
    err_jx = max(
        float(np.abs(_f32(o).reshape(r.shape) - r).max())
        for o, r in zip(jax_comp(*puts), ref)
    )
    err_ml = max(
        float(np.abs(_f32(o).reshape(r.shape) - r).max())
        for o, r in zip(mlx_comp(*puts), ref)
    )
    med = statistics.median
    return SimpleNamespace(
        mlx_build=med(b_ml),
        mlx_comp=med(c_ml),
        mlx_run=med(r_ml),
        jax_build=med(b_jx),
        jax_comp=med(c_jx),
        jax_run=med(r_jx),
        err_jx=err_jx,
        err_ml=err_ml,
    )


def _fmt_row(cells, widths):
    return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"


def format_table(layout, rows):
    head = [
        "size",
        "mlx_build",
        "mlx_comp",
        "mlx_run",
        "jax_build",
        "jax_comp",
        "jax_run",
        "run x",
        "err(jx/ml)",
    ]
    lines = [[str(c) for c in r] for r in rows]
    widths = [max(len(head[i]), *(len(l[i]) for l in lines)) for i in range(len(head))]
    out = [f"\n### layout = {layout}  (times in ms, median; run = per-iter)"]
    out.append(_fmt_row(head, widths))
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    out += [_fmt_row(l, widths) for l in lines]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        nargs="+",
        default=["1b", "3b", "8b"],
        choices=list(PRESETS),
        help="Llama-3 presets to sweep",
    )
    ap.add_argument(
        "--layout",
        nargs="+",
        default=["dp8", "mesh222"],
        choices=["single", "dp8", "mesh222"],
    )
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--run-iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048, help="realistic LLM context")
    ap.add_argument(
        "--platform", default=None, help="xla platform; default = jax default"
    )
    ap.add_argument("--out", default="bench_results.md", help="write results here too")
    ap.add_argument(
        "--dump-hlo",
        default=None,
        help="dir to write StableHLO + optimized HLO per (preset, layout)",
    )
    args = ap.parse_args()

    mx.random.seed(0)
    np.random.seed(0)
    platform = args.platform or xb.get_backend().platform
    devices = jax.devices(platform)
    need = {"single": 1, "dp8": 8, "mesh222": 8}
    dump_dir = os.path.expanduser(args.dump_hlo) if args.dump_hlo else None
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    report = [
        "# export-hlo transformer bench (Llama-3 GQA presets)",
        f"platform={platform}  devices={len(devices)}  trials={args.trials}  "
        f"run_iters={args.run_iters}  batch={args.batch}  seqlen={args.seqlen}",
    ]
    print(report[0])
    print(report[1])

    for layout in args.layout:
        if len(devices) < need[layout]:
            msg = f"\n[skip] layout {layout}: needs {need[layout]} devices, have {len(devices)}"
            print(msg)
            report.append(msg)
            continue
        rows = []
        for name in args.preset:
            cfg = preset_config(name, args.batch, args.seqlen)
            try:
                r = bench(
                    cfg,
                    layout,
                    devices,
                    platform,
                    args.trials,
                    args.warmup,
                    args.run_iters,
                    dump_dir=dump_dir,
                    tag=f"{name}_{layout}",
                )
            except Exception as e:  # noqa: BLE001
                rows.append(
                    [name, f"FAIL: {type(e).__name__}", "", "", "", "", "", "", ""]
                )
                print(f"  {name} {layout}: FAIL {type(e).__name__}: {e}")
                continue
            rows.append(
                [
                    name,
                    f"{r.mlx_build * 1e3:.1f}",
                    f"{r.mlx_comp * 1e3:.1f}",
                    f"{r.mlx_run * 1e3:.3f}",
                    f"{r.jax_build * 1e3:.1f}",
                    f"{r.jax_comp * 1e3:.1f}",
                    f"{r.jax_run * 1e3:.3f}",
                    f"{r.mlx_run / r.jax_run:.2f}",
                    f"{r.err_jx:.1e}/{r.err_ml:.1e}",
                ]
            )
        table = format_table(layout, rows)
        print(table)
        report.append(table)

    with open(args.out, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
