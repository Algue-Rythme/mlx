# Copyright © 2025 Apple Inc.

from typing import Any, Callable

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

_DTYPES = {
    mx.bool_: "i1",
    mx.uint8: "ui8",
    mx.uint16: "ui16",
    mx.uint32: "ui32",
    mx.uint64: "ui64",
    mx.int8: "i8",
    mx.int16: "i16",
    mx.int32: "i32",
    mx.int64: "i64",
    mx.float16: "f16",
    mx.bfloat16: "bf16",
    mx.float32: "f32",
    mx.float64: "f64",
    mx.complex64: "complex<f32>",
}

_FLOATS = {"f16", "bf16", "f32", "f64"}

_MAX_CONST_SIZE = 1 << 16

_LOG2E = 1.4426950408889634
_LOG10E = 0.4342944819032518

_PRECISION = {"default": "DEFAULT", "high": "HIGH", "highest": "HIGHEST"}

_UNARY = {
    "Abs": "abs",
    "Negative": "negate",
    "Exp": "exponential",
    "Sign": "sign",
    "Floor": "floor",
    "Ceil": "ceil",
    "Round": "round_nearest_even",
    "Sin": "sine",
    "Cos": "cosine",
    "Tan": "tan",
    "Tanh": "tanh",
    "Sigmoid": "logistic",
    "LogicalNot": "not",
    "Expm1": "exponential_minus_one",
    "Log1p": "log_plus_one",
}

_BINARY = {
    "Add": "add",
    "Subtract": "subtract",
    "Multiply": "multiply",
    "Divide": "divide",
    "Maximum": "maximum",
    "Minimum": "minimum",
    "Power": "power",
    "Remainder": "remainder",
    "LogicalAnd": "and",
    "LogicalOr": "or",
}

_COMPARE = {
    "Equal": "EQ",
    "NotEqual": "NE",
    "Greater": "GT",
    "GreaterEqual": "GE",
    "Less": "LT",
    "LessEqual": "LE",
}

_BITWISE = {0: "and", 1: "or", 2: "xor", 3: "shift_left", 4: "shift_right_logical"}

_RESHAPE = {"Reshape", "ExpandDims", "Squeeze", "Flatten", "Unflatten"}

_PASSTHROUGH = {"Full", "Copy", "Contiguous", "StopGradient"}

_CHLO_UNARY = {
    "ArcSin": "asin",
    "ArcCos": "acos",
    "ArcTan": "atan",
    "ArcSinh": "asinh",
    "ArcCosh": "acosh",
    "ArcTanh": "atanh",
    "Erf": "erf",
    "ErfInv": "erf_inv",
}

_REDUCE = {
    0: ("and", "and"),
    1: ("or", "or"),
    2: ("add", "sum"),
    3: ("multiply", "prod"),
    4: ("minimum", "min"),
    5: ("maximum", "max"),
}

_INF = {
    "f16": ("0x7C00", "0xFC00"),
    "bf16": ("0x7F80", "0xFF80"),
    "f32": ("0x7F800000", "0xFF800000"),
    "f64": ("0x7FF0000000000000", "0xFFF0000000000000"),
}


def _tystr(shape, el):
    if len(shape) == 0:
        return f"tensor<{el}>"
    return "tensor<" + "x".join(str(d) for d in shape) + f"x{el}>"


def _type(shape, dtype):
    return _tystr(shape, _DTYPES[dtype])


def _v(name):
    return "%" + name


def _reduce(dst, op, src, init, axes, src_ty, el, out_ty):
    dims = ", ".join(str(a) for a in axes)
    return (
        f"{dst} = stablehlo.reduce({src} init: {init}) applies stablehlo.{op} "
        f"across dimensions = [{dims}] : ({src_ty}, tensor<{el}>) -> {out_ty}"
    )


def _bcast(dst, src, dims, src_ty, out_ty):
    d = ", ".join(str(x) for x in dims)
    return (
        f"{dst} = stablehlo.broadcast_in_dim {src}, dims = [{d}] : "
        f"({src_ty}) -> {out_ty}"
    )


def _softmax(p):
    """Stable softmax over the last axis: exp(x - max) / sum(exp(x - max))."""
    x = _v(p["inputs"][0][0])
    o, shape, dtype = p["outputs"][0]
    el = _DTYPES[dtype]
    ty = _tystr(shape, el)
    rty = _tystr(shape[:-1], el)
    ax = len(shape) - 1
    kept = range(ax)
    n = {s: _v(o + s) for s in ("mi", "m", "mb", "s", "e", "zi", "ss", "sb")}
    return [
        f"{n['mi']} = stablehlo.constant dense<{_INF[el][1]}> : tensor<{el}>",
        _reduce(n["m"], "maximum", x, n["mi"], [ax], ty, el, rty),
        _bcast(n["mb"], n["m"], kept, rty, ty),
        f"{n['s']} = stablehlo.subtract {x}, {n['mb']} : {ty}",
        f"{n['e']} = stablehlo.exponential {n['s']} : {ty}",
        f"{n['zi']} = stablehlo.constant dense<0.0> : tensor<{el}>",
        _reduce(n["ss"], "add", n["e"], n["zi"], [ax], ty, el, rty),
        _bcast(n["sb"], n["ss"], kept, rty, ty),
        f"{_v(o)} = stablehlo.divide {n['e']}, {n['sb']} : {ty}",
    ]


def _logsumexp(p):
    """Stable log-sum-exp over the last axis: max + log(sum(exp(x - max)))."""
    x = _v(p["inputs"][0][0])
    o, out_shape, dtype = p["outputs"][0]
    el = _DTYPES[dtype]
    ishape = p["inputs"][0][1]
    ty = _tystr(ishape, el)
    rty = _tystr(ishape[:-1], el)
    ax = len(ishape) - 1
    kept = range(ax)
    n = {s: _v(o + s) for s in ("mi", "m", "mb", "s", "e", "zi", "ss", "lg", "a")}
    return [
        f"{n['mi']} = stablehlo.constant dense<{_INF[el][1]}> : tensor<{el}>",
        _reduce(n["m"], "maximum", x, n["mi"], [ax], ty, el, rty),
        _bcast(n["mb"], n["m"], kept, rty, ty),
        f"{n['s']} = stablehlo.subtract {x}, {n['mb']} : {ty}",
        f"{n['e']} = stablehlo.exponential {n['s']} : {ty}",
        f"{n['zi']} = stablehlo.constant dense<0.0> : tensor<{el}>",
        _reduce(n["ss"], "add", n["e"], n["zi"], [ax], ty, el, rty),
        f"{n['lg']} = stablehlo.log {n['ss']} : {rty}",
        f"{n['a']} = stablehlo.add {n['m']}, {n['lg']} : {rty}",
        f"{_v(o)} = stablehlo.reshape {n['a']} : ({rty}) -> {_tystr(out_shape, el)}",
    ]


def _logaddexp(p):
    """Stable log-add-exp: max(a, b) + log(exp(a - max) + exp(b - max))."""
    a = _v(p["inputs"][0][0])
    b = _v(p["inputs"][1][0])
    o, shape, dtype = p["outputs"][0]
    ty = _tystr(shape, _DTYPES[dtype])
    n = {s: _v(o + s) for s in ("m", "da", "db", "ea", "eb", "sm", "lg")}
    return [
        f"{n['m']} = stablehlo.maximum {a}, {b} : {ty}",
        f"{n['da']} = stablehlo.subtract {a}, {n['m']} : {ty}",
        f"{n['db']} = stablehlo.subtract {b}, {n['m']} : {ty}",
        f"{n['ea']} = stablehlo.exponential {n['da']} : {ty}",
        f"{n['eb']} = stablehlo.exponential {n['db']} : {ty}",
        f"{n['sm']} = stablehlo.add {n['ea']}, {n['eb']} : {ty}",
        f"{n['lg']} = stablehlo.log {n['sm']} : {ty}",
        f"{_v(o)} = stablehlo.add {n['m']}, {n['lg']} : {ty}",
    ]


def _dense(value, el):
    if isinstance(value, list):
        return "[" + ", ".join(_dense(v, el) for v in value) + "]"
    if el == "i1":
        return "true" if value else "false"
    if el in _FLOATS:
        return repr(float(value))
    return str(int(value))


def _constant(name, arr):
    if arr.size > _MAX_CONST_SIZE:
        raise NotImplementedError(
            f"constant {name} has {arr.size} elements; pass it as a function argument"
        )
    el = _DTYPES[arr.dtype]
    return (
        f"{_v(name)} = stablehlo.constant dense<{_dense(arr.tolist(), el)}> : "
        f"{_type(arr.shape, arr.dtype)}"
    )


def _compare_type(el):
    if el in _FLOATS:
        return "FLOAT"
    if el.startswith("ui"):
        return "UNSIGNED"
    return "SIGNED"


def _int_bounds(el):
    if el.startswith("ui"):
        return 0, 2 ** int(el[2:]) - 1
    w = int(el[1:])
    return -(2 ** (w - 1)), 2 ** (w - 1) - 1


def _reduce_init(kind, el):
    if kind in ("and", "or"):
        return "true" if kind == "and" else "false"
    if el in _FLOATS:
        m = {"sum": "0.0", "prod": "1.0", "min": _INF[el][0], "max": _INF[el][1]}
        return m[kind]
    lo, hi = _int_bounds(el)
    return {"sum": "0", "prod": "1", "min": str(hi), "max": str(lo)}[kind]


def _mean_last(prefix, src, shape, el):
    rty = _tystr(shape[:-1], el)
    ty = _tystr(shape, el)
    ax = len(shape) - 1
    zi, ss, inv, mn = (_v(prefix + s) for s in ("_zi", "_ss", "_inv", "_mn"))
    lines = [
        f"{zi} = stablehlo.constant dense<0.0> : tensor<{el}>",
        _reduce(ss, "add", src, zi, [ax], ty, el, rty),
        f"{inv} = stablehlo.constant dense<{1.0 / shape[-1]!r}> : {rty}",
        f"{mn} = stablehlo.multiply {ss}, {inv} : {rty}",
    ]
    return lines, mn


def _rmsnorm(p):
    """RMSNorm: x * rsqrt(mean(x^2, -1) + eps) * weight."""
    x, w = _v(p["inputs"][0][0]), _v(p["inputs"][1][0])
    eps = p["arguments"][0]
    o, shape, dtype = p["outputs"][0]
    el = _DTYPES[dtype]
    ty = _tystr(shape, el)
    rty = _tystr(shape[:-1], el)
    wty = _tystr(shape[-1:], el)
    kept, last = range(len(shape) - 1), [len(shape) - 1]
    n = {s: _v(o + s) for s in ("_sq", "_e", "_me", "_rs", "_rb", "_xn", "_wb")}
    lines = [f"{n['_sq']} = stablehlo.multiply {x}, {x} : {ty}"]
    ml, ms = _mean_last(o + "_ms", n["_sq"], shape, el)
    lines += ml + [
        f"{n['_e']} = stablehlo.constant dense<{eps!r}> : {rty}",
        f"{n['_me']} = stablehlo.add {ms}, {n['_e']} : {rty}",
        f"{n['_rs']} = stablehlo.rsqrt {n['_me']} : {rty}",
        _bcast(n["_rb"], n["_rs"], kept, rty, ty),
        f"{n['_xn']} = stablehlo.multiply {x}, {n['_rb']} : {ty}",
        _bcast(n["_wb"], w, last, wty, ty),
        f"{_v(o)} = stablehlo.multiply {n['_xn']}, {n['_wb']} : {ty}",
    ]
    return lines


def _layernorm(p):
    """LayerNorm: (x - mean) * rsqrt(var + eps) * weight + bias."""
    x, w, b = (_v(p["inputs"][i][0]) for i in range(3))
    eps = p["arguments"][0]
    o, shape, dtype = p["outputs"][0]
    el, ty = _DTYPES[dtype], _tystr(shape, _DTYPES[dtype])
    rty, wty = _tystr(shape[:-1], el), _tystr(shape[-1:], el)
    kept, last = range(len(shape) - 1), [len(shape) - 1]
    nm = lambda s: _v(o + s)
    ml, mu = _mean_last(o + "_mu", x, shape, el)
    lines = ml + [
        _bcast(nm("_mb"), mu, kept, rty, ty),
        f"{nm('_xc')} = stablehlo.subtract {x}, {nm('_mb')} : {ty}",
        f"{nm('_sq')} = stablehlo.multiply {nm('_xc')}, {nm('_xc')} : {ty}",
    ]
    vl, var = _mean_last(o + "_var", nm("_sq"), shape, el)
    lines += vl + [
        f"{nm('_e')} = stablehlo.constant dense<{eps!r}> : {rty}",
        f"{nm('_ve')} = stablehlo.add {var}, {nm('_e')} : {rty}",
        f"{nm('_rs')} = stablehlo.rsqrt {nm('_ve')} : {rty}",
        _bcast(nm("_rb"), nm("_rs"), kept, rty, ty),
        f"{nm('_xn')} = stablehlo.multiply {nm('_xc')}, {nm('_rb')} : {ty}",
        _bcast(nm("_wb"), w, last, wty, ty),
        f"{nm('_y')} = stablehlo.multiply {nm('_xn')}, {nm('_wb')} : {ty}",
        _bcast(nm("_bb"), b, last, wty, ty),
        f"{_v(o)} = stablehlo.add {nm('_y')}, {nm('_bb')} : {ty}",
    ]
    return lines


_COMPOSITE = {
    "RMSNorm": ("mlx.rms_norm", _rmsnorm),
    "LayerNorm": ("mlx.layer_norm", _layernorm),
}

_THREEFRY_ROT = [[13, 15, 26, 6], [17, 29, 16, 24]]


def _randombits(p):
    """Faithful Threefry-2x32: out[j]=hash(key,(j,half+j)).0, out[half+j]=.1."""
    shape, width = p["arguments"]
    key = p["inputs"][0]
    o, out_shape, dtype = p["outputs"][0]
    el = _DTYPES[dtype]
    n = 1
    for d in out_shape:
        n *= d
    if width != 4 or el != "ui32" or tuple(key[1]) != (2,) or n % 2 != 0:
        raise NotImplementedError("only single-key even-size uint32 RandomBits")
    kn = _v(key[0])
    half = n // 2
    u1 = "tensor<ui32>"
    uh = f"tensor<{half}xui32>"

    lines = []
    add = lines.append
    # scalar keys k0, k1 and k2 = k0 ^ k1 ^ 0x1BD11BDA, broadcast to (half,)
    ks = []
    for i in range(2):
        s1, sc = _v(o + f"_k{i}s"), _v(o + f"_k{i}")
        add(
            f"{s1} = stablehlo.slice {kn} [{i}:{i + 1}:1] : "
            f"(tensor<2xui32>) -> tensor<1xui32>"
        )
        add(f"{sc} = stablehlo.reshape {s1} : (tensor<1xui32>) -> {u1}")
        ks.append(sc)
    magic, xk, k2 = _v(o + "_magic"), _v(o + "_xk"), _v(o + "_k2")
    add(f"{magic} = stablehlo.constant dense<466688986> : {u1}")
    add(f"{xk} = stablehlo.xor {ks[0]}, {ks[1]} : {u1}")
    add(f"{k2} = stablehlo.xor {xk}, {magic} : {u1}")
    ks.append(k2)
    ksb = []
    for i in range(3):
        b = _v(o + f"_kb{i}")
        add(_bcast(b, ks[i], [], u1, uh))
        ksb.append(b)
    io, hc, c1 = _v(o + "_io"), _v(o + "_hc"), _v(o + "_c1")
    add(f"{io} = stablehlo.iota dim = 0 : {uh}")
    add(f"{hc} = stablehlo.constant dense<{half}> : {uh}")
    add(f"{c1} = stablehlo.add {io}, {hc} : {uh}")
    cf, cs = _v(o + "_cf0"), _v(o + "_cs0")
    add(f"{cf} = stablehlo.add {io}, {ksb[0]} : {uh}")
    add(f"{cs} = stablehlo.add {c1}, {ksb[1]} : {uh}")
    t = 0
    for rnd in range(5):
        for r in _THREEFRY_ROT[rnd % 2]:
            nf = _v(o + f"_f{t}")
            add(f"{nf} = stablehlo.add {cf}, {cs} : {uh}")
            rl, rr = _v(o + f"_rl{t}"), _v(o + f"_rr{t}")
            add(f"{rl} = stablehlo.constant dense<{r}> : {uh}")
            add(f"{rr} = stablehlo.constant dense<{32 - r}> : {uh}")
            names = ("shl", "shr", "rot", "xr")
            shl, shr, rot, xr = (_v(o + f"_{s}{t}") for s in names)
            add(f"{shl} = stablehlo.shift_left {cs}, {rl} : {uh}")
            add(f"{shr} = stablehlo.shift_right_logical {cs}, {rr} : {uh}")
            add(f"{rot} = stablehlo.or {shl}, {shr} : {uh}")
            add(f"{xr} = stablehlo.xor {rot}, {nf} : {uh}")
            cf, cs = nf, xr
            t += 1
        af, as_ = _v(o + f"_af{rnd}"), _v(o + f"_as{rnd}")
        add(f"{af} = stablehlo.add {cf}, {ksb[(rnd + 1) % 3]} : {uh}")
        ci = _v(o + f"_ci{rnd}")
        add(f"{ci} = stablehlo.constant dense<{rnd + 1}> : {uh}")
        tmp = _v(o + f"_ts{rnd}")
        add(f"{tmp} = stablehlo.add {cs}, {ksb[(rnd + 2) % 3]} : {uh}")
        add(f"{as_} = stablehlo.add {tmp}, {ci} : {uh}")
        cf, cs = af, as_
    add(
        f"{_v(o)} = stablehlo.concatenate {cf}, {cs}, dim = 0 : "
        f"({uh}, {uh}) -> {_tystr(out_shape, el)}"
    )
    return lines


def _dyn_starts(prefix, idx, idx_shape, idx_dtype, axes, rank):
    iel = _DTYPES[idx_dtype]
    idx_ty = _tystr(idx_shape, iel)
    pos = {a: j for j, a in enumerate(axes)}
    lines, starts, zero = [], [], None
    for d in range(rank):
        if d in pos:
            j = pos[d]
            s1, sc = _v(f"{prefix}_i{d}s"), _v(f"{prefix}_i{d}")
            lines.append(
                f"{s1} = stablehlo.slice {idx} [{j}:{j + 1}:1] : "
                f"({idx_ty}) -> tensor<1x{iel}>"
            )
            lines.append(
                f"{sc} = stablehlo.reshape {s1} : (tensor<1x{iel}>) -> tensor<{iel}>"
            )
            starts.append(sc)
        else:
            if zero is None:
                zero = _v(f"{prefix}_z")
                lines.append(f"{zero} = stablehlo.constant dense<0> : tensor<{iel}>")
            starts.append(zero)
    return lines, starts, iel


_SCATTER_COMBINE = {0: "maximum", 1: "minimum", 2: "add", 3: "multiply", 4: None}
_SCATTER_AXIS_COMBINE = {0: "add", 1: None}


def _combiner(prefix, el, combine):
    old, new = _v(prefix + "_old"), _v(prefix + "_new")
    lines = [f"^bb0({old}: tensor<{el}>, {new}: tensor<{el}>):"]
    if combine is None:
        lines.append(f"  stablehlo.return {new} : tensor<{el}>")
    else:
        c = _v(prefix + "_c")
        lines.append(f"  {c} = stablehlo.{combine} {old}, {new} : tensor<{el}>")
        lines.append(f"  stablehlo.return {c} : tensor<{el}>")
    return lines


def _stack_indices(prefix, idx_ops, idx_ins, g_shape, iel):
    gr = len(g_shape)
    n = len(idx_ops)
    one_ty = _tystr([*g_shape, 1], iel)
    if n == 1:
        si = _v(prefix + "_si")
        line = (
            f"{si} = stablehlo.reshape {idx_ops[0]} : "
            f"({_tystr(g_shape, iel)}) -> {one_ty}"
        )
        return [line], si, one_ty
    parts, lines = [], []
    for k in range(n):
        r = _v(prefix + f"_r{k}")
        lines.append(
            f"{r} = stablehlo.reshape {idx_ops[k]} : "
            f"({_tystr(idx_ins[k][1], iel)}) -> {one_ty}"
        )
        parts.append(r)
    si = _v(prefix + "_si")
    si_ty = _tystr([*g_shape, n], iel)
    lines.append(
        f"{si} = stablehlo.concatenate {', '.join(parts)}, dim = {gr} : "
        f"({', '.join([one_ty] * n)}) -> {si_ty}"
    )
    return lines, si, si_ty


def _primitive(p, precision):
    name = p["name"]
    ins = p["inputs"]
    args = p["arguments"]
    out_name, out_shape, out_dtype = p["outputs"][0]
    out = _v(out_name)
    ty = _type(out_shape, out_dtype)
    el = _DTYPES[out_dtype]
    ops = [_v(i[0]) for i in ins]
    in_ty = _type(ins[0][1], ins[0][2]) if ins else None

    if name in _BINARY:
        return [f"{out} = stablehlo.{_BINARY[name]} {ops[0]}, {ops[1]} : {ty}"]
    if name in _UNARY:
        return [f"{out} = stablehlo.{_UNARY[name]} {ops[0]} : {ty}"]
    if name == "Square":
        return [f"{out} = stablehlo.multiply {ops[0]}, {ops[0]} : {ty}"]
    if name == "Sqrt":
        return [f"{out} = stablehlo.{'rsqrt' if args[0] else 'sqrt'} {ops[0]} : {ty}"]
    if name == "Log":
        if args[0] == 2:
            return [f"{out} = stablehlo.log {ops[0]} : {ty}"]
        scale = _LOG2E if args[0] == 0 else _LOG10E
        n0 = _v(out_name + "_l")
        n1 = _v(out_name + "_c")
        return [
            f"{n0} = stablehlo.log {ops[0]} : {ty}",
            f"{n1} = stablehlo.constant dense<{scale!r}> : {ty}",
            f"{out} = stablehlo.multiply {n0}, {n1} : {ty}",
        ]
    if name in ("Sinh", "Cosh"):
        combine = "subtract" if name == "Sinh" else "add"
        n = {s: _v(out_name + s) for s in ("_n", "_e", "_en", "_d", "_h")}
        return [
            f"{n['_n']} = stablehlo.negate {ops[0]} : {ty}",
            f"{n['_e']} = stablehlo.exponential {ops[0]} : {ty}",
            f"{n['_en']} = stablehlo.exponential {n['_n']} : {ty}",
            f"{n['_d']} = stablehlo.{combine} {n['_e']}, {n['_en']} : {ty}",
            f"{n['_h']} = stablehlo.constant dense<0.5> : {ty}",
            f"{out} = stablehlo.multiply {n['_d']}, {n['_h']} : {ty}",
        ]
    if name == "Conjugate":
        rty = _tystr(out_shape, "f32")
        n = {s: _v(out_name + s) for s in ("_re", "_im", "_n")}
        return [
            f"{n['_re']} = stablehlo.real {ops[0]} : ({in_ty}) -> {rty}",
            f"{n['_im']} = stablehlo.imag {ops[0]} : ({in_ty}) -> {rty}",
            f"{n['_n']} = stablehlo.negate {n['_im']} : {rty}",
            f"{out} = stablehlo.complex {n['_re']}, {n['_n']} : {ty}",
        ]
    if name == "Softmax":
        return _softmax(p)
    if name == "LogSumExp":
        return _logsumexp(p)
    if name == "LogAddExp":
        return _logaddexp(p)
    if name == "RMSNorm":
        return _rmsnorm(p)
    if name == "LayerNorm":
        return _layernorm(p)
    if name == "RandomBits":
        return _randombits(p)
    if name == "Split":
        indices, axis = args
        ishape = ins[0][1]
        axis %= len(ishape)
        bounds = [0, *indices, ishape[axis]]
        lines = []
        for i, o in enumerate(p["outputs"]):
            spec = ", ".join(
                f"{bounds[i]}:{bounds[i + 1]}:1" if d == axis else f"0:{ishape[d]}:1"
                for d in range(len(ishape))
            )
            o_ty = _type(o[1], o[2])
            lines.append(
                f"{_v(o[0])} = stablehlo.slice {ops[0]} [{spec}] : ({in_ty}) -> {o_ty}"
            )
        return lines
    if name == "Pad":
        axes, low, high = args
        rank = len(ins[0][1])
        lo, hi = [0] * rank, [0] * rank
        for a, l, h in zip(axes, low, high):
            lo[a], hi[a] = l, h
        lo_s = ", ".join(map(str, lo))
        hi_s = ", ".join(map(str, hi))
        it_s = ", ".join(["0"] * rank)
        pv_ty = _type(ins[1][1], ins[1][2])
        return [
            f"{out} = stablehlo.pad {ops[0]}, {ops[1]}, low = [{lo_s}], "
            f"high = [{hi_s}], interior = [{it_s}] : ({in_ty}, {pv_ty}) -> {ty}"
        ]
    if name == "AddMM":
        alpha, beta = args
        r = len(ins[0][1])
        b = ", ".join(str(i) for i in range(r - 2))
        bd = f"batching_dims = [{b}] x [{b}], " if r > 2 else ""
        cd = f"contracting_dims = [{r - 1}] x [{r - 2}]"
        rhs_ty = _type(ins[1][1], ins[1][2])
        dot = _v(out_name + "_mm")
        pc = f"precision = [{precision}, {precision}]"
        lines = [
            f"{dot} = stablehlo.dot_general {ops[0]}, {ops[1]}, {bd}{cd}, {pc} : "
            f"({in_ty}, {rhs_ty}) -> {ty}"
        ]
        lhs, rhs = dot, ops[2]
        if alpha != 1.0:
            ac, ap = _v(out_name + "_ac"), _v(out_name + "_ap")
            lines.append(f"{ac} = stablehlo.constant dense<{alpha!r}> : {ty}")
            lines.append(f"{ap} = stablehlo.multiply {dot}, {ac} : {ty}")
            lhs = ap
        if beta != 1.0:
            bc, bp = _v(out_name + "_bc"), _v(out_name + "_bp")
            lines.append(f"{bc} = stablehlo.constant dense<{beta!r}> : {ty}")
            lines.append(f"{bp} = stablehlo.multiply {ops[2]}, {bc} : {ty}")
            rhs = bp
        lines.append(f"{out} = stablehlo.add {lhs}, {rhs} : {ty}")
        return lines
    if name == "Arange":
        start, stop, step = args
        lit = (lambda v: repr(float(v))) if el in _FLOATS else (lambda v: str(int(v)))
        if step == 1 and start == 0:
            return [f"{out} = stablehlo.iota dim = 0 : {ty}"]
        io = _v(out_name + "_io")
        lines = [f"{io} = stablehlo.iota dim = 0 : {ty}"]
        cur = io
        if step != 1:
            sc = _v(out_name + "_sc")
            dst = _v(out_name + "_sp") if start != 0 else out
            lines.append(f"{sc} = stablehlo.constant dense<{lit(step)}> : {ty}")
            lines.append(f"{dst} = stablehlo.multiply {cur}, {sc} : {ty}")
            cur = dst
        if start != 0:
            oc = _v(out_name + "_oc")
            lines.append(f"{oc} = stablehlo.constant dense<{lit(start)}> : {ty}")
            lines.append(f"{out} = stablehlo.add {cur}, {oc} : {ty}")
        return lines
    if name == "BitwiseBinary":
        return [f"{out} = stablehlo.{_BITWISE[args[0]]} {ops[0]}, {ops[1]} : {ty}"]
    if name in _COMPARE:
        ct = _compare_type(_DTYPES[ins[0][2]])
        return [
            f"{out} = stablehlo.compare {_COMPARE[name]}, {ops[0]}, {ops[1]}, {ct} : "
            f"({in_ty}, {in_ty}) -> {ty}"
        ]
    if name == "Select":
        return [
            f"{out} = stablehlo.select {ops[0]}, {ops[1]}, {ops[2]} : {in_ty}, {ty}"
        ]
    if name == "AsType":
        return [f"{out} = stablehlo.convert {ops[0]} : ({in_ty}) -> {ty}"]
    if name in _RESHAPE or name in _PASSTHROUGH:
        return [f"{out} = stablehlo.reshape {ops[0]} : ({in_ty}) -> {ty}"]
    if name == "Matmul":
        r = len(ins[0][1])
        b = ", ".join(str(i) for i in range(r - 2))
        bd = f"batching_dims = [{b}] x [{b}], " if r > 2 else ""
        cd = f"contracting_dims = [{r - 1}] x [{r - 2}]"
        rhs_ty = _type(ins[1][1], ins[1][2])
        pc = f"precision = [{precision}, {precision}]"
        return [
            f"{out} = stablehlo.dot_general {ops[0]}, {ops[1]}, {bd}{cd}, {pc} : "
            f"({in_ty}, {rhs_ty}) -> {ty}"
        ]
    if name == "Slice":
        spec = ", ".join(f"{s}:{e}:{t}" for s, e, t in zip(*args))
        return [f"{out} = stablehlo.slice {ops[0]} [{spec}] : ({in_ty}) -> {ty}"]
    if name == "Cholesky":
        lower = "false" if args[0] else "true"
        return [
            f"{out} = stablehlo.cholesky {ops[0]}, lower = {lower} : ({in_ty}) -> {ty}"
        ]
    if name == "Real":
        return [f"{out} = stablehlo.real {ops[0]} : ({in_ty}) -> {ty}"]
    if name == "Imag":
        return [f"{out} = stablehlo.imag {ops[0]} : ({in_ty}) -> {ty}"]
    if name == "Depends":
        lines = []
        for i, o in enumerate(p["outputs"]):
            o_ty = _type(o[1], o[2])
            lines.append(
                f"{_v(o[0])} = stablehlo.reshape {ops[i]} : ({o_ty}) -> {o_ty}"
            )
        return lines
    if name == "NumberOfElements":
        axes, inverted, _ = args
        n = 1
        for ax in axes:
            n *= ins[0][1][ax]
        value = (1.0 / n) if inverted else n
        lit = repr(float(value)) if el in _FLOATS else str(int(value))
        return [f"{out} = stablehlo.constant dense<{lit}> : {ty}"]
    if name == "Transpose":
        dims = ", ".join(str(d) for d in args[0])
        return [
            f"{out} = stablehlo.transpose {ops[0]}, dims = [{dims}] : ({in_ty}) -> {ty}"
        ]
    if name == "Concatenate":
        in_tys = ", ".join(_type(i[1], i[2]) for i in ins)
        return [
            f"{out} = stablehlo.concatenate {', '.join(ops)}, dim = {args[0]} : "
            f"({in_tys}) -> {ty}"
        ]
    if name == "Broadcast":
        off = len(out_shape) - len(ins[0][1])
        dims = ", ".join(str(off + i) for i in range(len(ins[0][1])))
        return [
            f"{out} = stablehlo.broadcast_in_dim {ops[0]}, dims = [{dims}] : "
            f"({in_ty}) -> {ty}"
        ]
    if name == "Reduce":
        op, kind = _REDUCE[args[0]]
        axes = args[1]
        red_shape = [d for i, d in enumerate(out_shape) if i not in axes]
        red_ty = _type(red_shape, out_dtype)
        init = _v(out_name + "_init")
        tmp = _v(out_name + "_r")
        dims = ", ".join(str(d) for d in axes)
        return [
            f"{init} = stablehlo.constant dense<{_reduce_init(kind, el)}> : "
            f"tensor<{el}>",
            f"{tmp} = stablehlo.reduce({ops[0]} init: {init}) applies stablehlo.{op} "
            f"across dimensions = [{dims}] : ({in_ty}, tensor<{el}>) -> {red_ty}",
            f"{out} = stablehlo.reshape {tmp} : ({red_ty}) -> {ty}",
        ]

    if name in _CHLO_UNARY:
        return [f"{out} = chlo.{_CHLO_UNARY[name]} {ops[0]} : {in_ty} -> {ty}"]
    if name == "ArcTan2":
        return [
            f"{out} = chlo.broadcast_atan2 {ops[0]}, {ops[1]} : "
            f"({in_ty}, {in_ty}) -> {ty}"
        ]
    if name == "DynamicSlice":
        axes, slice_size = args
        rank = len(ins[0][1])
        starts_lines, starts, iel = _dyn_starts(
            out_name, ops[1], ins[1][1], ins[1][2], axes, rank
        )
        st_tys = ", ".join([f"tensor<{iel}>"] * rank)
        sizes = ", ".join(map(str, slice_size))
        starts_lines.append(
            f"{out} = stablehlo.dynamic_slice {ops[0]}, {', '.join(starts)}, "
            f"sizes = [{sizes}] : ({in_ty}, {st_tys}) -> {ty}"
        )
        return starts_lines
    if name == "DynamicSliceUpdate":
        rank = len(ins[0][1])
        starts_lines, starts, iel = _dyn_starts(
            out_name, ops[2], ins[2][1], ins[2][2], args[0], rank
        )
        upd_ty = _type(ins[1][1], ins[1][2])
        st_tys = ", ".join([f"tensor<{iel}>"] * rank)
        starts_lines.append(
            f"{out} = stablehlo.dynamic_update_slice {ops[0]}, {ops[1]}, "
            f"{', '.join(starts)} : ({in_ty}, {upd_ty}, {st_tys}) -> {ty}"
        )
        return starts_lines
    if name == "SliceUpdate":
        reduce_type, start, end, strides = args
        if reduce_type != 4:
            raise NotImplementedError("reducing slice update is not supported")
        if any(s != 1 for s in strides):
            raise NotImplementedError("strided slice update is not supported")
        rank = len(start)
        lines = []
        starts = []
        for d in range(rank):
            c = _v(out_name + f"_s{d}")
            lines.append(f"{c} = stablehlo.constant dense<{start[d]}> : tensor<i32>")
            starts.append(c)
        st_tys = ", ".join(["tensor<i32>"] * rank)
        upd_ty = _type(ins[1][1], ins[1][2])
        lines.append(
            f"{out} = stablehlo.dynamic_update_slice {ops[0]}, {ops[1]}, "
            f"{', '.join(starts)} : ({in_ty}, {upd_ty}, {st_tys}) -> {ty}"
        )
        return lines

    if name == "Convolution":
        strides, pad_lo, pad_hi, kdil, idil, groups, flip = args
        nd = len(strides)
        sp = ", ".join(str(i) for i in range(nd))
        dn = f"[b, {sp}, f]x[o, {sp}, i]->[b, {sp}, f]"
        pad = ", ".join(f"[{lo}, {hi}]" for lo, hi in zip(pad_lo, pad_hi))
        rev = ", ".join(["true" if flip else "false"] * nd)
        w_ty = _type(ins[1][1], ins[1][2])
        pc = f"#stablehlo<precision {precision}>"
        return [
            f"{out} = stablehlo.convolution({ops[0]}, {ops[1]}) dim_numbers = {dn}, "
            f"window = {{stride = [{', '.join(map(str, strides))}], pad = [{pad}], "
            f"lhs_dilate = [{', '.join(map(str, idil))}], "
            f"rhs_dilate = [{', '.join(map(str, kdil))}], reverse = [{rev}]}} "
            f"{{batch_group_count = 1 : i64, feature_group_count = {groups} : i64, "
            f"precision_config = [{pc}, {pc}]}} : "
            f"({in_ty}, {w_ty}) -> {ty}"
        ]

    if name == "Sort":
        axis = args[0] % len(out_shape)
        ct = _compare_type(el)
        cmp = _v(out_name + "_cmp")
        a, b = _v(out_name + "_a"), _v(out_name + "_b")
        return [
            f'{out} = "stablehlo.sort"({ops[0]}) ({{',
            f"^bb0({a}: tensor<{el}>, {b}: tensor<{el}>):",
            f"  {cmp} = stablehlo.compare LT, {a}, {b}, {ct} : "
            f"(tensor<{el}>, tensor<{el}>) -> tensor<i1>",
            f"  stablehlo.return {cmp} : tensor<i1>",
            f"}}) {{dimension = {axis} : i64, is_stable = true}} : ({in_ty}) -> {ty}",
        ]
    if name == "ArgSort":
        axis = args[0] % len(out_shape)
        vel = _DTYPES[ins[0][2]]
        ct = _compare_type(vel)
        cmp, io, vout = _v(out_name + "_cmp"), _v(out_name + "_io"), _v(out_name + "_v")
        a, b = _v(out_name + "_a"), _v(out_name + "_b")
        c, d = _v(out_name + "_c"), _v(out_name + "_d")
        return [
            f"{io} = stablehlo.iota dim = {axis} : {ty}",
            f'{vout}, {out} = "stablehlo.sort"({ops[0]}, {io}) ({{',
            f"^bb0({a}: tensor<{vel}>, {b}: tensor<{vel}>, "
            f"{c}: tensor<{el}>, {d}: tensor<{el}>):",
            f"  {cmp} = stablehlo.compare LT, {a}, {b}, {ct} : "
            f"(tensor<{vel}>, tensor<{vel}>) -> tensor<i1>",
            f"  stablehlo.return {cmp} : tensor<i1>",
            f"}}) {{dimension = {axis} : i64, is_stable = true}} : "
            f"({in_ty}, {ty}) -> ({in_ty}, {ty})",
        ]

    if name == "Gather":
        axes, slice_sizes = args
        g_shape = ins[1][1]
        iel = _DTYPES[ins[1][2]]
        gr = len(g_shape)
        op_rank = len(ins[0][1])
        lines, si, si_ty = _stack_indices(out_name, ops[1:], ins[1:], g_shape, iel)
        offset = ", ".join(str(gr + d) for d in range(op_rank))
        dn = (
            f"#stablehlo.gather<offset_dims = [{offset}], collapsed_slice_dims = [], "
            f"start_index_map = [{', '.join(map(str, axes))}], index_vector_dim = {gr}>"
        )
        lines.append(
            f'{out} = "stablehlo.gather"({ops[0]}, {si}) {{dimension_numbers = {dn}, '
            f"slice_sizes = array<i64: {', '.join(map(str, slice_sizes))}>, "
            f"indices_are_sorted = false}} : ({in_ty}, {si_ty}) -> {ty}"
        )
        return lines
    if name == "GatherAxis":
        axis = args[0] % len(ins[0][1])
        r = len(ins[0][1])
        iel = _DTYPES[ins[1][2]]
        si = _v(out_name + "_si")
        si_ty = _tystr([*ins[1][1], 1], iel)
        batch = ", ".join(str(d) for d in range(r) if d != axis)
        dn = (
            f"#stablehlo.gather<offset_dims = [], collapsed_slice_dims = [{axis}], "
            f"operand_batching_dims = [{batch}], "
            f"start_indices_batching_dims = [{batch}], "
            f"start_index_map = [{axis}], index_vector_dim = {r}>"
        )
        return [
            f"{si} = stablehlo.reshape {ops[1]} : "
            f"({_type(ins[1][1], ins[1][2])}) -> {si_ty}",
            f'{out} = "stablehlo.gather"({ops[0]}, {si}) {{dimension_numbers = {dn}, '
            f"slice_sizes = array<i64: {', '.join(['1'] * r)}>, "
            f"indices_are_sorted = false}} : ({in_ty}, {si_ty}) -> {ty}",
        ]
    if name == "Scatter":
        reduce_type, axes = args
        g_shape = ins[1][1]
        iel = _DTYPES[ins[1][2]]
        gr = len(g_shape)
        op_rank = len(ins[0][1])
        n_idx = len(axes)
        upd = ops[-1]
        upd_ty = _type(ins[-1][1], ins[-1][2])
        lines, si, si_ty = _stack_indices(
            out_name, ops[1 : 1 + n_idx], ins[1 : 1 + n_idx], g_shape, iel
        )
        window = ", ".join(str(gr + d) for d in range(op_rank))
        dn = (
            f"#stablehlo.scatter<update_window_dims = [{window}], "
            f"inserted_window_dims = [], "
            f"scatter_dims_to_operand_dims = [{', '.join(map(str, axes))}], "
            f"index_vector_dim = {gr}>"
        )
        lines.append(f'{out} = "stablehlo.scatter"({ops[0]}, {si}, {upd}) ({{')
        lines.extend(_combiner(out_name, el, _SCATTER_COMBINE[reduce_type]))
        lines.append(
            f"}}) {{scatter_dimension_numbers = {dn}, indices_are_sorted = false, "
            f"unique_indices = false}} : ({in_ty}, {si_ty}, {upd_ty}) -> {ty}"
        )
        return lines
    if name == "ScatterAxis":
        reduce_type, axis = args
        axis %= len(ins[0][1])
        r = len(ins[0][1])
        iel = _DTYPES[ins[1][2]]
        si = _v(out_name + "_si")
        si_ty = _tystr([*ins[1][1], 1], iel)
        upd = ops[2]
        upd_ty = _type(ins[2][1], ins[2][2])
        batch = ", ".join(str(d) for d in range(r) if d != axis)
        dn = (
            f"#stablehlo.scatter<update_window_dims = [], "
            f"inserted_window_dims = [{axis}], "
            f"input_batching_dims = [{batch}], "
            f"scatter_indices_batching_dims = [{batch}], "
            f"scatter_dims_to_operand_dims = [{axis}], index_vector_dim = {r}>"
        )
        lines = [
            f"{si} = stablehlo.reshape {ops[1]} : "
            f"({_type(ins[1][1], ins[1][2])}) -> {si_ty}"
        ]
        lines.append(f'{out} = "stablehlo.scatter"({ops[0]}, {si}, {upd}) ({{')
        lines.extend(_combiner(out_name, el, _SCATTER_AXIS_COMBINE[reduce_type]))
        lines.append(
            f"}}) {{scatter_dimension_numbers = {dn}, indices_are_sorted = false, "
            f"unique_indices = false}} : ({in_ty}, {si_ty}, {upd_ty}) -> {ty}"
        )
        return lines

    raise NotImplementedError(f"unsupported primitive {name}")


def _func(name, ins, outs, lines):
    sig = ", ".join(f"{_v(n)}: {_type(s, d)}" for n, s, d in ins)
    rts = [_type(s, d) for _, s, d in outs]
    res = rts[0] if len(rts) == 1 else "(" + ", ".join(rts) + ")"
    ret = ", ".join(_v(n) for n, _, _ in outs)
    body = "\n".join(f"    {ln}" for ln in lines)
    tail = f"    return {ret} : {', '.join(rts)}"
    return f"  func.func private @{name}({sig}) -> {res} {{\n{body}\n{tail}\n  }}"


def _composite(p, cname, decomp, idx):
    # A named composite op plus the private decomposition function it references.
    out_name, out_shape, out_dtype = p["outputs"][0]
    fname = f"{cname.replace('.', '_')}_{idx}"
    in_tys = ", ".join(_type(s, d) for _, s, d in p["inputs"])
    ops = ", ".join(_v(i[0]) for i in p["inputs"])
    call = (
        f'{_v(out_name)} = stablehlo.composite "{cname}" {ops} '
        f"{{decomposition = @{fname}}} : ({in_tys}) -> {_type(out_shape, out_dtype)}"
    )
    fp = {
        "name": p["name"],
        "arguments": p["arguments"],
        "outputs": [("r", out_shape, out_dtype)],
        "inputs": [(f"a{i}", s, d) for i, (_, s, d) in enumerate(p["inputs"])],
    }
    return [call], _func(fname, fp["inputs"], fp["outputs"], decomp(fp))


def _build(events, composites=frozenset(), precision="highest"):
    inputs, outputs, constants, primitives = [], [], [], []
    for e in events:
        t = e["type"]
        if t == "inputs":
            inputs = e["inputs"]
        elif t == "outputs":
            outputs = e["outputs"]
        elif t == "constants":
            constants.extend(e["constants"])
        elif t == "primitive":
            primitives.append(e)

    args = ", ".join(f"{_v(n)}: {_type(s, d)}" for n, s, d in inputs)
    res_types = [_type(s, d) for _, s, d in outputs]
    res = res_types[0] if len(res_types) == 1 else "(" + ", ".join(res_types) + ")"

    cmap = _COMPOSITE
    prec = _PRECISION[precision]
    lines = [f"    {_constant(n, a)}" for n, a in constants]
    funcs = []
    for i, p in enumerate(primitives):
        if p["name"] in composites and p["name"] in cmap:
            cname, decomp = cmap[p["name"]]
            cl, fn = _composite(p, cname, decomp, i)
            lines.extend(f"    {ln}" for ln in cl)
            funcs.append(fn)
        else:
            lines.extend(f"    {ln}" for ln in _primitive(p, prec))
    out_vals = ", ".join(_v(n) for n, _, _ in outputs)
    lines.append(f"    return {out_vals} : {', '.join(res_types)}")

    main = f"  func.func @main({args}) -> {res} {{\n" + "\n".join(lines) + "\n  }"
    return "module @m {\n" + "\n".join([main, *funcs]) + "\n}\n"


def export_to_hlo(
    fn: Callable, *args, composites=frozenset(), precision="highest", **kwargs
) -> str:
    """Trace ``fn`` on the given inputs and return a StableHLO module as text.

    Args:
        fn (callable): The function to export.
        args: Example positional inputs used to trace ``fn``.
        composites (set): Primitive names to emit as ``stablehlo.composite`` ops
            (with a decomposition body) instead of inlining, e.g. ``{"RMSNorm"}``.
        precision (str): Matmul/convolution precision, one of ``"default"``,
            ``"high"``, ``"highest"``. ``"highest"`` matches MLX's fp32 accumulation
            on all backends; ``"default"`` lets the backend pick (e.g. bf16 on TPU).
        kwargs: Example keyword inputs used to trace ``fn``.

    Returns:
        str: The StableHLO module in textual MLIR assembly.
    """
    events = []
    mx.export_function(events.append, fn, *args, **kwargs)
    return _build(events, composites, precision)


def _flat(tree):
    items = tree_flatten(tree)
    return [v for _, v in items], [k for k, _ in items]


def flatten_args(*args) -> list:
    """Flatten pytree ``args`` into a flat list of arrays (executable inputs)."""
    leaves = []
    for a in args:
        leaves += _flat(a)[0]
    return leaves


def unflatten_out(leaves: list, out_keys: list) -> Any:
    """Rebuild the pytree output from the executable's flat result arrays."""
    return tree_unflatten(list(zip(out_keys, leaves)))


def export_tree(
    fn: Callable, *args, composites=frozenset(), precision="highest"
) -> tuple:
    """Export ``fn`` over pytree arguments (e.g. an ``nn.Module``'s parameters).

    Args:
        fn (callable): Function taking pytree args and returning any pytree of arrays.
        args: Example pytree inputs (nested dicts/lists of arrays, or bare arrays).
        composites (set): Primitive names to emit as ``stablehlo.composite`` ops.
        precision (str): Matmul/convolution precision; see :func:`export_to_hlo`.

    Returns:
        tuple: ``(hlo_text, out_keys)``. Flatten call inputs with :func:`flatten_args`
        in the same argument order, and rebuild outputs with :func:`unflatten_out`.
    """
    keys, counts, leaves = [], [], []
    for a in args:
        lv, ks = _flat(a)
        leaves += lv
        keys.append(ks)
        counts.append(len(lv))
    out_keys = []

    def flat_fn(*flat):
        rebuilt, i = [], 0
        for c, ks in zip(counts, keys):
            rebuilt.append(tree_unflatten(list(zip(ks, flat[i : i + c]))))
            i += c
        out_leaves, oks = _flat(fn(*rebuilt))
        out_keys.append(oks)
        return out_leaves

    return (
        export_to_hlo(flat_fn, *leaves, composites=composites, precision=precision),
        out_keys[0],
    )
