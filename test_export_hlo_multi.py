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

_PLATFORM = os.environ.get("EXPORT_HLO_PLATFORM") or xb.get_backend().platform
_devices = jax.devices(_PLATFORM)
N = len(_devices)
requires_multi = pytest.mark.skipif(N < 2, reason="need at least 2 devices")


def to_exported(text, in_avals, out_aval):
    backend = xb.get_backend(_PLATFORM)
    with jmlir.make_ir_context():
        module = ir.Module.parse(text)
        ver = hlo.get_version_from_compatibility_requirement(
            hlo.StablehloCompatibilityRequirement.WEEK_4
        )
        blob = _jax.mlir.serialize_portable_artifact(
            module, ver, backend.serialize_with_sdy
        )
    n = len(in_avals)
    return jexport.Exported(
        fun_name="main",
        in_tree=jax.tree_util.tree_structure((tuple(range(n)), {})),
        in_avals=tuple(in_avals),
        out_tree=jax.tree_util.tree_structure(0),
        out_avals=(out_aval,),
        _in_named_shardings=(None,) * n,
        _out_named_shardings=(None,),
        in_shardings_hlo=(None,) * n,
        out_shardings_hlo=(None,),
        nr_devices=1,
        platforms=(_PLATFORM,),
        ordered_effects=(),
        unordered_effects=(),
        disabled_safety_checks=(),
        mlir_module_serialized=blob,
        calling_convention_version=jexport.maximum_supported_calling_convention_version,
        module_kept_var_idx=tuple(range(n)),
        uses_global_constants=False,
        _get_vjp=None,
    )


def check_sharded(fn, arrays, in_specs, out_spec, expect_collective):
    ref = np.array(fn(*arrays))
    in_avals = [
        jax.core.ShapedArray(np.array(a).shape, np.array(a).dtype) for a in arrays
    ]
    out_aval = jax.core.ShapedArray(ref.shape, ref.dtype)
    exp = to_exported(export_to_hlo(fn, *arrays), in_avals, out_aval)

    mesh = Mesh(np.array(_devices), ("d",))
    in_sh = [NamedSharding(mesh, s) for s in in_specs]
    out_sh = NamedSharding(mesh, out_spec)
    f = jax.jit(exp.call, in_shardings=tuple(in_sh), out_shardings=out_sh)
    puts = [jax.device_put(np.array(a), s) for a, s in zip(arrays, in_sh)]

    collective = "all-reduce" in f.lower(*puts).compile().as_text()
    out = f(*puts)
    assert np.allclose(np.array(out), ref, atol=1e-4)
    assert collective == expect_collective


@requires_multi
@pytest.mark.parametrize(
    "fn,pick,in_specs,out_spec,expect",
    [
        pytest.param(
            lambda a, b: mx.log(mx.abs(a - b)) * b,
            lambda x, y: [x, y],
            [P("d"), P("d")],
            P("d"),
            False,
            id="pointwise",
        ),
        pytest.param(
            lambda a: mx.sum(a, axis=0),
            lambda x, y: [x],
            [P("d")],
            P(),
            True,
            id="reduce",
        ),
    ],
)
def test_sharded(fn, pick, in_specs, out_spec, expect):
    x = mx.array(np.random.rand(N, 4).astype(np.float32))
    y = mx.array(np.random.rand(N, 4).astype(np.float32) + 1)
    check_sharded(fn, pick(x, y), in_specs, out_spec, expect)
