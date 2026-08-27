#!/usr/bin/env python3
"""Plot export-hlo bench results: grouped MLX-vs-JAX bars, one column per layout.

Reads the markdown tables written by bench_export_hlo_transformer.py (--out) and
renders build / compile / run as separate rows so their very different scales
never share an axis. Usage:

  python plot_bench.py bench_results.md -o bench.png
"""

import argparse
import re

MLX_COLOR = "#4C72B0"  # blue
JAX_COLOR = "#DD8452"  # orange
# (row label, mlx column, jax column, bar-label format)
METRICS = [
    ("Graph build\n(milliseconds)", "mlx_build", "jax_build", "%.0f"),
    ("XLA compilation\n(milliseconds)", "mlx_comp", "jax_comp", "%.0f"),
    ("Execution per step\n(milliseconds)", "mlx_run", "jax_run", "%.2f"),
]
LAYOUT_ORDER = ["single", "dp8", "mesh222"]
LAYOUT_LABELS = {
    "single": "Single device",
    "dp8": "Data parallel\n(8 devices)",
    "mesh222": "3D parallel: data + tensor + context\n(2×2×2 mesh, 8 devices)",
}
LEGEND = ["MLX (exported to StableHLO)", "JAX (hand-written)"]


def humanize_config(config):
    kv = dict(re.findall(r"(\w+)=(\S+)", config))
    parts = []
    if "platform" in kv:
        parts.append(kv["platform"].upper())
    if "devices" in kv:
        parts.append(f"{kv['devices']} devices")
    if "trials" in kv:
        parts.append(f"median of {kv['trials']} independent runs")
    if "batch" in kv:
        parts.append(f"batch size {kv['batch']}")
    if "seqlen" in kv:
        parts.append(f"sequence length {kv['seqlen']}")
    parts.append("shorter bars are better")
    return "   ·   ".join(parts)


def parse(path):
    """-> (config_line, {layout: {"sizes": [...], "cols": {col: [floats]}}})."""
    config = ""
    data = {}
    cur, header = None, None
    for raw in open(path):
        s = raw.strip()
        if s.startswith("platform="):
            config = s
        m = re.match(r"#{2,}\s*layout\s*=\s*(\S+)", s)
        if m:
            cur, header = m.group(1), None
            data[cur] = {"sizes": [], "cols": {}}
            continue
        if not (cur and s.startswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(cells[0]) <= set("-") or cells[0] == "size":
            continue
        row = dict(zip(header, cells))
        need = [c for _, mc, jc, _ in METRICS for c in (mc, jc)]
        try:
            vals = {c: float(row[c]) for c in need}
        except (KeyError, ValueError):
            continue  # FAIL / malformed row
        data[cur]["sizes"].append(cells[0])
        for c, val in vals.items():
            data[cur]["cols"].setdefault(c, []).append(val)
    return config, data


def plot(config, data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )

    layouts = [l for l in LAYOUT_ORDER if l in data] + [
        l for l in data if l not in LAYOUT_ORDER
    ]
    layouts = [l for l in layouts if data[l]["sizes"]]
    nrows, ncols = len(METRICS), len(layouts)

    all_sizes = [s for l in layouts for s in data[l]["sizes"]]
    is_dw = all(re.fullmatch(r"\d+x\d+", s) for s in all_sizes)
    xlabel = "Model size  (layers × hidden width)" if is_dw else "Model"

    def tick(s):
        return s.replace("x", " × ") if is_dw else s.upper()

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.9 * ncols, 3.0 * nrows),
        sharey="row",
        squeeze=False,
    )

    for j, layout in enumerate(layouts):
        d = data[layout]
        sizes = d["sizes"]
        x = np.arange(len(sizes))
        w = 0.38
        for i, (ylabel, mc, jc, fmt) in enumerate(METRICS):
            ax = axes[i][j]
            mlx = d["cols"].get(mc, [])
            jax_ = d["cols"].get(jc, [])
            b1 = ax.bar(x - w / 2, mlx, w, color=MLX_COLOR, label=LEGEND[0])
            b2 = ax.bar(x + w / 2, jax_, w, color=JAX_COLOR, label=LEGEND[1])
            ax.bar_label(b1, fmt=fmt, padding=2, fontsize=8)
            ax.bar_label(b2, fmt=fmt, padding=2, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([tick(s) for s in sizes])
            ax.grid(axis="y", ls=":", alpha=0.4)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(
                    LAYOUT_LABELS.get(layout, layout), fontweight="bold", pad=10
                )
            if i == nrows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
        # headroom for bar labels on the shared-y rows
    for i in range(nrows):
        top = max((p.get_height() for ax in axes[i] for p in ax.patches), default=1)
        for ax in axes[i]:
            ax.set_ylim(0, top * 1.18)

    handles = [b1, b2]
    fig.legend(
        handles,
        LEGEND,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=10,
    )
    fig.suptitle(
        "MLX-to-StableHLO export vs. hand-written JAX\n"
        "Llama-3 Adam training step on TPU",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    if config:
        fig.text(
            0.5,
            0.945,
            humanize_config(config),
            ha="center",
            va="top",
            fontsize=9.5,
            color="0.35",
        )
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(out, bbox_inches="tight")
    pdf = out.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {out} and {pdf}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="bench_results.md")
    ap.add_argument("-o", "--out", default="bench.png")
    args = ap.parse_args()
    config, data = parse(args.results)
    if not any(data[l]["sizes"] for l in data):
        raise SystemExit(f"no numeric rows parsed from {args.results}")
    plot(config, data, args.out)


if __name__ == "__main__":
    main()
