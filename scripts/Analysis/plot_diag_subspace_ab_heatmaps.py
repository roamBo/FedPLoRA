#!/usr/bin/env python3
"""Plot same-run LoRA-A/LoRA-B pairwise subspace heatmaps from diag_subspace_AB.

This script consumes the ``--dump_matrices`` NPZ produced by
``scripts/Analysis/diag_subspace_AB.py``.  It is intentionally CPU-only and does
not recompute any LoRA states, so the plotted A and B panels share exactly the
same client order, training run, layers, and similarity definition.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _default_out(matrix_npz: Path, suffix: str) -> Path:
    return matrix_npz.with_name(matrix_npz.stem + suffix)


def _domain_ticks(domains: list[str]) -> tuple[list[float], list[str], list[int]]:
    centers: list[float] = []
    labels: list[str] = []
    boundaries: list[int] = []
    start = 0
    for idx in range(1, len(domains) + 1):
        if idx == len(domains) or domains[idx] != domains[start]:
            end = idx
            centers.append((start + end - 1) / 2.0)
            labels.append(domains[start][:4])
            if start > 0:
                boundaries.append(start)
            start = idx
    return centers, labels, boundaries


def _load_matrix(path: Path, value: str) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    with np.load(path) as data:
        domains = [str(x) for x in data["client_domains"].tolist()]
        if value == "similarity":
            a_mat = data["a_similarity"].astype(float)
            b_mat = data["b_similarity"].astype(float)
            cbar = "subspace similarity"
        else:
            a_mat = data["a_angle_deg"].astype(float)
            b_mat = data["b_angle_deg"].astype(float)
            cbar = "mean principal angle (deg.)"
    return a_mat, b_mat, domains, cbar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_npz", required=True, help="NPZ produced by diag_subspace_AB.py --dump_matrices.")
    ap.add_argument("--value", choices=["similarity", "angle"], default="similarity")
    ap.add_argument("--output_png", default="", help="Default: <matrix_npz_stem>_AB_heatmap.png")
    ap.add_argument("--output_pdf", default="", help="Optional PDF output path.")
    ap.add_argument("--title_suffix", default="", help="Optional suffix appended to panel titles.")
    args = ap.parse_args()

    matrix_npz = Path(args.matrix_npz)
    if not matrix_npz.is_file():
        raise SystemExit(f"[plot-ab][error] missing matrix npz: {matrix_npz}")

    a_mat, b_mat, domains, cbar = _load_matrix(matrix_npz, args.value)
    if a_mat.shape != b_mat.shape or a_mat.shape[0] != len(domains):
        raise SystemExit(
            f"[plot-ab][error] inconsistent shapes: A={a_mat.shape} B={b_mat.shape} domains={len(domains)}"
        )

    if args.value == "similarity":
        vmin, vmax, cmap = 0.0, 1.0, "Blues"
    else:
        finite = np.concatenate([a_mat[np.isfinite(a_mat)], b_mat[np.isfinite(b_mat)]])
        vmin, vmax, cmap = float(np.percentile(finite, 2)), float(np.percentile(finite, 98)), "magma_r"

    centers, labels, boundaries = _domain_ticks(domains)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True)
    titles = (r"LoRA-$A$ row-space", r"LoRA-$B$ column-space")
    mats = (a_mat, b_mat)
    im = None
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title}{args.title_suffix}", fontsize=9)
        ax.set_xticks(centers, labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(centers, labels, fontsize=7)
        for boundary in boundaries:
            ax.axhline(boundary - 0.5, color="white", linewidth=0.9)
            ax.axvline(boundary - 0.5, color="white", linewidth=0.9)
        ax.tick_params(length=0)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.036, pad=0.025, label=cbar)

    out_png = Path(args.output_png) if args.output_png else _default_out(matrix_npz, "_AB_heatmap.png")
    os.makedirs(out_png.parent, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[plot-ab][ok] png={out_png}")

    if args.output_pdf:
        out_pdf = Path(args.output_pdf)
        os.makedirs(out_pdf.parent, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"[plot-ab][ok] pdf={out_pdf}")


if __name__ == "__main__":
    main()
