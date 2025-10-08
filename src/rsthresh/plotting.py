from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .detector import ThresholdDetector, DEFAULT_LEVELS


def plot_summary(
    det: ThresholdDetector,
    figpath: str = "outputs/Figures/thresholds_summary.svg",
    levels: Tuple[float, ...] = DEFAULT_LEVELS,   # KDE isodensity levels (density quantiles)
    figsize: Tuple[float, float] = (10.0, 6.0),    # ~quarter page
    width_ratios: Tuple[float, float, float] = (1.0, 1.0, 1.2),
    point_size: int = 22,
    line_width: float = 1.2,
    fontsize: int = 9,
    show_mode: bool = False                       # mark KDE mode with a star
) -> plt.Figure:
    """
    Compact 3-panel figure:
      (a) target (smoothed) + global separator S + transition timings
      (b) d(target)/dt (spline derivative) + timings
      (c) x–y scatter of transitions + KDE isodensity contours

    Requires:
      - det.series exists (after resample_and_smooth + estimate_separator)
      - det.results exists (after detect_transitions)
    Will compute KDE if missing.
    """
    # sanity checks with actionable hints
    if det.series is None:
        raise RuntimeError("Missing data. Run: load_csv().resample_and_smooth().estimate_separator().detect_transitions()")
    if det.results is None or det.results.transitions is None:
        raise RuntimeError("Missing transitions. Run: estimate_separator().detect_transitions() before plotting.")

    # compute KDE if absent
    if not det.results.kde_grids:
        det.compute_kde(levels=levels)

    df = det.series.copy()
    trans = det.results.transitions.copy()
    S = det.results.separator_S
    age = df["age"].to_numpy()

    sgn = -1
    d_target = sgn * det._spl["target"].derivative(1)(age)

    colors = {"up": "tab:red", "down": "tab:blue"}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, width_ratios=width_ratios, height_ratios=[1, 1], hspace=0.25, wspace=0.25)

    # (a) target
    axA = fig.add_subplot(gs[0, 0:2])
    axA.plot(df["age"], df["target"], lw=1.5)
    if S is not None:
        axA.axhline(S, color="k", ls="--", lw=1.0, alpha=0.7, label="Separator S")
    for _, r in trans.iterrows():
        axA.axvline(r["t"], color=colors.get(r["dir"], "0.5"), lw=0.9, alpha=0.6)
    axA.set_ylabel(f"{det.labels.get('target','target')} (smoothed)")
    axA.set_title("(a) Regime detection", loc="left", fontsize=fontsize)
    axA.legend(loc="upper right", fontsize=fontsize-2, frameon=False)

    # (b) derivative
    axB = fig.add_subplot(gs[1, 0:2])
    axB.plot(df["age"], d_target, lw=line_width)
    for _, r in trans.iterrows():
        axB.axvline(r["t"], color=colors.get(r["dir"], "0.5"), lw=0.9, alpha=0.6)
    axB.axhline(0, color='k', ls="--", lw=0.5, alpha=0.7)
    axB.set_ylabel(f"d({det.labels.get('target','target')})/dt")
    axB.set_xlabel("Age (ka BP)")
    axB.set_title("(b) Derivative & timing", loc="left", fontsize=fontsize)

    # (c) x–y KDE
    axC = fig.add_subplot(gs[:, 2])
    for direction in ("up", "down"):
        sub = trans[trans["dir"] == direction]
        if len(sub) == 0:
            continue
        axC.scatter(sub["x"], sub["y"], s=point_size, alpha=0.95,
                    label=f"{direction} (n={len(sub)})",
                    edgecolors="white", linewidths=0.5, c=colors[direction])

        dgrid = det.results.kde_grids.get(direction, None)
        if dgrid is not None:
            XX, YY, ZZ = dgrid["X"], dgrid["Y"], dgrid["Z"]
            lvl = dgrid.get("levels", None)
            if lvl is not None:
                axC.contour(XX, YY, ZZ, levels=lvl, colors=colors[direction], linewidths=2)
                if show_mode:
                    jmax = np.argmax(ZZ)
                    jx, jy = np.unravel_index(jmax, ZZ.shape)
                    axC.plot(XX[jx, jy], YY[jx, jy], marker="*", ms=7,
                             color=colors[direction], mec="k", mew=0.5)

    axC.set_xlabel(det.labels.get("x", "x"))
    axC.set_ylabel(det.labels.get("y", "y"))
    axC.yaxis.set_label_position("right")
    axC.yaxis.tick_right()
    axC.set_title("(c) Threshold windows (KDE isodensity)", loc="left", fontsize=fontsize)
    axC.legend(fontsize=fontsize-2, frameon=False)

    for ax in (axA, axB, axC):
        ax.tick_params(labelsize=fontsize)

    fig.savefig(figpath, dpi=300, bbox_inches="tight")
    # log
    n_up = int((trans["dir"] == "up").sum())
    n_dn = int((trans["dir"] == "down").sum())
    det.log.info(f"Summary figure saved: {figpath} "
                 f"(transitions: total={len(trans)}, up={n_up}, down={n_dn}; KDE levels={levels})")
    return fig
