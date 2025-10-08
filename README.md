# regimeshift-thresholds (rsthresh)

![Licence MIT](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)

---

Simple, robust **regime-shift threshold detection** for paleoclimate time series (glacial–interglacial scale).
Given a 4-column CSV (age + 3 series), the pipeline smooths the records, estimates a **global separator** between weak/strong regimes, detects **transition timings**, and summarizes the likely **threshold windows** in forcing space via 2D KDE.

- Minimal inputs, **3 smoothing options** (spline / low-pass / moving average)
- Global separator via **GMM/KDE/Quantile/Fixed**
- Transition timing at **max |d(target)/dt|** near the separator crossing
- **KDE isodensity** contours for threshold windows (e.g., CO₂–RSL)
- Clean logs + JSON/CSV/SVG outputs

---

## Install

```bash
git clone https://github.com/NathStevenard/regimeshift-thresholds
cd regimeshift-thresholds

conda create -n rsthresh python=3.11 -y
conda activate rsthresh

pip install -e .

```

---

## Input format

CSV with exactly 4 columns (positional mapping):

- age (ka BP, increasing)
- target (series to segment into weak/strong regimes)
- x (first forcing/axis for KDE)
- y (second forcing/axis for KDE)

Column names can be anything (e.g., age, ISOWhybrid, CO2, RSL), the mapping is by position.
Age must be numeric and monotonic after sorting.

---

## Quickstart

An example is given in the examples/ repository. Find below a basic use of the algorithm.

```python
"""THIS IS AN EXAMPLE, USE THE "run_thresholds.py" FILE TO SETUP THE ALGORITHM"""

from pathlib import Path
from rsthresh import ThresholdDetector
from rsthresh.plotting import plot_summary
from rsthresh.logging_utils import setup_logger, log_separator
from rsthresh import __version__

# Define log configuration
log = setup_logger(log_dir="outputs", name="log_report", console_level=logging.INFO, file_level=logging.DEBUG)
log_separator(log, title="RUN START")
log.info(f"rsthresh version: {__version__}")

det = (
    ThresholdDetector(
        dt_ka=1.0,
        smooth_method="ma",
        spline_s="auto",
        lp_cutoff_ka=12.0,
        ma_window_ka=7.0,
        persist_ka=4.0,
        search_window_ka=4.0,
        sigma_window_ka=10,
        min_delta_sigma=0.7,
        impute_method="linear",
        logger=log
    )
    .load_csv(Path("examples/data.csv"))
    .resample_and_smooth()
    .estimate_separator(method="gmm")   # or "kde" | "quantile" | "fixed"
    .detect_transitions()
    .compute_kde()
)

plot_summary(det, figpath="outputs/thresholds_summary.svg", show_mode=True)
det.results.transitions.to_csv("outputs/transitions.csv", index=False)
det.save_report("outputs/threshold_report.json")
```

---

## Outputs

Three outputs are exported.

- **Transition table**
```markdown
outputs/
└── Results/
    └── transitions.csv        # timing and direction (up=weak-strong, down=strong-weak), plus x/y at each transition.
```
- **Report**
```markdown
outputs/
└── Reports/
    └── threshold_report_spline_<datetime>.json         # params, counts, ranges, and metadata.
```
- **Figure**
```markdown
outputs/
└── Figures/
    └── thresholds_summary_<datetime>.svg   # smoothed target + separator + timings; derivative + timings; x–y KDE isodensity.
```

---

## Smoothing options

Choose one method for all three series (target/x/y):

- **Spline**: smooth_method="spline", spline_s="auto" (default; very smooth, great for G–IG).
- **Low-pass Butterworth**: smooth_method="lowpass", lp_cutoff_ka=12.0 (keeps periods > 12 ka; zero-phase).
- **Moving Average**: smooth_method="ma", ma_window_ka=7.0 (GLT_lo style from Barker et al., 2011).

Derivatives are computed on the smoothed curve.

---

## Separator methods

Choose one method to define the separator between the two "modes" of your target record.

- **"gmm"** (default): 2-component Gaussian Mixture; separator = density valley between component means.
- **"kde"**: 1D KDE; separator = minimum between the two dominant modes (fallback to median).
- **"quantile"**: separator = quantile(y, q) (default q=0.5).
- **"fixed"**: user-provided S.

```python
det.estimate_separator(method="fixed", S=0.12)
det.estimate_separator(method="quantile", q=0.55)
```

---

## Key parameters (typical G–IG settings)

Additional parameters can be modified.

- dt_ka=1.0 — resampling step (ka).
- persist_ka=4.0 — state persistence window (ka) to suppress flicker.
- search_window_ka=4.0 — half-window around crossing to pick timing at max |d(target)/dt|.
- sigma_window_ka=40.0 — window (ka) to compute local σ for amplitude gating.
- min_delta_sigma=0.9 — non-dimensional amplitude gate; increase for stricter transitions.
- kde_bandwidth="scott" — KDE bandwidth rule (or a scalar factor).

If you get too few transitions -> lower min_delta_sigma or persist_ka, or reduce sigma_window_ka.
If you get too many -> increase them.

---

## Logging

All steps log useful info (loaded rows, smoothing choice, separator value, number of transitions, KDE status).
You can pass your own logger:
```python
det = ThresholdDetector(logger=my_logger, ...)
```

---

## Tests

A minimal pytest smoke test is included:
```python
from rsthresh import __version__
```
Please cite this code using the CITATION.cff in the repository.

If you use this code in scientific articles, please cite the original study:
***incoming***

---

## License
MIT © 2025 Nathan Stevenard