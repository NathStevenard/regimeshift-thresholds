from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple
import os

import numpy as np
import pandas as pd
import logging
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, filtfilt

from src.rsthresh._version import __version__

DEFAULT_LEVELS = (0.90, 0.95, 0.99)

# ---- helpers -----------------------------------------------------------
def _butter_lowpass(y: np.ndarray, dt: float, cutoff_ka: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass. cutoff_ka is the desired cutoff period (ka)."""
    if cutoff_ka is None or cutoff_ka <= 0:
        return y
    fs = 1.0/dt
    fc = 1.0/cutoff_ka
    wn = min(max(fc/(0.5*fs), 1e-6), 0.999)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, y, method="gust")


def _movavg(y: np.ndarray, dt: float, window_ka: float) -> np.ndarray:
    """Centered moving average (boxcar) with window in ka."""
    if window_ka is None or window_ka <= 0:
        return y
    w = max(3, int(round(window_ka / dt)))
    if w % 2 == 0:
        w += 1
    ker = np.ones(w, dtype=float) / w
    return np.convolve(y, ker, mode="same")


# ---- results container -------------------------------------------------------
@dataclass
class ThresholdResult:
    transitions: pd.DataFrame            # columns: t, dir, target, x, y, dx_dt, dy_dt
    kde_grids: Dict[str, Dict[str, np.ndarray]]
    separator_S: float                   # global separator (scalar)


# ---- main class --------------------------------------------------------------
class ThresholdDetector:
    """
    Minimal, generic pipeline:
      load_csv -> resample_and_smooth -> estimate_separator -> detect_transitions -> compute_kde -> save_report

    CSV expected: exactly 4 columns (positional mapping):
      col0=age ; col1=target ; col2=x ; col3=y
    All labels/axes in outputs reuse the original column names.
    """
    def __init__(self,
                 dt_ka: float = 1.0,                    # step for re-sampling
                 persist_ka: float = 4.0,               # window of persistance for transition
                 search_window_ka: float = 4.0,         # searching window for transition
                 sigma_window_ka: float | None = None,  # gate amplitude in ka
                 min_delta_sigma: float = 0.9,          # amplitude gate in units of local sigma
                 kde_bandwidth: str | float = "scott",  # short-scale isolation on target (optional)
                 smooth_method: str = "spline",         # "spline" | "lowpass" | "ma"
                 spline_s: float | str = "auto",  # for "spline": "auto" (=N) or float
                 lp_cutoff_ka: float = 20.0,            # for "lowpass" (Butterworth): pass > 20 ka
                 ma_window_ka: float = 10.0,            # for "ma" (moving window average)
                 impute_method: str = 'linear',         # "linear" | "ffill" | "bfill" | "mean" | "median"
                 output_dir: str | Path = "outputs",    # output path
                 logger: Optional[logging.Logger] = None):
        self.dt_ka = dt_ka
        self.persist_ka = persist_ka
        self.search_window_ka = search_window_ka
        self.sigma_window_ka = sigma_window_ka
        self.min_delta_sigma = min_delta_sigma
        self.kde_bandwidth = kde_bandwidth
        self.smooth_method = smooth_method
        self.lp_cutoff_ka = lp_cutoff_ka
        self.ma_window_ka = ma_window_ka
        self.spline_s = spline_s
        self.impute_method = impute_method

        self._persist_pts = 0
        self._separator_S: Optional[float] = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger or logging.getLogger("rsthresh")

        self.series: Optional[pd.DataFrame] = None     # standardized internal df: age,target,x,y
        self.results: Optional[ThresholdResult] = None
        self._spl: Dict[str, UnivariateSpline] = {}    # per-key splines

        os.makedirs(f"{output_dir}/Figures", exist_ok=True)
        os.makedirs(f"{output_dir}/Reports", exist_ok=True)
        os.makedirs(f"{output_dir}/Results", exist_ok=True)

    # -- column mapping & I/O --------------------------------------------------
    def _auto_map_columns(self, df: pd.DataFrame):
        names = list(df.columns)
        if len(names) != 4:
            raise ValueError(f"Expected exactly 4 columns (age + 3 series). Got {len(names)}: {names}")
        age, target, x, y = names[0], names[1], names[2], names[3]
        self.cols = {"age": age, "target": target, "x": x, "y": y}
        self.labels = {"age": age, "target": target, "x": x, "y": y}
        self.log.info(f"Column mapping: age='{age}', target='{target}', x='{x}', y='{y}'")

    def load_csv(self, path: str | Path) -> "ThresholdDetector":
        df_in = pd.read_csv(path)
        self._auto_map_columns(df_in)
        c = self.cols
        df = df_in[[c["age"], c["target"], c["x"], c["y"]]].copy()
        df = df.dropna(subset=[c["age"], c["target"]]).sort_values(c["age"]).reset_index(drop=True)
        self.series = df.rename(columns={c["age"]: "age", c["target"]: "target", c["x"]: "x", c["y"]: "y"})
        self.log.info(f"Loaded CSV '{path}' with {len(self.series)} rows. NaNs: {self.labels['x']}="
                      f"{self.series['x'].isna().sum()}, {self.labels['y']}={self.series['y'].isna().sum()}")
        return self

    # -- preprocessing (resample + optional short-scale isolation + smoothing) --
    def resample_and_smooth(self) -> "ThresholdDetector":
        assert self.series is not None, "Call load_csv() first."
        df = self.series
        age = df["age"].to_numpy()
        a0, a1 = float(age.min()), float(age.max())
        grid = np.arange(a0, a1 + self.dt_ka / 2, self.dt_ka)

        out = {"age": grid}
        for key in ["target", "x", "y"]:
            # Raw serie (if contains NaN)
            y_raw = df[key].to_numpy()

            if np.isnan(y_raw).any():
                s = pd.Series(y_raw, index=df["age"].to_numpy())
                m = self.impute_method.lower()
                if m == "linear":
                    s_imp = s.interpolate(method="values", limit_direction="both")
                elif m == "ffill":
                    s_imp = s.fillna(method="ffill").fillna(method="bfill")
                elif m == "bfill":
                    s_imp = s.fillna(method="bfill").fillna(method="ffill")
                elif m == "mean":
                    s_imp = s.fillna(s.mean())
                elif m == "median":
                    s_imp = s.fillna(s.median())
                else:
                    raise ValueError(f"Unknown impute_method='{self.impute_method}'")
                y_source = s_imp.to_numpy()
            else:
                y_source = y_raw

            # Resampling on regular grid
            y = np.interp(grid, age, y_source)

            # smoothing (depending on the method used)
            if self.smooth_method == "lowpass":
                y_s = _butter_lowpass(y, dt=self.dt_ka, cutoff_ka=self.lp_cutoff_ka)
                spl_der = UnivariateSpline(grid, y_s, s=0.0, k=3)       # to produce "clean" dx/dt

            elif self.smooth_method == "ma":
                y_s = _movavg(y, dt=self.dt_ka, window_ka=self.ma_window_ka)
                spl_der = UnivariateSpline(grid, y_s, s=0.0, k=3)       # to produce "clean" dx/dt

            elif self.smooth_method == "spline":
                s_val = (len(grid) if str(self.spline_s).lower() == "auto" else float(self.spline_s))
                spl = UnivariateSpline(grid, y, s=s_val, k=3)
                y_s = spl(grid)
                spl_der = spl

            else:
                raise ValueError(f"Unknown smooth_method='{self.smooth_method}'. Choose 'spline'|'lowpass'|'ma'.")

            out[key] = y_s
            self._spl[key] = spl_der

        if self.smooth_method == "spline":
            msg = f"Smoothing=Spline (s={'N' if str(self.spline_s).lower() == 'auto' else self.spline_s})"
        elif self.smooth_method == "lowpass":
            msg = f"Smoothing=Low-pass Butterworth (cutoff={self.lp_cutoff_ka} ka)"
        else:
            msg = f"Smoothing=Moving average (window={self.ma_window_ka} ka)"
        self.log.info(msg)

        self.series = pd.DataFrame(out)
        self.log.info(f"Smoothed on {len(grid)} points (dt={self.dt_ka} ka).")
        return self

    # -- separator & persistent state -----------------------------------------
    def estimate_separator(self, method: str = "gmm", **kwargs) -> "ThresholdDetector":
        """
        Estimate global separator S on smoothed target, then build a persistent binary state.
        Methods:
          gmm (default): 2-Gaussian mixture, S = density valley between component means.
          kde: 1D KDE; S = minimum between two highest modes; else median fallback.
          otsu: histogram Otsu threshold (max inter-class variance).
          quantile: S = quantile(y, q=0.5).
          fixed: S provided by user (kwarg S or self.fixed_separator).
        """
        assert self.series is not None, "Call resample_and_smooth() first."
        y = self.series["target"].to_numpy()

        def _apply_persistence(state_raw: np.ndarray) -> np.ndarray:
            win = max(3, int(round(self.persist_ka / self.dt_ka)))
            if win % 2 == 0:
                win += 1
            ser = pd.Series(state_raw)
            state = (ser.rolling(win, center=True, min_periods=1)
                       .apply(lambda v: 1 if v.mean() >= 0.5 else 0)
                       .astype(int).to_numpy())
            self._persist_pts = win
            return state

        if method == "gmm":
            random_state = kwargs.get("random_state", 42)
            fallback = kwargs.get("fallback", "kde")
            gm = GaussianMixture(n_components=2, random_state=random_state).fit(y.reshape(-1, 1))
            means = np.sort(gm.means_.ravel())
            if not np.isfinite(means).all() or (abs(means[1] - means[0]) < 1e-9):
                self.log.warning("GMM unimodal/degenerate; falling back to '%s'.", fallback)
                return self.estimate_separator(method=fallback, **kwargs)
            xg = np.linspace(np.percentile(y, 1), np.percentile(y, 99), 600).reshape(-1, 1)
            pdf = np.exp(gm.score_samples(xg))
            between = (xg[:, 0] >= means[0]) & (xg[:, 0] <= means[1])
            if between.sum() < 3:
                self.log.warning("No between-means region for GMM; falling back to '%s'.", fallback)
                return self.estimate_separator(method=fallback, **kwargs)
            idx_between = np.where(between)[0]
            i_min = idx_between[0] + int(np.argmin(pdf[idx_between]))
            S = float(xg[i_min, 0])

        elif method == "kde":
            bw = kwargs.get("bw_method", "scott")
            xg = np.linspace(np.percentile(y, 1), np.percentile(y, 99), 800)
            kde = gaussian_kde(y, bw_method=bw)
            dens = kde(xg)
            sd = np.diff(dens)
            peaks = np.where((np.hstack([sd, 0]) < 0) & (np.hstack([0, sd]) > 0))[0]
            if len(peaks) >= 2:
                top2 = peaks[np.argsort(dens[peaks])[-2:]]
                i0, i1 = np.sort(top2)
                S = float(xg[i0 + int(np.argmin(dens[i0:i1 + 1]))])
            else:
                S = float(np.median(y))
                self.log.warning("KDE found <2 modes; using median as separator.")

        elif method == "otsu":
            bins = int(kwargs.get("bins", 128))
            hist, edges = np.histogram(y, bins=bins, density=False)
            p = hist.astype(float) / max(1, hist.sum())
            m = 0.5 * (edges[:-1] + edges[1:])
            omega = np.cumsum(p)
            mu = np.cumsum(p * m)
            mu_t = mu[-1] if len(mu) else 0.0
            denom = (omega * (1 - omega))
            denom[denom == 0] = np.nan
            sigma_b2 = (mu_t * omega - mu) ** 2 / denom
            k = int(np.nanargmax(sigma_b2))
            S = float(0.5 * (edges[k] + edges[k + 1]))

        elif method == "quantile":
            q = float(kwargs.get("q", 0.5))
            q = min(max(q, 0.0), 1.0)
            S = float(np.quantile(y, q))

        elif method == "fixed":
            S = kwargs.get("S", getattr(self, "fixed_separator", None))
            if S is None:
                raise ValueError("method='fixed' requires kwarg S or self.fixed_separator.")
            S = float(S)

        else:
            raise ValueError(f"Unknown method='{method}'. Choose from: gmm, kde, otsu, quantile, fixed.")

        state_raw = (y >= S).astype(int)
        state = _apply_persistence(state_raw)
        self.series["state"] = state
        self._separator_S = S
        self.log.info(f"Separator estimated (method={method}): S={S:.6g}; persistence={self._persist_pts} pts")
        return self

    # -- transition detection --------------------------------------------------
    def detect_transitions(self) -> "ThresholdDetector":
        assert self.series is not None and "state" in self.series, \
            "Call estimate_separator() first to build a state."
        df = self.series
        age = df["age"].to_numpy()
        dt = self.dt_ka
        w = max(1, int(round(self.search_window_ka / dt)))
        sgn = -1

        # analytical derivative
        d_target = sgn * self._spl["target"].derivative(1)(age)

        # choose sigma window
        sigw = float(self.sigma_window_ka) if self.sigma_window_ka is not None else 40.0
        kstd = max(5, int(round(sigw / dt)))
        sigma_loc = pd.Series(df["target"]).rolling(kstd, center=True, min_periods=1).std().to_numpy()

        crossings = np.where(np.diff(df["state"].to_numpy()) != 0)[0]
        rows = []
        for idx in crossings:
            i0 = max(0, idx - w); i1 = min(len(df) - 1, idx + w)
            seg = slice(i0, i1 + 1)
            j = i0 + int(np.argmax(np.abs(d_target[seg])))

            # local pre/post plateaus
            pre = slice(max(0, j - 8), max(0, j - 1))
            post = slice(j + 1, min(len(df), j + 8))
            if (pre.stop - pre.start) < 2 or (post.stop - post.start) < 2:
                continue
            delta = float(np.nanmean(df["target"].to_numpy()[post]) -
                          np.nanmean(df["target"].to_numpy()[pre]))

            # direction in "flow" convention
            delta_flow = -delta
            direction = "up" if delta_flow > 0 else "down"

            # amplitude gate
            if abs(delta) < self.min_delta_sigma * float(np.nanmean(sigma_loc[seg])):
                self.log.debug(f"Rejected transition at age≈{df['age'].iloc[j]:.1f} ka (amplitude too small).")
                continue

            # read x/y & derivatives at t
            t = float(df["age"].iloc[j])
            xv = float(df["x"].iloc[j]); yv = float(df["y"].iloc[j])
            dx_dt = float(sgn * self._spl["x"].derivative(1)(age)[j])
            dy_dt = float(sgn * self._spl["y"].derivative(1)(age)[j])

            rows.append({"t": t, "dir": direction, "target": float(df["target"].iloc[j]),
                         "x": xv, "y": yv, "dx_dt": dx_dt, "dy_dt": dy_dt})

        trans = pd.DataFrame(rows)
        n_up = int((trans["dir"] == "up").sum()) if len(trans) else 0
        n_dn = int((trans["dir"] == "down").sum()) if len(trans) else 0
        self.log.info(f"Detected {len(trans)} transitions (up={n_up}, down={n_dn}).")
        self.results = ThresholdResult(transitions=trans, kde_grids={}, separator_S=self._separator_S)
        return self

    # -- KDE -------------------------------------------------------------------
    def compute_kde(self,
                    levels: Tuple[float, ...] = DEFAULT_LEVELS,
                    grid_steps: Tuple[int, int] = (100, 100)) -> "ThresholdDetector":
        assert self.results is not None, "Call detect_transitions() first."
        trans = self.results.transitions
        kde_grids = {}
        for direction in ("up", "down"):
            sub = trans[trans["dir"] == direction]
            if len(sub) < 6:
                self.log.warning(f"KDE skipped for '{direction}' (n={len(sub)} < 6).")
                continue
            X = np.vstack([sub["x"].to_numpy(), sub["y"].to_numpy()])
            kde = gaussian_kde(X, bw_method=self.kde_bandwidth)

            # grid
            xmin, xmax = float(np.nanmin(X[0])), float(np.nanmax(X[0]))
            ymin, ymax = float(np.nanmin(X[1])), float(np.nanmax(X[1]))
            Xg = np.linspace(xmin, xmax, grid_steps[0])
            Yg = np.linspace(ymin, ymax, grid_steps[1])
            XX, YY = np.meshgrid(Xg, Yg)
            ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

            # isodensity thresholds as quantiles of the density field
            qs = np.quantile(ZZ, levels)
            kde_grids[direction] = {"X": XX, "Y": YY, "Z": ZZ, "levels": qs, "levels_in": levels}
            self.log.info(f"KDE computed for '{direction}': grid={grid_steps}, bw={self.kde_bandwidth}, "
                          f"levels(density)={levels}.")
        self.results.kde_grids = kde_grids
        return self

    # -- report ----------------------------------------------------------------
    def save_report(self, path: str | Path | None = None) -> Path:
        assert self.results is not None, "Call detect_transitions() first."
        path = Path(path or (self.output_dir / "/Reports/threshold_report.json"))
        trans = self.results.transitions
        summary = {
            "version": __version__,
            "columns": self.cols,
            "labels": self.labels,
            "params": {
                "dt_ka": self.dt_ka,
                "persist_ka": self.persist_ka,
                "search_window_ka": self.search_window_ka,
                "sigma_window_ka": self.sigma_window_ka,
                "min_delta_sigma": self.min_delta_sigma,
                "smooth_method": self.smooth_method,
                "lp_cutoff_ka": self.lp_cutoff_ka,
                "ma_window_ka": self.ma_window_ka,
                "spline_s": self.spline_s,
                "impute_method": self.impute_method,
                "kde_bandwidth": self.kde_bandwidth,
            },
            "separator_S": self.results.separator_S,
            "counts": {
                "total": int(len(trans)),
                "up": int((trans["dir"] == "up").sum()),
                "down": int((trans["dir"] == "down").sum()),
            },
            "ranges": {
                "up": {
                    "x_min": float(trans.query("dir=='up'")["x"].min()) if len(trans) else None,
                    "x_max": float(trans.query("dir=='up'")["x"].max()) if len(trans) else None,
                    "y_min": float(trans.query("dir=='up'")["y"].min()) if len(trans) else None,
                    "y_max": float(trans.query("dir=='up'")["y"].max()) if len(trans) else None,
                },
                "down": {
                    "x_min": float(trans.query("dir=='down'")["x"].min()) if len(trans) else None,
                    "x_max": float(trans.query("dir=='down'")["x"].max()) if len(trans) else None,
                    "y_min": float(trans.query("dir=='down'")["y"].min()) if len(trans) else None,
                    "y_max": float(trans.query("dir=='down'")["y"].max()) if len(trans) else None,
                }
            }
        }
        path.write_text(json.dumps(summary, indent=2))
        self.log.info(f"Report saved: {path}")
        return path
