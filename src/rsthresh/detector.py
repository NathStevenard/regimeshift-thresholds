from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import logging
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, filtfilt
from src.rsthresh._version import __version__

DEFAULT_LEVELS = (0.95, 0.97, 0.99)

# ---- helpers -----------------------------------------------------------
def _butter_lowpass(y: np.ndarray, dt: float, cutoff_ka: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass. cutoff_ka is the desired cutoff period (ka)."""
    if cutoff_ka is None or cutoff_ka <= 0:
        return y
    fs = 1.0 / dt
    fc = 1.0 / cutoff_ka
    wn = min(max(fc / (0.5 * fs), 1e-6), 0.999)
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
    kde_grids: Dict[str, Dict[str, np.ndarray | tuple[float, ...]]]
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
                 spline_s: float | str = "auto",        # for "spline": "auto" (=N) or float
                 lp_cutoff_ka: float = 12.0,            # for "lowpass" (Butterworth): pass > 12 ka
                 ma_window_ka: float = 7.0,             # for "ma" (moving window average)
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
        self._raw_on_grid: Dict[str, np.ndarray] = {}
        self._raw_bounds: Dict[str, Tuple[float, float]] = {}

        self._persist_pts = 0
        self._separator_S: Optional[float] = None

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger or logging.getLogger("rsthresh")

        self.series: Optional[pd.DataFrame] = None
        self.results: Optional[ThresholdResult] = None
        self._spl: Dict[str, UnivariateSpline] = {}

        (self.output_dir / "Figures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "Reports").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "Results").mkdir(parents=True, exist_ok=True)

    # -- class helpers ----------------------------------------------------

    def _sample_around(self, key: str, t_evt: float) -> float:
        """Give the average values of a time-serie around a +/- 3 ka window."""
        raw_col = f"{key}_raw"
        if raw_col in self.series.columns:
            arr = self.series[raw_col].to_numpy()
        elif hasattr(self, "_raw_on_grid") and key in self._raw_on_grid:
            arr = self._raw_on_grid[key]
        else:
            # use the smoothed data is raw is not available
            arr = self.series[key].to_numpy()

        age = self.series["age"].to_numpy()
        dt = float(self.dt_ka)

        # half fixed window = 3 ka
        half = max(1, int(round(3.0 / dt)))

        # find the nearest index
        j = int(np.argmin(np.abs(age - t_evt)))
        i0 = max(0, j - half)
        i1 = min(len(age) - 1, j + half)

        window = arr[i0:i1 + 1]

        # average the 6 ka window
        val = float(np.nanmean(window))

        # if all data are NaN
        if not np.isfinite(val):
            if hasattr(self, "_spl") and key in self._spl:
                val = float(self._spl[key](t_evt))
            else:
                val = float(arr[j])

        return val

    def _value_at(self, key: str, t_evt: float) -> float:
        """Give a single value dépending on the smoothing method"""
        age = self.series["age"].to_numpy()
        j = int(np.argmin(np.abs(age - t_evt)))
        m = self.smooth_method.lower()

        if m == "spline":
            return float(self._spl[key](t_evt))
        elif m == "ma":
            return float(self.series[key].to_numpy()[j])
        elif m == "lowpass":
            return float(self.series[key].to_numpy()[j])
        elif m == "none":
            return float(self.series[f"{key}_raw"].to_numpy()[j])
        else:
            return float(self._spl[key](t_evt))

    def _derivative_at(self, key: str, t_evt: float) -> float:
        """Adjust the derivative function depending on the smoothing method"""
        m = self.smooth_method.lower()
        if m == "spline":
            return float(self._spl[key].derivative(1)(t_evt))
        age = self.series["age"].to_numpy()
        y = self.series[key].to_numpy()
        j = int(np.argmin(np.abs(age - t_evt)))
        jm1 = max(0, j - 1)
        jp1 = min(len(age) - 1, j + 1)
        return float((y[jp1] - y[jm1]) / (2.0 * self.dt_ka))

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
        """
        1) Places the series on a regular grid (dt = self.dt_ka).
           -> adds *_raw (interpolated, NOT smoothed) for local sampling.
        2) Smoothes each variable according to self.smooth_method ∈ {“spline”,“lowpass”,‘ma’,“none”}.
           - “spline”: UnivariateSpline on irregular data, s=self.smooth_s (float or dict per var)
           - “lowpass”: Butterworth lowpass (order=4, cutoff self.lp_cutoff_ka in ka)
           - “ma”: centered moving average (window self.ma_window_ka in ka)
           - “none”: no smoothing (copy of *_raw)
        3) Fills self.series with: age, target/x/y (smoothed) + target_raw/x_raw/y_raw.
           Stores self._spl (spline for derivatives if available), self._raw_on_grid, self._raw_bounds.
        """
        s_cfg = getattr(self, "smooth_spline_s", None)
        if s_cfg is None:
            s_cfg = getattr(self, "smooth_s", None)

        # ------------ cleaning & regular gridding ------------
        df = (self.series.loc[:, ["age", "target", "x", "y"]]
              .dropna(subset=["age"])
              .sort_values("age")
              .reset_index(drop=True))
        f = df[~df["age"].duplicated(keep="first")].reset_index(drop=True)

        age_irreg = df["age"].to_numpy(float)
        if age_irreg.size < 5:
            raise ValueError("Not enough data to smooth.")

        dt = float(self.dt_ka)
        gmin, gmax = float(np.nanmin(age_irreg)), float(np.nanmax(age_irreg))
        grid = np.arange(gmin, gmax + 0.5 * dt, dt, dtype=float)
        if grid[-1] < gmax:
            grid = np.append(grid, gmax)

        out = {"age": grid}
        raw_on_grid, raw_bounds = {}, {}
        self._spl = {}

        # -------- loop ----------
        for key in ["target", "x", "y"]:
            y_src = df[key].to_numpy(float)
            mask = np.isfinite(y_src)

            if self.impute_method == "linear":
                if mask.sum() >= 2:
                    y_raw = np.interp(grid, age_irreg[mask], y_src[mask])
                else:
                    y_raw = np.interp(grid, age_irreg, np.nan_to_num(y_src, nan=np.nanmedian(y_src)))
            elif self.impute_method == "ffill":
                y_f = pd.Series(y_src).ffill().bfill().to_numpy()
                y_raw = np.interp(grid, age_irreg, y_f)
            elif self.impute_method == "bfill":
                y_f = pd.Series(y_src).bfill().ffill().to_numpy()
                y_raw = np.interp(grid, age_irreg, y_f)
            elif self.impute_method in ("mean", "median"):
                fill = np.nanmean(y_src) if self.impute_method == "mean" else np.nanmedian(y_src)
                y_f = np.where(np.isfinite(y_src), y_src, fill)
                y_raw = np.interp(grid, age_irreg, y_f)
            else:  # défaut sûr
                if mask.sum() >= 2:
                    y_raw = np.interp(grid, age_irreg[mask], y_src[mask])
                else:
                    y_raw = np.interp(grid, age_irreg, np.nan_to_num(y_src, nan=np.nanmedian(y_src)))

            raw_on_grid[key] = y_raw

            # robust limits on irregular data
            qlo, qhi = np.nanquantile(y_src, 0.005), np.nanquantile(y_src, 0.995)
            raw_bounds[key] = (float(qlo), float(qhi))

            # ----- smoothing depending on the method -----
            m = self.smooth_method.lower()
            if m == "spline":
                s_val = (len(grid) if str(self.spline_s).lower() == "auto" else float(self.spline_s))
                spl = UnivariateSpline(grid, y_raw, s=s_val, k=3)
                y_smooth = spl(grid)
                self._spl[key] = spl
            elif m == "lowpass":
                y_smooth = _butter_lowpass(y_raw, dt=self.dt_ka, cutoff_ka=self.lp_cutoff_ka)
                self._spl[key] = UnivariateSpline(grid, y_smooth, s=0.0, k=3)
            elif m == "ma":
                y_smooth = _movavg(y_raw, dt=self.dt_ka, window_ka=self.ma_window_ka)
                self._spl[key] = UnivariateSpline(grid, y_smooth, s=0.0)
            elif m == "none":
                y_smooth = y_raw.copy()
                self._spl[key] = UnivariateSpline(grid, y_smooth, s=0.0)
            else:
                raise ValueError(f"unknown smooth_method: {self.smooth_method!r}")

            out[key] = y_smooth

        # add raw data in the output table
        out["target_raw"] = raw_on_grid["target"]
        out["x_raw"] = raw_on_grid["x"]
        out["y_raw"] = raw_on_grid["y"]

        self.series = pd.DataFrame(out)
        self._raw_on_grid = raw_on_grid
        self._raw_bounds = raw_bounds

        return self

    # -- separator & persistent state -----------------------------------------
    def estimate_separator(self, method: str = "gmm", **kwargs) -> "ThresholdDetector":
        """
        Estimate global separator S on smoothed target, then build a persistent binary state.
        Methods:
          gmm (default): 2-Gaussian mixture, S = density valley between component means.
          kde: 1D KDE; S = minimum between two highest modes; else median fallback.
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
                self.log.warning("KDE found < 2 modes; using median as separator.")

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
            j = i0 + int(np.argmax(np.abs(d_target[seg])))  # |d(target)/dt| peak

            pre = slice(max(0, j - 8), max(0, j - 1))
            post = slice(j + 1, min(len(df), j + 8))
            if (pre.stop - pre.start) < 2 or (post.stop - post.start) < 2:
                continue
            delta = float(np.nanmean(df["target"].to_numpy()[post]) -
                          np.nanmean(df["target"].to_numpy()[pre]))

            # Transition direction
            delta_flow = -delta
            direction = "up" if delta_flow > 0 else "down"

            # --- gate amplitude ---
            if abs(delta) < self.min_delta_sigma * float(np.nanmean(sigma_loc[seg])):
                self.log.debug(f"Rejected transition at age≈{df['age'].iloc[j]:.1f} ka (amplitude too small).")
                continue

            # ================== find the dx/dt peak ==================
            t_peak = float(df["age"].iloc[j])
            t_evt = t_peak

            # ================== sampling at t_evt (x,y, derivative) ==================
            xv = self._sample_around("x", t_evt)
            yv = self._sample_around("y", t_evt)
            tgt  = self._value_at("target", t_evt)
            dx_dt = sgn * self._derivative_at("x", t_evt)
            dy_dt = sgn * self._derivative_at("y", t_evt)

            rows.append({"t": t_evt, "dir": direction, "target": tgt,
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
        path = Path(path) if path else (self.output_dir / "Reports" / "threshold_report.json")
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
