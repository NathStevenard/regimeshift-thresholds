from pathlib import Path
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from src.rsthresh.detector import ThresholdDetector
from src.rsthresh.plotting import plot_summary
from src.rsthresh.logging_utils import setup_logger, log_separator
from src.rsthresh import __version__

# Define log configuration
log = setup_logger(log_dir="outputs", name="log_report", console_level=logging.INFO, file_level=logging.DEBUG)
log_separator(log, title="RUN START")
log.info(f"rsthresh version: {__version__}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This is an example of regime shift detection, build to detect 'glacial-interglacial' shifts over a long (800,000 years)
period.
For any question/information, contact me at: nathan.stevenard@univ-grenoble-alpes.fr
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~ CHANGE INPUT AND ALGORITHM PARAMETERS BELOW ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ------- Data path -------
DATA = Path(__file__).with_name("data.csv")

# ------- Smoothing method -------
"""
Three smoothing methods are proposed. 
All methods have respective parameters.
Fell free to adjust these parameters if needed, it will be directly adjusted in the pipeline.
"""
# CHOOSE YOUR METHOD HERE
smooth_method="spline"          # (str) || "spline" (default) | "lowpass" (low band pass) | "ma" (moving average)

# MODIFY SETTINGS DEPENDING ON THE METHOD
# 1. ~~ Spline settings ~~
spline_s="auto"                 # (float), (str) || "auto" -> length of the dataset

# 2. ~~ Low band pass settings ~~
lp_cutoff_ka=20.0               # (float) || change the period as desired

# 3. ~~ Moving average settings ~~
ma_window_ka=10.0               # (float) || change the window as desired

# ------- Additional settings -------

impute_method="linear"          # (str) || "linear" | "ffill" | "bfill" | "mean" | "median". To fill NaN values
separator="gmm"                 # (str) || "gmm" | "kde" | "otsu" | "quantile" | "fixed"
kde_bandwidth="scott"           # (str), (float) || "scott" | "silverman". Define the kde bandwidth

S=0.0                           # (float) || values for the "fixed" separator
dt_ka=1.0                       # (float) || step for resampling in ka
persist_ka=4.0                  # (float) || window in ka of the state persistance filter
search_window_ka=4.0            # (float) || half of the window around the S crossing to find the max dx/dt
sigma_window_ka=40.0            # (float) || gate amplitude in ka
min_delta_sigma=0.9             # (float) || without dimension. Change it from 0.7-1.2 depending on the gate amplitude

"""
Other additionnal settings are available, but not necessary. 
You can check the Class in detector.py for more information.
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~ DO NOT CHANGE THE BLOCK BELOW ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

det = (
    ThresholdDetector(
        dt_ka=dt_ka,
        smooth_method=smooth_method,
        spline_s=spline_s if smooth_method=="spline" else None,
        lp_cutoff_ka=lp_cutoff_ka if smooth_method=="lowpass" else None,
        ma_window_ka=ma_window_ka if smooth_method=="ma" else None,
        persist_ka=persist_ka,
        search_window_ka=search_window_ka,
        sigma_window_ka=sigma_window_ka,
        min_delta_sigma=min_delta_sigma,
        impute_method=impute_method,
        logger=log
    )
    .load_csv(DATA)
    .resample_and_smooth()
    .estimate_separator(method=separator, S=S)
    .detect_transitions()
    .compute_kde()
)
det.results.transitions.to_csv(f"outputs/Results/transitions_{smooth_method}_{stamp}.csv", index=False)

# Save the results as a figure of the threshold analysis

plot_summary(det, figpath=f"outputs/Figures/thresholds_summary_{smooth_method}_{stamp}.svg", show_mode=True)
plt.show()

# Export the log report as "outputs/threshold_report.json"
det.save_report(f"outputs/Reports/threshold_report_{smooth_method}_{stamp}.json")
log_separator(log, title="RUN END")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~ ADD CODE BELOW THIS LINE (IF DESIRED) ~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~