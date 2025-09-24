import numpy as np
import pandas as pd
from pathlib import Path
from src.rsthresh.detector import ThresholdDetector

def test_end_to_end(tmp_path: Path):
    # Create random .csv file: 4 columns (age, target, x, y)

    age = np.arange(0, 200+1, 1.0)  # 0..200 ka
    target = np.sin(2*np.pi*age/100) + 0.1*np.random.RandomState(42).randn(age.size)
    x = np.linspace(180, 300, age.size) # ex. CO2-like
    y = np.linspace(-120, 5, age.size)  # ex. RSL-like
    df = pd.DataFrame({"age": age, "target": target, "x": x, "y": y})
    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)

    # Run the detection
    det = (ThresholdDetector(dt_ka=1.0,
                             smooth_method="spline",
                             spline_s="auto",
                             persist_ka=4.0,
                             search_window_ka=4.0,
                             sigma_window_ka=40.0,
                             min_delta_sigma=0.9)
           .load_csv(csv)
           .resample_and_smooth()
           .estimate_separator(method="gmm")
           .detect_transitions()
           .compute_kde())

    # Checks
    assert det.series is not None
    assert det.results is not None
    assert len(det.results.transitions) >= 0