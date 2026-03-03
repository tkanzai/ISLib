# ISLib — Instance Selection Library

ISLib is a Python library for identifying **optimal training periods** for regression models applied to time-series data. It combines unsupervised clustering (K-Means) and sliding-window regression to select the data instances that maximize model predictive performance.

## Key Features

- **Cluster Analysis** — identifies distinct operational regimes via K-Means and ranks them by how well each regime predicts the others, retaining only the most informative ones.
- **Sliding Window Analysis** — sweeps window sizes over the time series to find the training period length that minimises prediction error (MSE), with automatic change-point detection.
- **Full Analysis** — end-to-end pipeline that chains both methods: clusters first, then optimises the window on the filtered data.
- Generates embedded **Markdown reports** with figures for documentation and reproducibility.
- Works in both **supervised** (target column) and **unsupervised** (PCA reconstruction) modes.

---

## Installation

### From source

```bash
git clone https://github.com/your-org/islib.git
cd islib
pip install -e .
```

### Dependencies only

```bash
pip install -r requirements.txt
```

> **Python ≥ 3.8** is required.

---

## Quick Start

```python
import pandas as pd
from islib import InstanceSelectionLib

# Load a time-indexed DataFrame
df = pd.read_csv("Dataset.csv", sep=";", parse_dates=["data_datetime"])
df["data_datetime"] = df["data_datetime"].dt.tz_localize(None)
df = df.set_index("data_datetime").sort_index()

# Instantiate the library
islib = InstanceSelectionLib(
    max_clusters=25,
    min_window=30,
    resolution=100,
    tolerance=0.1,
    percentile_limit=70,
)

# --- Option 1: Full pipeline ---
(df_processed, sorted_clusters, test_errors,
 min_error_idx, df_results, df_optimized, report_md) = islib.full_analysis(df)

# --- Option 2: Clustering only ---
df_clustered, sorted_clusters, df_periods, cluster_md = islib.cluster_analysis(df)

# --- Option 3: Sliding window only ---
test_errors, min_error_idx, spikes, df_optimized, window_md = islib.window_analysis(df)
```

See [`Example.ipynb`](Example.ipynb) for a complete walkthrough.

---

## API Reference

### `InstanceSelectionLib(…)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_clusters` | `int` | `50` | Maximum number of K-Means clusters to evaluate. |
| `min_window` | `int` | `10` | Minimum training-window size (rows). |
| `resolution` | `int` | `100` | Number of window sizes to evaluate in the sliding window sweep. |
| `tolerance` | `float` | `0.1` | Relative tolerance above the minimum MSE to accept a smaller window. |
| `variance_threshold` | `float` | `0.95` | Explained-variance threshold for PCA (unsupervised mode). |
| `percentile_limit` | `float` | `25` | Percentile cut-off for cluster selection (lower = stricter). |
| `min_size_cluster` | `int` | `2` | Minimum cluster size; smaller clusters are treated as outliers. |
| `remove_outlier` | `bool` | `True` | Whether to discard outlier clusters before regression. |
| `spike_threshold` | `int` | `3` | Percentile threshold for change-point detection in the MSE profile. |
| `show_figures` | `bool` | `True` | Whether to display figures inline. |

### Methods

| Method | Description |
|--------|-------------|
| `full_analysis(df, target=None)` | Complete pipeline: clustering → window optimisation. |
| `cluster_analysis(df, target=None)` | Clustering step only. |
| `window_analysis(df, target=None)` | Sliding-window step only. |

---

## Input Data Format

ISLib expects a **pandas DataFrame** with:
- A `DatetimeIndex` (timezone-naive).
- Numeric feature columns (non-numeric values are coerced and forward/back-filled).
- An optional target column name passed via the `target` argument.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
