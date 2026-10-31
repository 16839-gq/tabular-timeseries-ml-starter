"""Generate deterministic, entirely synthetic tabular and time-series datasets."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def make_tabular(rng: np.random.Generator) -> None:
    n_makes, per_make = 36, 18
    n = n_makes * per_make
    make_idx = np.repeat(np.arange(n_makes), per_make)
    regions = np.array(["US", "EU", "Caribbean"])[make_idx % 3]
    category = np.where(make_idx % 4 == 0, "Catamaran", "Monohull")
    year = rng.integers(2000, 2027, size=n)
    length = rng.normal(12.0 + 3.5 * (category == "Catamaran"), 2.3, n).clip(6, 25)
    beam = (0.30 * length + 3.0 * (category == "Catamaran") + rng.normal(0, 0.3, n)).clip(1.9)
    power = 2.6 * length + rng.normal(0, 5, n)
    make_effects = rng.normal(0, 0.09, size=n_makes)
    log_price = (10.9 + 0.060 * (year - 2000) + 0.095 * length +
                 0.12 * (category == "Catamaran") + make_effects[make_idx] +
                 rng.normal(0, 0.20, n))
    price = np.exp(log_price)
    frame = pd.DataFrame({
        "record_id": np.arange(n),
        "make": [f"Make_{i:02d}" for i in make_idx],
        "region": regions,
        "category": category,
        "year": year,
        "length": length,
        "beam": beam,
        "engine_kw": power,
        "price": price,
    })
    frame.loc[rng.choice(n, 40, replace=False), "engine_kw"] = -1  # explicit placeholder
    frame.loc[rng.choice(n, 18, replace=False), "beam"] = np.nan
    frame.to_csv(RAW / "tabular_demo.csv", index=False)
    unlabeled = frame.sample(10, random_state=43).drop(columns="price").copy()
    unlabeled["record_id"] += 10000
    unlabeled.to_csv(RAW / "tabular_unlabeled_demo.csv", index=False)


def make_timeseries(rng: np.random.Generator) -> None:
    n = 2000
    time = pd.date_range("2023-05-01", periods=n, freq="h", tz="UTC")
    hour = time.hour.to_numpy()
    solar = (4000 * np.maximum(0, np.sin(2 * np.pi * (hour - 6) / 24)) +
             rng.normal(0, 190, n)).clip(0)
    wind = (2700 + 950 * np.sin(np.arange(n) / 32) + rng.normal(0, 350, n)).clip(0)
    storage = (500 + 200 * np.cos(2 * np.pi * (hour - 19) / 24) +
               rng.normal(0, 90, n)).clip(0)
    thermal = (11000 - 0.50 * solar - 0.56 * wind +
               1100 * np.sin(2 * np.pi * (hour - 6) / 24) +
               rng.normal(0, 470, n)).clip(200)
    frame = pd.DataFrame({"time": time, "solar": solar, "wind": wind,
                          "storage": storage, "thermal_total": thermal})
    frame.loc[rng.choice(n, 65, replace=False), "solar"] = np.nan
    frame.loc[rng.choice(n, 55, replace=False), "wind"] = np.nan
    frame.loc[rng.choice(n, 48, replace=False), "storage"] = np.nan
    frame.loc[rng.choice(n, 13, replace=False), "thermal_total"] = np.nan
    frame.to_csv(RAW / "timeseries_demo.csv", index=False)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    make_tabular(np.random.default_rng(2026))
    make_timeseries(np.random.default_rng(2027))
    print("Wrote synthetic CSVs under data/raw/ (files are ignored by git)")
