"""Build the model-ready table: shiftguard features for date t plus the future targets."""

import pandas as pd
from shiftguard.weather import build_features

from weather_experiments.config import PROCESSED_DATA_DIR

CCI_TARGETS = ["target_cci_t1", "target_cci_t2", "target_cci_t3"]
WHC_TARGET = "target_whc_t7"

# Constant in Sydney (0 on every day 1980-2025), so it carries no information.
EXCLUDED_BASE_VARS = {"snowfall"}

FEATURE_GROUPS = {
    "raw": lambda c: "_lag" not in c and "_roll" not in c and "_diff" not in c
    and not c.startswith(("doy_", "month_")),
    "lags": lambda c: "_lag" in c or "_diff" in c,
    "rolling": lambda c: "_roll" in c,
    "calendar": lambda c: c.startswith(("doy_", "month_")),
}


def load_daily() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "sydney_daily.parquet")


def build_model_table(daily: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return features joined with targets (rows lacking full history dropped) and feature names."""
    daily = load_daily() if daily is None else daily
    features = build_features(daily)
    features = features[[c for c in features.columns if c not in EXCLUDED_BASE_VARS]]
    table = features.join(daily[CCI_TARGETS + [WHC_TARGET]]).dropna(subset=list(features.columns))
    return table, list(features.columns)


def select_groups(feature_names: list[str], groups: list[str]) -> list[str]:
    """Feature names belonging to the given groups (used for ablation)."""
    return [c for c in feature_names if any(FEATURE_GROUPS[g](c) for g in groups)]
