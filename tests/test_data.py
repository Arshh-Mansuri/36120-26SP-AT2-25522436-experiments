"""Sanity checks on the processed dataset produced by ``weather_experiments.dataset``."""

import pandas as pd
import pytest

from weather_experiments.config import PROCESSED_DATA_DIR, PRODUCTION_CUTOFF

DAILY_PATH = PROCESSED_DATA_DIR / "sydney_daily.parquet"

pytestmark = pytest.mark.skipif(
    not DAILY_PATH.exists(), reason="run `python -m weather_experiments.dataset` first"
)


@pytest.fixture(scope="module")
def daily():
    return pd.read_parquet(DAILY_PATH)


def test_no_production_data(daily):
    assert daily.index.max() < pd.Timestamp(PRODUCTION_CUTOFF)


def test_index_is_continuous_daily(daily):
    assert daily.index.is_monotonic_increasing
    assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_targets_in_range(daily):
    assert daily["cci"].dropna().between(0, 100).all()
    assert daily["whi"].dropna().between(0, 100).all()
    assert set(daily["whc"].dropna().unique()) <= {0, 1, 2, 3}


def test_targets_are_future_values(daily):
    assert daily["target_cci_t1"].iloc[:-1].equals(daily["cci"].shift(-1).iloc[:-1])
    assert daily["target_whc_t7"].iloc[-7:].isna().all()


def test_physical_ranges(daily):
    assert daily["relative_humidity_2m"].dropna().between(0, 100).all()
    assert daily["cloud_cover"].dropna().between(0, 100).all()
    assert (daily["precipitation"].dropna() >= 0).all()
