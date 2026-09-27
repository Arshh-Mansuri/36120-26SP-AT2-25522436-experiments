"""Download Sydney weather from Open-Meteo and build the daily modelling dataset.

Usage::

    uv run python -m weather_experiments.dataset            # 1980-2025
    uv run python -m weather_experiments.dataset --start-year 2000
"""

from pathlib import Path
import time

from loguru import logger
import pandas as pd
from shiftguard.weather import add_targets, aggregate_daily, fetch_hourly
from tqdm import tqdm
import typer

from weather_experiments.config import (
    END_YEAR,
    LATITUDE,
    LONGITUDE,
    PROCESSED_DATA_DIR,
    PRODUCTION_CUTOFF,
    RAW_DATA_DIR,
    START_YEAR,
    TIMEZONE,
)

app = typer.Typer()

REQUEST_PAUSE_SECONDS = 4


def download_hourly(start_year: int, end_year: int, raw_dir: Path) -> pd.DataFrame:
    """Download hourly data year by year, caching each year so reruns are incremental."""
    cache_dir = raw_dir / "hourly_by_year"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in tqdm(range(start_year, end_year + 1), desc="Open-Meteo"):
        path = cache_dir / f"sydney_hourly_{year}.parquet"
        if not path.exists():
            fetch_hourly(
                f"{year}-01-01", f"{year}-12-31",
                latitude=LATITUDE, longitude=LONGITUDE, timezone=TIMEZONE,
            ).to_parquet(path)
            # A year of 11 hourly variables costs ~30 API calls; stay under 600 calls/minute.
            time.sleep(REQUEST_PAUSE_SECONDS)
        frames.append(pd.read_parquet(path))
    hourly = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    return hourly[hourly["time"] < PRODUCTION_CUTOFF].reset_index(drop=True)


@app.command()
def main(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    raw_dir: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "sydney_daily.parquet",
):
    """Download hourly data, aggregate to daily observations and attach the targets."""
    if end_year >= int(PRODUCTION_CUTOFF[:4]):
        raise typer.BadParameter(f"end_year must be before {PRODUCTION_CUTOFF[:4]} (production data)")

    hourly = download_hourly(start_year, end_year, raw_dir)
    hourly_path = raw_dir / f"sydney_hourly_{start_year}_{end_year}.parquet"
    hourly.to_parquet(hourly_path)
    logger.info(f"{len(hourly):,} hourly rows -> {hourly_path}")

    daily = add_targets(aggregate_daily(hourly))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path)
    logger.success(
        f"{len(daily):,} daily rows ({daily.index.min().date()} to {daily.index.max().date()}), "
        f"{daily['cci'].isna().sum()} incomplete days -> {output_path}"
    )


if __name__ == "__main__":
    app()
