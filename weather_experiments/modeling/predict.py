"""Offline inference with the saved production artefacts for any input date in the processed dataset.

Example: uv run python -m weather_experiments.modeling.predict 2025-06-01
"""

from datetime import timedelta

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from weather_experiments.config import MODELS_DIR
from weather_experiments.features import build_model_table

app = typer.Typer()

CCI_ARTEFACT = MODELS_DIR / "comfort_climate" / "comfort_climate_ensemble.joblib"
WHC_ARTEFACT = MODELS_DIR / "weather_hazard" / "weather_hazard_xgb.joblib"


@app.command()
def main(input_date: str):
    day = pd.Timestamp(input_date)
    table, _ = build_model_table()
    if day not in table.index:
        raise typer.BadParameter(f"{day.date()} is not in the processed dataset (needs 30 days of history).")

    cci = joblib.load(CCI_ARTEFACT)
    X = table.loc[[day], cci["features"]]
    low, high = cci["output_range"]
    for h in cci["horizons"]:
        pred = np.clip(np.mean([m.predict(X)[0] for m in cci["models"][f"t+{h}"]]), low, high)
        actual = table.loc[day, f"target_cci_t{h}"]
        logger.info(f"CCI {(day + timedelta(days=h)).date()}: predicted {pred:.1f} | actual {actual:.1f}")

    whc = joblib.load(WHC_ARTEFACT)
    label = int(whc["model"].predict(table.loc[[day], whc["features"]])[0])
    actual = table.loc[day, "target_whc_t7"]
    actual_label = whc["labels"][int(actual)] if pd.notna(actual) else "unknown"
    logger.info(f"WHC {(day + timedelta(days=whc['horizon'])).date()}: predicted {whc['labels'][label]} "
                f"| actual {actual_label}")


if __name__ == "__main__":
    app()
