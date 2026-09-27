"""Compare baselines and candidate models for both targets.

Models are selected on the validation period (2020-2022); the test period (2023-2025) is scored
for reference. Results are written to ``reports/``.

Usage::

    uv run python -m weather_experiments.modeling.compare
"""

import json
import time

from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger
import numpy as np
import pandas as pd
from shiftguard import adversarial_auc
from shiftguard.weather import (
    ClimatologyBaseline,
    classification_scores,
    regression_scores,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import typer
from xgboost import XGBClassifier, XGBRegressor

from weather_experiments.config import REPORTS_DIR, SEED, TEST_START, VAL_START
from weather_experiments.features import (
    CCI_TARGETS,
    WHC_TARGET,
    build_model_table,
    select_groups,
)

app = typer.Typer()
LABELS = [0, 1, 2, 3]


def split(table: pd.DataFrame):
    train = table[table.index < VAL_START]
    val = table[(table.index >= VAL_START) & (table.index < TEST_START)]
    test = table[table.index >= TEST_START]
    return train, val, test


def regressors():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=SEED
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            random_state=SEED, verbose=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=5, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
        ),
    }


def classifiers():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, max_features="sqrt", class_weight="balanced",
            n_jobs=-1, random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            class_weight="balanced", random_state=SEED, verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
        ),
    }


def run_cci(train, val, test, features) -> pd.DataFrame:
    rows = []
    for target in CCI_TARGETS:
        h = int(target[-1])
        tr, va, te = (d.dropna(subset=[target]) for d in (train, val, test))
        for split_name, d in (("val", va), ("test", te)):
            rows.append({"target": target, "model": "Persistence", "split": split_name,
                         **regression_scores(d[target], d["cci"])})
        clim = ClimatologyBaseline(horizon=h).fit(tr.index, tr[target])
        for split_name, d in (("val", va), ("test", te)):
            rows.append({"target": target, "model": "Climatology", "split": split_name,
                         **regression_scores(d[target], clim.predict(d.index))})
        for name, model in regressors().items():
            start = time.time()
            model.fit(tr[features], tr[target])
            for split_name, d in (("val", va), ("test", te)):
                pred = np.clip(model.predict(d[features]), 0, 100)
                rows.append({"target": target, "model": name, "split": split_name,
                             **regression_scores(d[target], pred),
                             "fit_seconds": round(time.time() - start, 1)})
            logger.info(f"CCI {target} {name} done")
    return pd.DataFrame(rows)


def run_cci_ablation(train, val, features) -> pd.DataFrame:
    rows = []
    steps = [["raw"], ["raw", "lags"], ["raw", "lags", "rolling"],
             ["raw", "lags", "rolling", "calendar"]]
    for target in CCI_TARGETS:
        tr, va = train.dropna(subset=[target]), val.dropna(subset=[target])
        for groups in steps:
            cols = select_groups(features, groups)
            model = regressors()["LightGBM"].fit(tr[cols], tr[target])
            rows.append({"target": target, "features": "+".join(groups), "n_features": len(cols),
                         **regression_scores(va[target], model.predict(va[cols]))})
    return pd.DataFrame(rows)


def run_whc(train, val, test, features) -> pd.DataFrame:
    rows = []
    tr, va, te = (d.dropna(subset=[WHC_TARGET]) for d in (train, val, test))
    y_tr = tr[WHC_TARGET].astype(int)

    def record(name, split_name, y_true, y_pred):
        rows.append({"model": name, "split": split_name,
                     **classification_scores(y_true, y_pred, labels=LABELS)})

    majority = int(y_tr.mode().iloc[0])
    clim = ClimatologyBaseline(horizon=7, by="month", task="classification").fit(tr.index, y_tr)
    for split_name, d in (("val", va), ("test", te)):
        y = d[WHC_TARGET].astype(int)
        record("Majority class", split_name, y, np.full(len(d), majority))
        record("Climatology (month)", split_name, y, clim.predict(d.index).astype(int))
        record("Persistence", split_name, y, d["whi"].pipe(_whi_class))

    for name, model in classifiers().items():
        fit_kwargs = {}
        if name == "XGBoost":
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_tr)
        model.fit(tr[features], y_tr, **fit_kwargs)
        for split_name, d in (("val", va), ("test", te)):
            record(f"{name} (balanced)", split_name, d[WHC_TARGET].astype(int),
                   model.predict(d[features]))
        logger.info(f"WHC {name} done")

    # Softer re-weighting: balanced weights are ~330x for Extreme, which can overfit 11 examples.
    sqrt_w = np.sqrt(compute_sample_weight("balanced", y_tr))
    soft = LGBMClassifier(**{**classifiers()["LightGBM"].get_params(), "class_weight": None})
    soft.fit(tr[features], y_tr, sample_weight=sqrt_w)
    unweighted = LGBMClassifier(**{**classifiers()["LightGBM"].get_params(), "class_weight": None})
    unweighted.fit(tr[features], y_tr)
    for split_name, d in (("val", va), ("test", te)):
        y = d[WHC_TARGET].astype(int)
        record("LightGBM (sqrt-balanced)", split_name, y, soft.predict(d[features]))
        record("LightGBM (unweighted)", split_name, y, unweighted.predict(d[features]))
    return pd.DataFrame(rows)


def _whi_class(whi: pd.Series) -> np.ndarray:
    return np.digitize(whi.to_numpy(), [25, 50, 75])


@app.command()
def main():
    table, features = build_model_table()
    train, val, test = split(table)
    logger.info(f"{len(features)} features | train {len(train)} val {len(val)} test {len(test)}")
    REPORTS_DIR.mkdir(exist_ok=True)

    drift = {
        "adversarial_auc_train_vs_val": adversarial_auc(train[features], val[features]),
        "adversarial_auc_train_vs_test": adversarial_auc(train[features], test[features]),
    }
    logger.info(f"Drift: {drift}")
    (REPORTS_DIR / "drift.json").write_text(json.dumps(drift, indent=2))

    cci = run_cci(train, val, test, features)
    cci.to_csv(REPORTS_DIR / "cci_model_comparison.csv", index=False)
    ablation = run_cci_ablation(train, val, features)
    ablation.to_csv(REPORTS_DIR / "cci_feature_ablation.csv", index=False)
    whc = run_whc(train, val, test, features)
    whc.to_csv(REPORTS_DIR / "whc_model_comparison.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== CCI (MAE) ===")
    print(cci.pivot_table(index="model", columns=["target", "split"], values="mae").round(3))
    print("\n=== CCI feature ablation (validation MAE, LightGBM) ===")
    print(ablation.pivot_table(index="features", columns="target", values="mae").round(3))
    print("\n=== WHC ===")
    cols = ["macro_f1", "balanced_accuracy", "accuracy", "recall_0", "recall_1", "recall_2",
            "recall_3"]
    print(whc.set_index(["split", "model"])[cols].sort_index().round(3))


if __name__ == "__main__":
    app()
