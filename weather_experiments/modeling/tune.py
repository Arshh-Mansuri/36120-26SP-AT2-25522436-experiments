"""Hyperparameter tuning with Optuna on rolling time-ordered folds (test set untouched until the end).

Folds (expanding window, inputs <= fold start - gap):
    train 1980-2013 -> score 2014-2016
    train 1980-2016 -> score 2017-2019
    train 1980-2019 -> score 2020-2022 (the validation period)

Run: uv run python -m weather_experiments.modeling.tune [--trials 30] [--target cci|whc|both]
"""

import json
import os

from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger
import numpy as np
import optuna
import pandas as pd
from shiftguard.weather import classification_scores, regression_scores
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import typer
from xgboost import XGBClassifier, XGBRegressor

from weather_experiments.config import REPORTS_DIR, SEED, TEST_START
from weather_experiments.features import CCI_TARGETS, WHC_TARGET, build_model_table, select_groups

app = typer.Typer()
FOLD_STARTS = ["2014-01-01", "2017-01-01", "2020-01-01"]
FOLD_LEN_YEARS = 3
LABELS = [0, 1, 2, 3]
# Threads per model; keep N_JOBS x parallel runs <= physical cores (oversubscription stalls LightGBM).
N_JOBS = int(os.getenv("N_JOBS", "4"))
optuna.logging.set_verbosity(optuna.logging.WARNING)


def log_trial(study, trial):
    logger.info(f"{study.study_name} trial {trial.number}: {trial.value:.4f} (best {study.best_value:.4f})")


def folds(data: pd.DataFrame, gap_days: int):
    for start in FOLD_STARTS:
        start = pd.Timestamp(start)
        end = start + pd.DateOffset(years=FOLD_LEN_YEARS)
        train = data.loc[data.index < start - pd.Timedelta(days=int(gap_days))]
        score = data.loc[(data.index >= start) & (data.index < end)]
        yield train, score


# ------------------------------------------------------------------ CCI (regression)

def xgb_reg(trial):
    return XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 60, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        objective=trial.suggest_categorical("objective", ["reg:squarederror", "reg:absoluteerror"]),
        random_state=SEED, n_jobs=N_JOBS,
    )


def lgbm_reg(trial):
    return LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        objective=trial.suggest_categorical("objective", ["regression", "regression_l1"]),
        random_state=SEED, verbose=-1, n_jobs=N_JOBS,
    )


def ridge_reg(trial):
    return make_pipeline(StandardScaler(), Ridge(alpha=trial.suggest_float("alpha", 0.1, 1000, log=True)))


REG_SPACES = {"XGBoost": xgb_reg, "LightGBM": lgbm_reg, "Ridge": ridge_reg}


def cv_mae(make, data, cols, target, gap):
    maes, preds = [], []
    for train, score in folds(data, gap):
        pred = np.clip(make().fit(train[cols], train[target]).predict(score[cols]), 0, 100)
        maes.append(regression_scores(score[target], pred)["mae"])
        preds.append(pd.Series(pred, index=score.index))
    return float(np.mean(maes)), pd.concat(preds)


def tune_cci(table, feature_sets, trials):
    rows, best = [], {}
    dev = table.loc[table.index < TEST_START]
    test = table.loc[table.index >= TEST_START]
    for h, target in zip((1, 2, 3), CCI_TARGETS):
        d, t = dev.dropna(subset=[target]), test.dropna(subset=[target])
        for fs_name, cols in feature_sets.items():
            oof, test_pred = {}, {}
            for model_name, space in REG_SPACES.items():
                n = trials if model_name != "Ridge" else 10
                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                            study_name=f"CCI t+{h} {model_name}")
                study.optimize(lambda tr, space=space, d=d, cols=cols, target=target, h=h:
                               cv_mae(lambda: space(tr), d, cols, target, h)[0], n_trials=n, callbacks=[log_trial])
                frozen = optuna.trial.FixedTrial(study.best_params)
                cv, oof[model_name] = cv_mae(lambda space=space, frozen=frozen: space(frozen), d, cols, target, h)
                model = space(frozen).fit(d[cols], d[target])
                test_pred[model_name] = np.clip(model.predict(t[cols]), 0, 100)
                rows.append({"horizon": f"t+{h}", "features": fs_name, "model": model_name, "cv_mae": cv,
                             "val_mae": regression_scores(d.loc[oof[model_name].index][target].loc["2020":], oof[model_name].loc["2020":])["mae"],
                             "test_mae": regression_scores(t[target], test_pred[model_name])["mae"]})
                best[(h, fs_name, model_name)] = study.best_params
                logger.info(rows[-1])
            for combo in (["XGBoost", "LightGBM"], ["XGBoost", "LightGBM", "Ridge"]):
                blend_oof = sum(oof[m] for m in combo) / len(combo)
                blend_test = sum(test_pred[m] for m in combo) / len(combo)
                y_oof = d.loc[blend_oof.index, target]
                rows.append({"horizon": f"t+{h}", "features": fs_name, "model": " + ".join(combo) + " (mean)",
                             "cv_mae": np.mean([regression_scores(y_oof.loc[s:e], blend_oof.loc[s:e])["mae"]
                                                for s, e in [("2014", "2016"), ("2017", "2019"), ("2020", "2022")]]),
                             "val_mae": regression_scores(y_oof.loc["2020":], blend_oof.loc["2020":])["mae"],
                             "test_mae": regression_scores(t[target], blend_test)["mae"]})
                logger.info(rows[-1])
    return pd.DataFrame(rows), best


# ------------------------------------------------------------------ WHC (classification)

def rf_clf(trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 200, 400, step=100),
        max_depth=trial.suggest_categorical("max_depth", [6, 8, 12, 16, 24]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 80, log=True),
        max_features=trial.suggest_categorical("max_features", ["sqrt", 0.1, 0.2]),
        class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        n_jobs=N_JOBS, random_state=SEED,
    )


def xgb_clf(trial):
    return XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 60, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        random_state=SEED, n_jobs=N_JOBS,
    )


def lgbm_clf(trial):
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 48, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
        random_state=SEED, verbose=-1, n_jobs=N_JOBS,
    )


CLF_SPACES = {"RandomForest": rf_clf, "XGBoost": xgb_clf, "LightGBM": lgbm_clf}


def fit_clf(model, X, y, weight_power):
    if isinstance(model, RandomForestClassifier):
        return model.fit(X, y)
    return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y) ** weight_power)


def full_proba(model, X):
    proba = np.zeros((len(X), len(LABELS)))
    proba[:, model.classes_.astype(int)] = model.predict_proba(X)
    return proba


def cv_proba(make, data, cols, weight_power):
    out = []
    for train, score in folds(data, 7):
        model = fit_clf(make(), train[cols], train[WHC_TARGET].astype(int), weight_power)
        out.append(pd.DataFrame(full_proba(model, score[cols]), index=score.index))
    return pd.concat(out)


def fold_f1(proba, y, class_scale=None):
    scale = np.ones(len(LABELS)) if class_scale is None else np.asarray(class_scale)
    pred = pd.Series((proba.to_numpy() * scale).argmax(axis=1), index=proba.index)
    return float(np.mean([classification_scores(y.loc[s:e], pred.loc[s:e], labels=LABELS)["macro_f1"]
                          for s, e in [("2014", "2016"), ("2017", "2019"), ("2020", "2022")]]))


def tune_scale(proba, y):
    """Class probability multipliers chosen on all three folds jointly (not on validation alone)."""
    best, best_f1 = [1, 1, 1, 1], fold_f1(proba, y)
    for s1 in (0.8, 1.0, 1.2, 1.4):
        for s2 in (1, 1.5, 2, 3, 4, 6):
            for s3 in (1, 2, 4):
                f1 = fold_f1(proba, y, [1, s1, s2, s3])
                if f1 > best_f1 + 1e-4:
                    best, best_f1 = [1, s1, s2, s3], f1
    return best, best_f1


def tune_whc(table, feature_sets, trials):
    rows, best = [], {}
    dev = table.loc[table.index < TEST_START].dropna(subset=[WHC_TARGET])
    test = table.loc[table.index >= TEST_START].dropna(subset=[WHC_TARGET])
    y_dev, y_test = dev[WHC_TARGET].astype(int), test[WHC_TARGET].astype(int)
    for fs_name, cols in feature_sets.items():
        oof, test_proba = {}, {}
        for model_name, space in CLF_SPACES.items():
            def objective(trial, space=space, model_name=model_name, cols=cols):
                power = 1.0 if model_name == "RandomForest" else trial.suggest_float("weight_power", 0.3, 1.0)
                return fold_f1(cv_proba(lambda: space(trial), dev, cols, power), y_dev.loc[FOLD_STARTS[0]:])
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                        study_name=f"WHC {model_name}")
            study.optimize(objective, n_trials=trials, callbacks=[log_trial])
            params = dict(study.best_params)
            power = params.pop("weight_power", 1.0)
            frozen = optuna.trial.FixedTrial(study.best_params)
            oof[model_name] = cv_proba(lambda space=space, frozen=frozen: space(frozen), dev, cols, power)
            model = fit_clf(space(frozen), dev[cols], y_dev, power)
            test_proba[model_name] = full_proba(model, test[cols])
            best[(fs_name, model_name)] = study.best_params
            rows += score_whc(fs_name, model_name, oof[model_name], test_proba[model_name], y_dev, y_test)
        blend_oof = sum(oof.values()) / len(oof)
        blend_test = sum(test_proba.values()) / len(test_proba)
        rows += score_whc(fs_name, "RF + XGB + LGBM (mean proba)", blend_oof, blend_test, y_dev, y_test)
    return pd.DataFrame(rows), best


def score_whc(fs_name, model_name, oof, test_proba, y_dev, y_test):
    y_oof = y_dev.loc[oof.index]
    rows = []
    scale, _ = tune_scale(oof, y_oof)
    for label, s in (("argmax", [1, 1, 1, 1]), (f"scaled {scale}", scale)):
        val_pred = (oof.loc["2020":].to_numpy() * s).argmax(axis=1)
        test_pred = (test_proba * s).argmax(axis=1)
        ts = classification_scores(y_test, test_pred, labels=LABELS)
        rows.append({"features": fs_name, "model": model_name, "decision": label,
                     "cv_macro_f1": fold_f1(oof, y_oof, s),
                     "val_macro_f1": classification_scores(y_oof.loc["2020":], val_pred, labels=LABELS)["macro_f1"],
                     "test_macro_f1": ts["macro_f1"], "test_bal_acc": ts["balanced_accuracy"],
                     "test_recall_high": ts["recall_2"]})
        logger.info(rows[-1])
    return rows


@app.command()
def main(trials: int = 30, target: str = "both"):
    table, features = build_model_table()
    # Feature sets chosen by the ablations in the notebooks (section D.3).
    cci_sets = {"raw+lags+calendar": select_groups(features, ["raw", "lags", "calendar"])}
    whc_sets = {"all": features}
    if target in ("cci", "both"):
        res, best = tune_cci(table, cci_sets, trials)
        res.to_csv(REPORTS_DIR / "cci_tuning.csv", index=False)
        (REPORTS_DIR / "cci_tuning_params.json").write_text(json.dumps({" | ".join(map(str, k)): v for k, v in best.items()}, indent=2))
        print(res.sort_values(["horizon", "cv_mae"]).to_string())
    if target in ("whc", "both"):
        res, best = tune_whc(table, whc_sets, trials)
        res.to_csv(REPORTS_DIR / "whc_tuning.csv", index=False)
        (REPORTS_DIR / "whc_tuning_params.json").write_text(json.dumps({" | ".join(k): v for k, v in best.items()}, indent=2))
        print(res.sort_values("cv_macro_f1", ascending=False).to_string())


if __name__ == "__main__":
    app()
