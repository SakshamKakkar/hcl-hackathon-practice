"""
model.py
Predict machine failure (1) vs normal (0) from sensor readings -> predictive maintenance.

Key decisions (be ready to defend each):
  * Baseline first: a "dummy" model that always predicts the majority class. Any real model must beat it.
  * Class imbalance (~7% failures): judge by recall / precision / F1, NOT accuracy.
  * Time-based split: train on the earlier 80% of readings, test on the latest 20%
    (a random split would let the model "see the future").
  * Pipeline: imputer + encoder are fitted on training data only -> no data leakage.
  * Threshold chosen by business cost, not the default 0.5.
"""

import joblib                                            # save / load the trained model to a file
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer            # apply different preprocessing to different columns
from sklearn.dummy import DummyClassifier                # the "always predict majority" baseline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer                 # fill missing values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline                    # chain preprocessing + model into one object
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.db import ROOT, run_query                       # reuse the database helper

MODEL_PATH = ROOT / "failure_model.joblib"               # where the trained pipeline is saved
NUMERIC = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min",
           "power_w", "temp_diff_k", "wear_x_torque"]    # numeric inputs (last three are engineered)
CATEGORICAL = ["machine_type"]                           # text category -> needs one-hot encoding
TARGET = "failure"


def load_data() -> pd.DataFrame:
    """Read readings from SQLite, sorted by time (needed for the time-based split)."""
    df = run_query("SELECT * FROM sensor_readings ORDER BY timestamp")
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer physics-based features. Domain knowledge often beats a fancier model."""
    df = df.copy()                                                      # never modify the caller's DataFrame
    df["power_w"] = df["torque_nm"] * df["rpm"] * 2 * np.pi / 60        # power = torque x angular velocity (rad/s)
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]         # small difference = poor cooling
    df["wear_x_torque"] = df["tool_wear_min"] * df["torque_nm"]         # worn tool under high load = overstrain
    return df                                                           # NaN torque -> NaN features; the imputer fixes them


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """First (1 - test_frac) of rows by time = train, the rest = test."""
    cut = int(len(df) * (1 - test_frac))                                # index where test data starts
    train, test = df.iloc[:cut], df.iloc[cut:]                          # iloc = slice by row position
    return train[NUMERIC + CATEGORICAL], train[TARGET], test[NUMERIC + CATEGORICAL], test[TARGET]


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model in one object, so .fit() learns imputation/scaling from training data only."""
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),     # median is robust to outliers
                          ("scale", StandardScaler())]), NUMERIC),          # scaling matters for logistic regression
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),       # L/M/H -> three 0/1 columns; unseen type -> all zeros
    ])
    return Pipeline([("pre", pre), ("model", model)])


CANDIDATES = {                                                              # models we compare, simplest first
    "baseline_majority": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),   # balanced = weight rare class up
    "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            min_samples_leaf=3, random_state=42, n_jobs=-1),  # min_samples_leaf limits overfitting
}


def compare_models(X_train, y_train) -> pd.DataFrame:
    """5-fold stratified cross-validation F1 on the training set for each candidate."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)     # stratified: each fold keeps the ~7% failure rate
    rows = []
    for name, model in CANDIDATES.items():
        scores = cross_val_score(build_pipeline(model), X_train, y_train, cv=cv, scoring="f1")
        rows.append({"model": name, "cv_f1_mean": round(scores.mean(), 3), "cv_f1_std": round(scores.std(), 3)})
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)


def best_threshold(y_true, proba, cost_missed: float, cost_false_alarm: float) -> float:
    """Pick the probability cut-off with the lowest total business cost.
    A missed failure (FN) means unplanned downtime; a false alarm (FP) means an unnecessary inspection."""
    thresholds = np.linspace(0.05, 0.95, 91)                            # try 0.05, 0.06, ..., 0.95
    costs = []
    for t in thresholds:
        pred = (proba >= t).astype(int)                                 # flag as failure if probability >= t
        fn = ((pred == 0) & (y_true == 1)).sum()                        # failures we missed
        fp = ((pred == 1) & (y_true == 0)).sum()                        # false alarms
        costs.append(fn * cost_missed + fp * cost_false_alarm)
    return float(thresholds[int(np.argmin(costs))])                     # threshold with the minimum cost


def evaluate(y_true, proba, threshold: float) -> dict:
    """All the metrics we report, at a given threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()   # labels=[0,1] keeps the order fixed
    return {
        "threshold": round(threshold, 2),
        "precision": round(precision_score(y_true, pred, zero_division=0), 3),  # of flagged, how many were real
        "recall": round(recall_score(y_true, pred, zero_division=0), 3),        # of real failures, how many caught
        "f1": round(f1_score(y_true, pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),                      # threshold-free ranking quality
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def train_and_save(cost_missed: float = 50000, cost_false_alarm: float = 2000) -> dict:
    """Full training run: compare -> fit best -> tune threshold -> test -> save. Returns a report dict."""
    df = load_data()
    X_tr, y_tr, X_te, y_te = time_split(df)
    comparison = compare_models(X_tr, y_tr)
    best_name = comparison.iloc[0]["model"]                             # highest cross-validated F1 wins
    # Tune the threshold on a validation slice (the latest 20% of the training period), never on the test set
    # and never on data the model was fitted on (that would make the threshold look better than it is).
    cut = int(len(X_tr) * 0.8)                                          # split training period: fit part | validation part
    tuner = build_pipeline(CANDIDATES[best_name]).fit(X_tr.iloc[:cut], y_tr.iloc[:cut])
    thr = best_threshold(y_tr.iloc[cut:].values, tuner.predict_proba(X_tr.iloc[cut:])[:, 1],
                         cost_missed, cost_false_alarm)
    pipe = build_pipeline(CANDIDATES[best_name]).fit(X_tr, y_tr)       # final model: refit on the whole training period
    test_metrics = evaluate(y_te.values, pipe.predict_proba(X_te)[:, 1], thr)
    joblib.dump({"pipeline": pipe, "threshold": thr, "model_name": best_name}, MODEL_PATH)
    importances = feature_importance(pipe)
    return {"comparison": comparison, "best_model": best_name, "test": test_metrics, "importance": importances}


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    """Which inputs matter most (explainability for the panel). Works for RF and logistic regression."""
    names = pipe.named_steps["pre"].get_feature_names_out()            # e.g. 'num__torque_nm', 'cat__machine_type_L'
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):                          # tree models
        values = model.feature_importances_
    elif hasattr(model, "coef_"):                                       # linear models: size of the coefficient
        values = np.abs(model.coef_[0])
    else:                                                               # dummy model has no importances
        return pd.DataFrame(columns=["feature", "importance"])
    return (pd.DataFrame({"feature": names, "importance": values})
              .sort_values("importance", ascending=False).reset_index(drop=True))


def predict_one(reading: dict) -> dict:
    """Score a single new reading (used by the app). reading = dict of raw sensor values + machine_type."""
    saved = joblib.load(MODEL_PATH)                                     # pipeline + threshold saved by train_and_save
    row = add_features(pd.DataFrame([reading]))                         # same feature engineering as training
    proba = float(saved["pipeline"].predict_proba(row[NUMERIC + CATEGORICAL])[0, 1])
    return {"failure_probability": round(proba, 3), "alert": bool(proba >= saved["threshold"]),
            "threshold": round(saved["threshold"], 2)}


if __name__ == "__main__":
    report = train_and_save()
    print(report["comparison"].to_string(index=False))
    print("Best:", report["best_model"])
    print("Test:", report["test"])
    print(report["importance"].head(5).to_string(index=False))
