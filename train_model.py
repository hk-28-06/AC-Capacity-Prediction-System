"""
train_model.py
---------------
Trains all three models on the 50,000-row dataset and persists everything
the Streamlit app needs.

Models:
  1. Linear Regression   -> main / interpretable predictor used by the app
  2. Random Forest        -> mid-complexity benchmark
  3. Extra Trees          -> high-accuracy benchmark (depth-capped so the
                             saved file stays well under GitHub's 100MB limit
                             with only a negligible accuracy cost)

Run:  python train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "Room_Area",
    "Room_Height",
    "Occupancy",
    "Outdoor_Temperature",
    "Window_Area",
    "Equipment_Load",
    "Insulation_Level",
    "Sun_Exposure",
]
TARGET = "AC_Capacity_Ton"
DATA_PATH = "data/AC_Capacity_Dataset_50000.csv"

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def evaluate(model, y_test, y_pred, cv_X=None, cv_y=None):
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }
    if cv_X is not None:
        metrics["CV_R2_mean"] = float(
            cross_val_score(model, cv_X, cv_y, cv=5, scoring="r2").mean()
        )
    return metrics


# ---- Model 1: Linear Regression (primary / interpretable model) --------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
lin_metrics = evaluate(lin_model, y_test, y_pred_lin, X, y)

# ---- Model 2: Random Forest (mid-complexity benchmark) -----------------
rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_metrics = evaluate(rf_model, y_test, y_pred_rf, X, y)

# ---- Model 3: Extra Trees (high-accuracy benchmark) ---------------------
# NOTE: depth capped at 16 / 150 trees instead of the uncapped 500-tree
# version. Uncapped gave MAE 0.066 / R2 97.86% at a 425MB file size;
# this version gives MAE 0.070 / R2 97.78% at ~22MB -- a negligible
# accuracy trade for something that actually fits in a GitHub repo and
# loads quickly in the deployed app.
et_model = ExtraTreesRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
et_model.fit(X_train, y_train)
y_pred_et = et_model.predict(X_test)
et_metrics = evaluate(et_model, y_test, y_pred_et, X, y)

print("=" * 55)
for name, m in [("Linear Regression", lin_metrics), ("Random Forest", rf_metrics), ("Extra Trees", et_metrics)]:
    print(name)
    for k, v in m.items():
        print(f"  {k:12s}: {v:.4f}")
    print("-" * 55)

# ---- Persist everything the app needs -----------------------------------
joblib.dump(lin_model, "model/linear_model.pkl")
joblib.dump(rf_model, "model/rf_model.pkl")
joblib.dump(et_model, "model/extratrees_model.pkl", compress=3)

feature_coefs = dict(zip(FEATURES, lin_model.coef_.tolist()))

artifact = {
    "features": FEATURES,
    "target": TARGET,
    "dataset_path": DATA_PATH,
    "n_samples": len(df),
    "linear_metrics": lin_metrics,
    "rf_metrics": rf_metrics,
    "et_metrics": et_metrics,
    "linear_intercept": float(lin_model.intercept_),
    "linear_coefficients": feature_coefs,
    "feature_ranges": {
        f: {"min": float(df[f].min()), "max": float(df[f].max()), "mean": float(df[f].mean())}
        for f in FEATURES
    },
    "y_test": y_test.tolist(),
    "y_pred_linear": y_pred_lin.tolist(),
    "y_pred_rf": y_pred_rf.tolist(),
    "y_pred_et": y_pred_et.tolist(),
}

with open("model/artifact.json", "w") as f:
    json.dump(artifact, f, indent=2)

print("Saved: model/linear_model.pkl, model/rf_model.pkl, model/extratrees_model.pkl, model/artifact.json")
