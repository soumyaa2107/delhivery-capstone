"""Part B - delivery-hours regression: split, leakage-safe scaling, baseline, depth diagnostic,
four candidates, learning-rate experiment, cross-validation and a single test evaluation."""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)
table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]

# 1. split 70 / 15 / 15
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 2. scaler fitted on training rows only
scaler = StandardScaler().fit(X_train)
train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test),
              round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
              round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)],
}).to_csv("outputs/part_b_checks.csv", index=False)


def score(name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": name, "split": split, "MAE": round(mean_absolute_error(y_true, y_pred), 4),
            "MSE": round(mse, 4), "RMSE": round(float(np.sqrt(mse)), 4), "R2": round(r2_score(y_true, y_pred), 4)}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# 3. baseline
rows = [score("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

# 4. tree-depth diagnostic
depth_rows, best_depth, best_val = [], None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr, va = rmse(y_train, t.predict(X_train)), rmse(y_val, t.predict(X_val))
    depth_rows.append({"max_depth": "unlimited" if d is None else d, "train_rmse": round(tr, 4), "val_rmse": round(va, 4)})
    if round(va, 6) < round(best_val, 6):  # strict improvement keeps the smaller depth on ties
        best_depth, best_val = d, va
pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)

# 5. four candidates
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
val_rmse = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_val)
    rows.append(score(name, "val", y_val, pred))
    val_rmse[name] = rmse(y_val, pred)

# 6. learning-rate experiment
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    p = g.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": round(rmse(y_val, p), 4), "val_r2": round(r2_score(y_val, p), 4)})
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# 7. cross-validation of the two best, then one test evaluation of the winner
ranking = sorted(val_rmse, key=val_rmse.get)
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # one object shared by both models
cv_rows = []
for name in ranking[:2]:
    s = cross_val_score(models[name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": name, "mean_r2": round(s.mean(), 4), "std_r2": round(s.std(), 4)})
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

final = ranking[0]
rows.append(score(final, "test", y_test, models[final].predict(X_test)))  # the only test scoring
pd.DataFrame(rows).to_csv("outputs/part_b_metrics.csv", index=False)
print("Validation ranking:", ranking, "| best tree depth:", best_depth, "| final model:", final)
