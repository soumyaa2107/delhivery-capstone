"""Part C - delay classifier: stratified split, baseline, feature signal, tuning,
alert threshold chosen on validation, and one test evaluation."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)
table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cells(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def metrics_from_cells(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": round((tn + tp) / total, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4)}


def row_for(label_key, label, y_true, y_pred):
    tn, fp, fn, tp = cells(y_true, y_pred)
    return {label_key: label, "TN": tn, "FP": fp, "FN": fn, "TP": tp, **metrics_from_cells(tn, fp, fn, tp)}


def f1_of(y_true, y_pred):
    return metrics_from_cells(*cells(y_true, y_pred))["f1"]


# 2. blind baseline + three classifiers (default 0.5 rule)
model_rows = [row_for("model", "blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]
logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append(row_for("model", "logistic_regression", y_val, logit.predict(X_val)))

best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_of(y_val, t.predict(X_val))
    if f > best_f1:  # strict > keeps the smaller depth on ties
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
model_rows.append(row_for("model", f"decision_tree_depth_{best_d}", y_val, tree.predict(X_val)))
rf_default = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append(row_for("model", "random_forest_default", y_val, rf_default.predict(X_val)))
pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)

# 3. which columns carry signal?
skb_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
scores = pd.DataFrame({"feature": FEATURES, "f_score": skb_all.scores_,
                       "rf_importance": rf_default.feature_importances_})
scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
scores[["f_score", "rf_importance"]] = scores[["f_score", "rf_importance"]].round(4)
scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_of(y_val, p.predict(X_val))})
pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning on the 7 selected columns (selector fitted on training rows only; test rows untouched)
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr7, y_train)

candidates = {
    "grid_search": {"params": gs.best_params_, "model": gs.best_estimator_},
    "randomized_search": {"params": rs.best_params_, "model": rs.best_estimator_},
    "default_forest_same_7_features": {"params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
                                       "model": default7},
}
tune_rows = []
for name, search in [("grid_search", gs), ("randomized_search", rs), ("default_forest_same_7_features", None)]:
    c = candidates[name]
    is_search = search is not None
    cand = len(search.cv_results_["params"]) if is_search else 1
    folds = 5 if is_search else 0
    tune_rows.append({"search": name, "candidates": cand, "cv_folds": folds, "total_fits": cand * folds if is_search else 1,
                      "best_params": json.dumps(c["params"], sort_keys=True),
                      "best_cv_accuracy": round(search.best_score_, 4) if is_search else "",
                      "val_f1": f1_of(y_val, c["model"].predict(Xva7))})
tuning = pd.DataFrame(tune_rows)
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1, ties -> default, grid, randomized
order = ["default_forest_same_7_features", "grid_search", "randomized_search"]
chosen = max(order, key=lambda n: (tuning.set_index("search").loc[n, "val_f1"], -order.index(n)))
cp = candidates[chosen]["params"]
final = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                  ("model", RandomForestClassifier(random_state=42, **cp))]).fit(X_train, y_train)

# 6. alert threshold from validation probabilities
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
p_val = final.predict_proba(X_val)[:, 1]
thr_rows = []
for t in THRESHOLDS:
    thr_rows.append(row_for("threshold", t, y_val, (p_val >= t).astype(int)))
thr = pd.DataFrame(thr_rows)
thr.to_csv("outputs/part_c_thresholds.csv", index=False)
ok = thr[thr["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1

# 7. one test evaluation
p_test = final.predict_proba(X_test)[:, 1]
test_row = row_for("threshold", threshold, y_test, (p_test >= threshold).astype(int))
pd.DataFrame([test_row]).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": 7, "n_estimators": cp["n_estimators"],
               "max_depth": "unlimited" if cp["max_depth"] is None else cp["max_depth"],
               "min_samples_leaf": cp["min_samples_leaf"], "threshold": threshold,
               "chosen_from": chosen}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("chosen:", chosen, cp, "threshold:", threshold)
