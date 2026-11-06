"""Part D (3/3) - score new parcels and turn the probabilities into an action list.

Run after part_d_segments.py and package_model.py:  python predict_delay.py
"""
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)
new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"].astype(float)))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new_raw, medians=medians)
feats = build_features(clean, hubs)
out = feats[["shipment_id", "hub_id"]].copy()
out["delay_probability"] = model.predict_proba(feats[feature_order])[:, 1].round(4)
out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
out = out.merge(segments, on="hub_id", how="left")


def action(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call" if row["flagged"] == 1 else "No action"


out["action"] = out.apply(action, axis=1)
out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]] \
    .to_csv("outputs/new_shipment_actions.csv", index=False)
print(out["action"].value_counts().to_string())
