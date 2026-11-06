"""Part D (1/3) - hub profile, K-Means segmentation, separation evidence and tier-vs-segment check."""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile, ascending hub_id (groupby order)
profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id").reset_index(drop=True)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# 2. scale the four profile columns and cluster
CLUSTER_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
Z = StandardScaler().fit_transform(profile[CLUSTER_COLS])


def fit(k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)


# 3. separation evidence: mean between-cluster pair distance / mean within-cluster pair distance
dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
iu = np.triu_indices(len(Z), k=1)  # 276 unordered pairs
sep_rows = []
for k in [2, 3, 4, 5]:
    labels = fit(k).labels_
    same = (labels[:, None] == labels[None, :])[iu]
    d = dist[iu]
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append({"K": k, "across_over_within": round(float(d[~same].mean() / d[same].mean()), 4),
                     "cluster_sizes": "-".join(str(int(s)) for s in sizes)})
pd.DataFrame(sep_rows).to_csv("outputs/part_d_separation.csv", index=False)

# 4. name segments by mean delay rate: lowest Standard, middle Watch, highest Escalate
km = fit(3)
profile["cluster"] = km.labels_
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

seg = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")[["hub_id", "segment", "hub_city", "hub_tier"]]
seg.to_csv("outputs/hub_segments.csv", index=False)

sp = profile.groupby("segment").agg(
    hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"), delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"), mean_n_stops=("mean_n_stops", "mean")).reset_index()
sp["segment"] = pd.Categorical(sp["segment"], ["Standard", "Watch", "Escalate"])
sp.sort_values("segment").round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = seg[seg["segment"] != seg["hub_tier"].map(implied)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("segments:", seg["segment"].value_counts().to_dict(), "| mismatches:", len(mismatch))
