import pandas as pd, numpy as np, json
from river import drift

w = pd.read_parquet("features/windows/bn_panel__xlm-roberta-base__perm00.parquet").sort_values("window_idx")
n = len(w); det = w.iloc[int(0.10*n):].reset_index(drop=True)      # detection epoch
z = det["z_S7"].to_numpy(float); dates = pd.to_datetime(det["median_date"])
delta = json.load(open("results/calibration.json"))["calibration"]["S7|xlm-roberta-base|raw"]["targets"]["0.001"]["delta"]

a = drift.ADWIN(delta=delta); alarms = []
for i, x in enumerate(z):
    a.update(float(x))
    if a.drift_detected:
        alarms.append(i)
print("alarms:", len(alarms))                                   # -> 21
print("alarm dates:", [str(dates[i].date()) for i in alarms])   # scattered 2016-2020, incl. 2020-03-12
