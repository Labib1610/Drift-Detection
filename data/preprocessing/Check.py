import numpy as np, pyarrow.parquet as pq
n_parquet = pq.read_metadata("data/interim/ns_panel.parquet").num_rows
npz = np.load("data/streams/ns_panel_perms.npz")
print(n_parquet, len(npz["perm_00"]))