import random
import numpy as np
import pandas as pd

MAP_SIZE = 100

XR_choices = set(np.arange(40, MAP_SIZE - 40 + 1, 1))
YR_choices = set(np.arange(40, MAP_SIZE - 40 + 1, 1))

df = pd.read_csv("../data_complete/data.csv")

# for each row, add two more rows that have the same map_id but different xr and yr

new_rows = []

for i, row in df.iterrows():
    xr = row["xr"]
    yr = row["yr"]

    xr_choices = XR_choices - {int(xr)}
    yr_choices = YR_choices - {int(yr)}

    for _ in range(5):
        new_rows.append(
            [
                row["map_id"],
                random.choice(list(xr_choices)),
                random.choice(list(yr_choices)),
            ]
        )

df = df._append(pd.DataFrame(new_rows, columns=["map_id", "xr", "yr"]))
df.to_csv("../data_complete/data_new.csv", index=False)
