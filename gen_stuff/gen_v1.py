"""
A basic version of generation.

Bottomline: only supports one Speed_x and Speed_y right now.

"""

import random
from matplotlib import pyplot as plt
import numpy as np
from phiflow_runner import run_flow, save_flow
from procedural_generation import generator
from procedural_generation.sampling import Tag, sample_poisson_disk
from pathlib import Path
import pandas as pd

SEED = 100

random.seed(SEED)
np.random.seed(SEED)

DATA_OUT_DIR = Path("../data_complete")
DATA_RECT_DIR = DATA_OUT_DIR / "rects"
DATA_WIND_FIELDS_DIR = DATA_OUT_DIR / "wind_fields"

MAP_FILE_PATH = DATA_OUT_DIR / "map_mapping.csv"
DATA_FILE_PATH = DATA_OUT_DIR / "data.csv"

MAP_SIZE = 100  # the map is square

# TODO: make these parameters, so we can have multiple speeds, multiple inlets
SPEED_X = 5
SPEED_Y = -5

XR_choices = np.arange(40, MAP_SIZE - 40 + 1, 1)
YR_choices = np.arange(40, MAP_SIZE - 40 + 1, 1)

print(XR_choices)
print(YR_choices)


def create_rects(
    n_samples=100,
    data_file_path=DATA_FILE_PATH,
    map_file_path=MAP_FILE_PATH,
    pre_time: int = 100,
    avg_time_window: int = 200,
):

    data_df = pd.DataFrame(columns=["map_index", "xr", "yr"])
    map_df = pd.DataFrame(columns=["map_index", "speed_x", "speed_y", "rect_index"])

    for i in range(n_samples):

        # XXX: Seeding here is not working, it causes a recursion error down the line. Looks like the poisson disk sampling is not deterministic.
        generatingMachine = generator.Generator(
            2,
            [
                (sample_poisson_disk, Tag.SKYSCRAPER, {"density": 28, "seed": None}),
                (
                    sample_poisson_disk,
                    Tag.HOUSE,
                    {"density": 15, "n_buildings": 75, "seed": None},
                ),
            ],
            scale=MAP_SIZE,  # TODO: make a system which allows us to use map_size = 100 but scale = 80, essentially adding a border.
        )

        generatingMachine.generate_sample()
        generatingMachine.export(DATA_RECT_DIR / f"{i}_r.npy")

        building_array = np.array(generatingMachine.buildings)
        flow_data = run_flow(
            building_array, pre_time, avg_time_window, MAP_SIZE, SPEED_X, SPEED_Y
        )
        save_flow(flow_data, DATA_WIND_FIELDS_DIR / f"{i}_m.npy")

        xr = np.random.choice(XR_choices)
        yr = np.random.choice(YR_choices)

        data_df.loc[i] = [i, xr, yr]
        map_df.loc[i] = [i, SPEED_X, SPEED_Y, i]

        data_df.to_csv(data_file_path, index=False)
        map_df.to_csv(map_file_path, index=False)


if __name__ == "__main__":
    DATA_OUT_DIR.mkdir(exist_ok=True)
    DATA_RECT_DIR.mkdir(exist_ok=True)
    DATA_WIND_FIELDS_DIR.mkdir(exist_ok=True)

    create_rects(
        n_samples=1000,
        data_file_path=DATA_FILE_PATH,
        map_file_path=MAP_FILE_PATH,
        pre_time=100,
        avg_time_window=200,
    )

    # shapes:
    # rect: (n, 4)
    # map: (map_size, map_size, 2)
