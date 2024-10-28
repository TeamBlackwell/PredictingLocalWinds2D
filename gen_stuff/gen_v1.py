"""
A basic version of generation.

Bottomline: only supports one Speed_x and Speed_y right now.

"""

import random
import numpy as np
from phiflow_runner import run_flow, run_flow3d, save_flow
from procedural_generation import generator
from procedural_generation.sampling import Tag, sample_poisson_disk
from pathlib import Path
import pandas as pd
from tqdm import trange

import warnings

warnings.filterwarnings("ignore")

SEED = 100

random.seed(SEED)
np.random.seed(SEED)

DATA_OUT_DIR = Path("../data_test")
DATA_RECT_DIR = DATA_OUT_DIR / "rects"
DATA_WIND_FIELDS_DIR = DATA_OUT_DIR / "wind_fields"

MAP_FILE_PATH = DATA_OUT_DIR / "map_mapping.csv"
DATA_FILE_PATH = DATA_OUT_DIR / "data.csv"

DATA_FILE_PATH_3D = DATA_OUT_DIR / "data_3d.csv"
MAP_FILE_PATH_3D = DATA_OUT_DIR / "map_mapping_3d.csv"

MAP_SIZE = 100  # the map is square

# TODO: make these parameters, so we can have multiple speeds, multiple inlets
SPEED_X = 5
SPEED_Y = -5
SPEED_Z = 2

XR_choices = np.arange(40, MAP_SIZE - 40 + 1, 1)
YR_choices = np.arange(40, MAP_SIZE - 40 + 1, 1)


def create_rects(
    n_samples=100,
    data_file_path=DATA_FILE_PATH,
    map_file_path=MAP_FILE_PATH,
    pre_time: int = 100,
    avg_time_window: int = 200,
    pre_done_count: int = 0,
):

    data_df = pd.DataFrame(columns=["map_id", "xr", "yr"])
    map_df = pd.DataFrame(columns=["map_id", "speed_x", "speed_y", "rect_id"])

    if pre_done_count == 0:
        data_df.to_csv(data_file_path, index=False)
        map_df.to_csv(map_file_path, index=False)
    else:
        if not data_file_path.exists() or not map_file_path.exists():
            raise ValueError(
                f"Pre-done count is {pre_done_count} but data or map file does not exist."
            )
        data_df = pd.read_csv(data_file_path)
        map_df = pd.read_csv(map_file_path)
        map_df = map_df[["map_id", "speed_x", "speed_y", "rect_id"]]
        data_df = data_df[["map_id", "xr", "yr"]]

    start_idx = pre_done_count + 1 if pre_done_count > 0 else 0

    print(f"Starting from {start_idx}, going till {start_idx + n_samples}")

    for i in trange(start_idx, start_idx + n_samples):

        # XXX: Seeding here is not working, it causes a recursion error down the line. Looks like the poisson disk sampling is not deterministic.
        generatingMachine = generator.Generator(
            2,
            [
                (
                    sample_poisson_disk,
                    Tag.SKYSCRAPER,
                    {"density": 28, "seed": None, "n_buildings": 20},
                ),
                (
                    sample_poisson_disk,
                    Tag.HOUSE,
                    {"density": 15, "n_buildings": 20, "seed": None},
                ),
            ],
            scale=MAP_SIZE,  # TODO: make a system which allows us to use map_size = 100 but scale = 80, essentially adding a border.
        )

        generatingMachine.generate_sample()
        generatingMachine.export(DATA_RECT_DIR / f"{i}_r.npy")

        building_array = np.array(generatingMachine.buildings)
        try:
            flow_data = run_flow(
                building_array, pre_time, avg_time_window, MAP_SIZE, SPEED_X, SPEED_Y
            )
            save_flow(flow_data, DATA_WIND_FIELDS_DIR / f"{i}_m.npy")
        except ValueError as e:
            print(f"Failed to generate flow for {i} - {e}")
            continue

        xr = np.random.choice(XR_choices)
        yr = np.random.choice(YR_choices)

        data_df.loc[i] = [i, xr, yr]
        map_df.loc[i] = [i, SPEED_X, SPEED_Y, i]

        data_df.to_csv(data_file_path, index=False)
        map_df.to_csv(map_file_path, index=False)


def create_rects_3d(
    n_samples=100,
    data_file_path=DATA_FILE_PATH_3D,
    map_file_path=MAP_FILE_PATH_3D,
    pre_time: int = 100,
    avg_time_window: int = 200,
    pre_done_count: int = 0,
):

    data_df = pd.DataFrame(columns=["map_id", "xr", "yr", "zr"])
    map_df = pd.DataFrame(
        columns=["map_id", "speed_x", "speed_y", "speed_z", "rect_id"]
    )

    if pre_done_count == 0:
        data_df.to_csv(data_file_path, index=False)
        map_df.to_csv(map_file_path, index=False)
    else:
        if not data_file_path.exists() or not map_file_path.exists():
            raise ValueError(
                f"Pre-done count is {pre_done_count} but data or map file does not exist."
            )
        data_df = pd.read_csv(data_file_path)
        map_df = pd.read_csv(map_file_path)
        map_df = map_df[["map_id", "speed_x", "speed_y", "speed_z", "rect_id"]]
        data_df = data_df[["map_id", "xr", "yr", "zr"]]

    start_idx = pre_done_count + 1 if pre_done_count > 0 else 0

    print(f"Starting from {start_idx}, going till {start_idx + n_samples}")

    for i in trange(start_idx, start_idx + n_samples):

        # XXX: Seeding here is not working, it causes a recursion error down the line. Looks like the poisson disk sampling is not deterministic.
        generatingMachine = generator.Generator(
            2,
            [
                (
                    sample_poisson_disk,
                    Tag.SKYSCRAPER,
                    {"density": 28, "seed": None, "n_buildings": 20},
                ),
                (
                    sample_poisson_disk,
                    Tag.HOUSE,
                    {"density": 15, "n_buildings": 20, "seed": None},
                ),
            ],
            scale=MAP_SIZE,  # TODO: make a system which allows us to use map_size = 100 but scale = 80, essentially adding a border.
        )

        generatingMachine.generate_sample()
        generatingMachine.export(DATA_RECT_DIR / f"{i}_r.npy")

        building_array = np.array(generatingMachine.buildings)

        # add heights
        building_array = np.concatenate(
            [building_array, np.random.randint(2, 10, (building_array.shape[0], 1))],
            axis=1,
        )

        # print(building_array.shape)

        try:
            flow_data = run_flow3d(
                building_array,
                pre_time,
                avg_time_window,
                MAP_SIZE,
                [SPEED_X, SPEED_Y, SPEED_Z],
            )
            save_flow(flow_data, DATA_WIND_FIELDS_DIR / f"{i}_m.npy")
        except ValueError as e:
            print(f"Failed to generate flow for {i} - {e}")
            continue

        xr = np.random.choice(XR_choices)
        yr = np.random.choice(YR_choices)
        zr = np.random.randint(2, 5)

        data_df.loc[i] = [i, xr, yr, zr]
        map_df.loc[i] = [i, SPEED_X, SPEED_Y, SPEED_Z, i]

        data_df.to_csv(data_file_path, index=False)
        map_df.to_csv(map_file_path, index=False)


if __name__ == "__main__":
    DATA_OUT_DIR.mkdir(exist_ok=True)
    DATA_RECT_DIR.mkdir(exist_ok=True)
    DATA_WIND_FIELDS_DIR.mkdir(exist_ok=True)

    create_rects_3d(
        n_samples=1,
        data_file_path=DATA_FILE_PATH,
        map_file_path=MAP_FILE_PATH,
        pre_time=2,
        avg_time_window=4,
        pre_done_count=0,
    )

    # create_rects(
    #     n_samples=1,
    #     data_file_path=DATA_FILE_PATH,
    #     map_file_path=MAP_FILE_PATH,
    #     pre_time=2,
    #     avg_time_window=4,
    #     pre_done_count=0,
    # )

    # shapes:
    # rect: (n, 4)
    # map: (map_size, map_size, 2)
