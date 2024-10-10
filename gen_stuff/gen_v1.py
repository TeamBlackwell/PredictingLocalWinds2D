"""
A basic version of generation.

Bottomline: only supports one Speed_x and Speed_y right now.

"""

import numpy as np
from gen_stuff.phiflow_runner import run_flow, save_flow
from procedural_generation import generator
from procedural_generation.sampling import Tag, sample_poisson_disk
from pathlib import Path

DATA_OUT_DIR = Path("../data_complete")
DATA_RECT_DIR = DATA_OUT_DIR / "rects"
DATA_WIND_FILEDS_DIR = DATA_OUT_DIR / "wind_fields"

MAP_FILE_PATH = DATA_OUT_DIR / "map_mapping.csv"

MAP_SIZE = 100  # the map is square

SPEED_X = 5
SPEED_Y = -5


def create_rects(n_samples=100):
    for i in range(n_samples):

        generatingMachine = generator.Generator(
            2,
            [
                (sample_poisson_disk, Tag.SKYSCRAPER, {"density": 28}),
                (sample_poisson_disk, Tag.HOUSE, {"density": 15, "n_buildings": 75}),
            ],
            scale=MAP_SIZE,
        )

        generatingMachine.generate_sample()
        building_array = np.array(generatingMachine.buildings)
        flow_data = run_flow(building_array, 100, 100, MAP_SIZE, SPEED_X, SPEED_Y)

        generatingMachine.export(DATA_RECT_DIR / f"{i}_r.npy")
        save_flow(flow_data, DATA_WIND_FILEDS_DIR / f"{i}_m.npy")


if __name__ == "__main__":
    DATA_OUT_DIR.mkdir(exist_ok=True)
    DATA_RECT_DIR.mkdir(exist_ok=True)
    DATA_WIND_FILEDS_DIR.mkdir(exist_ok=True)

    create_rects(n_samples=1)
