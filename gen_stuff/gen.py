from procedural_generation import generator
from procedural_generation.sampling import Tag, sample_poisson_disk
from pathlib import Path
from phi.torch import flow
from tqdm import trange
import matplotlib.pyplot as plt

DATA_OUT_DIR = Path("../data_complete")
DATA_RECT_DIR = DATA_OUT_DIR / "rects"
DATA_WIND_FILEDS_DIR = DATA_OUT_DIR / "wind_fields"

MAP_FILE_PATH = DATA_OUT_DIR / "map_mapping.csv"


WORLD_WIDTH = 100
WORLD_HEIGHT = 100

SPEED_X = 5
SPEED_Y = -5

SPEEDS = flow.tensor([SPEED_X, SPEED_Y])


def create_rects(n_samples=100):
    for i in range(n_samples):

        generatingMachine = generator.Generator(
            2,
            [
                (sample_poisson_disk, Tag.SKYSCRAPER, {"density": 28}),
                (sample_poisson_disk, Tag.HOUSE, {"density": 15, "n_buildings": 75}),
            ],
        )

        generatingMachine.generate_sample()
        generatingMachine.export(DATA_RECT_DIR / f"{i}_r.npy")


def run_flow():
    # read all files in DATA_RECT_DIR
    for file in DATA_RECT_DIR.glob("*.npy"):
        print(file)

    # for each file, run the flow
    SPEEDS = flow.tensor([SPEED_X, SPEED_Y])

    velocity = flow.StaggeredGrid(
        SPEEDS,
        flow.ZERO_GRADIENT,
        x=WORLD_WIDTH,
        y=WORLD_HEIGHT,
        bounds=flow.Box(x=WORLD_WIDTH, y=WORLD_HEIGHT),
    )


if __name__ == "__main__":
    DATA_OUT_DIR.mkdir(exist_ok=True)
    DATA_RECT_DIR.mkdir(exist_ok=True)
    DATA_WIND_FILEDS_DIR.mkdir(exist_ok=True)

    # create_rects(n_samples=100)
    # map_index,rect_index,wind_x,wind_y
    # run_flow()
