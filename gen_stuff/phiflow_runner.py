from pathlib import Path
import numpy as np
from phi.torch import flow
from tqdm import trange


def run_flow(
    rect_data: np.ndarray,
    pre_time: int,
    avg_time_window: int,
    map_size: int,
    speed_x: float,
    speed_y: float,
) -> np.ndarray:
    """
    Run a flow simulation with the given parameters. Mainly with the rectangles and the speed of winds.
    Currently supports only one speed_x and speed_y (one wind field).

    :param rect_data: np.ndarray of shape (n, 4) where n is the number of rectangles and 4 is the x1, y1, x2, y2
    :param pre_time: int, the number of time steps to run before average window starts
    :param avg_time_window: int, the number of time steps to average over, counted after pre_time
    :param map_size: int, the size of the map
    :param speed_x: float, the x speed
    :param speed_y: float, the y speed

    :return v_data: np.ndarray of shape (2, map_size, map_size), the velocity data
    """

    # for each file, run the flow
    SPEEDS = flow.tensor([speed_x, speed_y])

    velocity = flow.StaggeredGrid(
        SPEEDS,
        flow.ZERO_GRADIENT,
        x=map_size,
        y=map_size,
        bounds=flow.Box(x=map_size, y=map_size),
    )

    cuboid_list = []
    for start, end in rect_data:
        cuboid_list.append(
            flow.Box(flow.vec(x=start.x, y=start.y), flow.vec(x=end.x, y=end.y))
        )

    # make all of them obstacles

    obstacle_list = []
    for cuboid in cuboid_list:
        obstacle_list.append(flow.Obstacle(cuboid))

    BOUNDARY_BOX = flow.Box(x=(-1 * flow.INF, 0.5), y=None)
    BOUNDARY_MASK = flow.StaggeredGrid(
        BOUNDARY_BOX, velocity.extrapolation, velocity.bounds, velocity.resolution
    )

    pressure = None

    @flow.math.jit_compile
    def step(v, p):
        v = flow.advect.semi_lagrangian(v, v, 1.0)
        v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (
            speed_x,
            speed_y,
        )  # make sure you dont simulat OOB
        v, p = flow.fluid.make_incompressible(
            v, obstacle_list, flow.Solve("auto", 1e-5, x0=p)
        )  # make it do the boundary thign
        return v, p

    v_data, p_data = flow.iterate(
        step, flow.batch(time=300), velocity, pressure, range=trange
    )

    # visualization
    # anim = flow.plot(
    #     [traj.curl(), *cuboid_list[::-1]],
    #     animate="time",
    #     size=(6, 6),
    #     frame_time=10,
    #     overlay="list",
    # )
    # plt.show()

    return v_data.numpy()


def save_flow(flow_data: np.ndarray, path: Path | str) -> None:
    """
    Save the flow data, which is a np.ndarray of shape (2, map_size, map_size) to the given path.

    Saves any numpy array, of any shape. Saves to the given path. Path should end with .npy.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.endswith(".npy"):
        raise ValueError("Path should end with .npy")

    np.save(path, flow_data)
