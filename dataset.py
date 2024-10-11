import os

import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader


class LocalWindFieldDataset(Dataset):

    def __init__(
        self,
        config_file,
        map_file,
        rect_fol,
        map_fol,
        root_dir="",
        local=2,
        device=torch.device("cpu"),
    ):

        print("Local field set to: ", local)
        self.config_data = pd.read_csv(os.path.join(root_dir, config_file))
        self.map_data = pd.read_csv(os.path.join(root_dir, map_file))
        self.rect_file_path = os.path.join(root_dir, rect_fol)
        self.map_file_path = os.path.join(root_dir, map_fol)
        self.local = local

        self.map_data.set_index("map_id", inplace=True)

        self.device = device

        # self.cdt = self.config_data
        # self.mdt = self.map_data

    def __len__(self):
        return len(self.config_data)

    def _get_local_winds(self, robo_coords, winds):
        startcoord = robo_coords - self.local
        endcoord = robo_coords + self.local + 1

        if (
            startcoord[0] < 0
            or startcoord[1] < 0
            or endcoord[0] >= winds.shape[0]
            or endcoord[1] >= winds.shape[1]
        ):
            # TODO: can this instead be checked in the beginning, so this index is never reached?
            # as in data cleaning
            print("Out of bounds")

            return None

        local_winds = winds[startcoord[0] : endcoord[0], startcoord[1] : endcoord[1]]

        if np.isnan(local_winds).any():
            print("BIG ERROR: Nan values in local winds")
            return None

        return torch.tensor(local_winds, dtype=torch.float32, device=self.device)

    def __getitem__(self, idx):

        robot_config = self.config_data.iloc[idx]
        map_config = self.map_data.loc[robot_config["map_id"]]

        rect_path = os.path.join(
            self.rect_file_path, str(map_config["rect_id"]) + "_r.npy"
        )
        winds_path = os.path.join(
            self.map_file_path, str(robot_config["map_id"]) + "_m.npy"
        )
        winds = np.load(winds_path)
        robo_coords = torch.tensor([robot_config["xr"], robot_config["yr"]])

        local_winds = self._get_local_winds(robo_coords, winds)
        if local_winds is None:
            # self.cdt = self.cdt.drop(idx)
            # self.mdt = self.mdt.drop(robot_config["map_id"])

            # self.cdt.to_csv("myfacedata.csv", index=False)
            # self.mdt["map_id"] = self.mdt.index
            # self.mdt.to_csv("myfacemap.csv", index=False)

            local_winds = torch.zeros(
                2 * self.local + 1,
                2 * self.local + 1,
                2,
                dtype=torch.float64,
                device=self.device,
            )
        rects = np.load(rect_path)
        # lidar = getLidar(rects)
        lidar = torch.rand(
            360, dtype=torch.float32, device=self.device
        )  # TODO: TEMPORARY

        return (
            lidar,
            torch.tensor(
                winds[robo_coords[0], robo_coords[1]],
                dtype=torch.float32,
                device=self.device,
            ),
            local_winds,
        )
