import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LocalWindFieldDataset(Dataset):
    def __init__(self, csv_file, rect_fol, wind_fol, root_dir="", local=2):
        print("Local field set to: ", local)
        self.main_data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.rect_file_path = os.path.join(root_dir, rect_fol)
        self.wind_file_path = os.path.join(root_dir, wind_fol)
        self.local = local

    def __len__(self):
        return len(self.main_data)

    def getLocalWinds(self, robo_coords, winds):
        startcoord = robo_coords - self.local
        endcoord = robo_coords + self.local + 1
        if (
            startcoord[0] < 0
            or startcoord[1] < 0
            or endcoord[0] >= winds.shape[0]
            or endcoord[1] >= winds.shape[1]
        ):
            print("Out of bounds")
            return None
        local_winds = winds[startcoord[0] : endcoord[0], startcoord[1] : endcoord[1]]
        return torch.tensor(local_winds, dtype=torch.float64)

    def __getitem__(self, idx):
        data = self.main_data.iloc[idx]
        rect_path = os.path.join(self.rect_file_path, str(data["index"]) + "_r.npy")
        winds_path = os.path.join(self.wind_file_path, str(data["index"]) + "_m.npy")
        winds = np.load(winds_path)
        robo_coords = torch.tensor([data["xr"], data["yr"]])
        # print(f"Robot Coordinates: {robo_coords}")
        local_winds = self.getLocalWinds(robo_coords, winds)
        rect = np.load(rect_path)
        # lidar = getLidar(rects)
        lidar = torch.rand(360, dtype=torch.float64)
        return (
            lidar,
            torch.tensor(winds[robo_coords[0]][robo_coords[0]], dtype=torch.float64),
            local_winds,
        )
