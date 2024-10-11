import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from rich import print
import torch
from dataset import LocalWindFieldDataset
from model import BasicMLP

NUM_EPOCHS = 100

LR = 0.001


def main():
    local_wind_field_dataset = LocalWindFieldDataset(
        config_file="config.csv",
        map_file="map.csv",
        rect_fol="rects",
        map_fol="maps",
        root_dir="data",
        local=3,
    )
    model = BasicMLP(local=local_wind_field_dataset.local)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1}")
        for i, sample in enumerate(local_wind_field_dataset, 1):
            lidar, wind_at_robot, winds_y = sample
            optimizer.zero_grad()

            lidar = lidar.float()
            wind_at_robot = wind_at_robot.float()
            winds_y = winds_y.float()

            outputs = model(torch.cat((lidar, wind_at_robot.flatten()), dim=0))
            loss = criterion(outputs, winds_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"\t Average Loss: {running_loss/len(local_wind_field_dataset):.4f}")


if __name__ == "__main__":
    main()
