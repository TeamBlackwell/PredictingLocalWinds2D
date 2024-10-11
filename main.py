import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from rich import print
import torch
from dataset import LocalWindFieldDataset
from model import BasicMLP
from tqdm import tqdm
from sys import platform

NUM_EPOCHS = 100
LR = 0.001
LOCAL_FIELD_SIZE = 15
TRAIN_SIZE = 0.8

if platform == "darwin":
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    local_wind_field_dataset = LocalWindFieldDataset(
        config_file="data.csv",
        map_file="map_mapping.csv",
        rect_fol="rects",
        map_fol="wind_fields",
        root_dir="data_complete",
        local=LOCAL_FIELD_SIZE,
        device=DEVICE,
    )
    # split dataset into training and validation
    train_size_n = int(TRAIN_SIZE * len(local_wind_field_dataset))
    val_size_n = len(local_wind_field_dataset) - train_size_n
    train_dataset, val_dataset = torch.utils.data.random_split(
        local_wind_field_dataset, [train_size_n, val_size_n]
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = BasicMLP(local=local_wind_field_dataset.local, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    training_losses = []
    validation_losses = []

    # training loop
    for epoch in (ep_pbar := tqdm(range(1, NUM_EPOCHS + 1))):

        ep_pbar.set_description(f"Doing Epoch {epoch}/{NUM_EPOCHS}")

        # TRAINING SECTION
        train_loss = 0.0

        model.train()
        for i, sample in enumerate(pbar := tqdm(train_loader), 1):

            lidar, wind_at_robot, winds_y = sample

            # print(lidar.shape, wind_at_robot.shape, winds_y.shape)
            # (Batch, 360), (Batch, 2), (Batch, 2*local + 1, 2*local + 1, 2)
            X_val = torch.cat((lidar, wind_at_robot), dim=1)

            optimizer.zero_grad()

            output = model(X_val)
            loss = criterion(output, winds_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_val.size(0)

            pbar.set_description(f"Training. Loss: {(train_loss / i):.4f}")

        # EVALUATION
        valid_loss = 0.0
        model.eval()
        for i, sample in enumerate(pbar := tqdm(val_loader), 1):

            lidar, wind_at_robot, winds_y = sample
            X_val = torch.cat((lidar, wind_at_robot), dim=1)
            output = model(X_val)
            loss = criterion(output, winds_y)

            valid_loss += loss.item() * X_val.size(0)

            pbar.set_description(f"Validating. Loss: {(valid_loss / i):.4f}")

        train_loss = train_loss / train_size_n
        valid_loss = valid_loss / val_size_n

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)

        tqdm.write(
            f"Epoch {epoch}/{NUM_EPOCHS} Summary: Train Loss: {train_loss:.6f} | Validation Loss: {valid_loss:.6f}\n"
        )

    # Plot the loss curve
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), training_losses, label="Training Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
