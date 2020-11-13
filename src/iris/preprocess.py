#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import click
from common.config import load_config
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if (
        os.path.isfile(config.x_train_output_path)
        or os.path.isfile(config.y_train_output_path)
        or os.path.isfile(config.covariance_output_path)
        or os.path.isfile(config.means_output_path)
    ):
        click.confirm(f"Overwrite files?", abort=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset = pd.read_csv(config.input_filepath)
    dataset = pd.get_dummies(dataset, columns=["species"])  # One Hot Encoding
    values = list(dataset.columns.values)

    y = dataset[values[-3:]]
    y = np.array(y, dtype="float32")
    x = dataset[values[:-3]]
    x = np.array(x, dtype="float32")

    # Shuffle Data
    indices = np.random.choice(len(x), len(x), replace=False)
    x_values = x[indices]

    scaler = MinMaxScaler()

    test_size = 30
    x_train = x_values[:-test_size]
    x_train = scaler.fit_transform(x_train)

    x_values = scaler.transform(x_values)
    y_values = y[indices]

    x_tensor = torch.Tensor(x_values)
    y_tensor = torch.Tensor(y_values)

    # Calculate observational statistics
    covariance = torch.Tensor(np.cov(x_values, rowvar=False))
    means = torch.Tensor(np.mean(x_values, axis=0))

    # Save data
    torch.save(x_tensor, config.x_train_output_path)
    torch.save(y_tensor, config.y_train_output_path)
    torch.save(covariance, config.covariance_output_path)
    torch.save(means, config.means_output_path)


if __name__ == "__main__":
    main()
