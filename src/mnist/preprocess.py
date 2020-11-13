#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from art.utils import load_mnist
import click
from common.config import load_config
import json
import numpy as np
import os
import torch


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.path.isfile(config.x_train_output_path):
        click.confirm(f"Overwrite {config.x_train_output_path}?", abort=True)
    if os.path.isfile(config.y_train_output_path):
        click.confirm(f"Overwrite {config.y_train_output_path}?", abort=True)
    if os.path.isfile(config.x_test_output_path):
        click.confirm(f"Overwrite {config.x_test_output_path}?", abort=True)
    if os.path.isfile(config.y_test_output_path):
        click.confirm(f"Overwrite {config.y_test_output_path}?", abort=True)
    if os.path.isfile(config.covariance_output_path):
        click.confirm(f"Overwrite {config.covariance_output_path}?", abort=True)
    if os.path.isfile(config.means_output_path):
        click.confirm(f"Overwrite {config.means_output_path}?", abort=True)
    if os.path.isfile(config.clip_values_output_path):
        click.confirm(f"Overwrite {config.clip_values_output_path}?", abort=True)


    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load data
    (
        (x_train, y_train),
        (x_test, y_test),
        min_pixel_value,
        max_pixel_value,
    ) = load_mnist()
    clip_values = {
        "min_pixel_value": min_pixel_value,
        "max_pixel_value": max_pixel_value,
    }

    # Swap axes to PyTorch's NCHW format

    x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

    x_obvs = torch.Tensor(x_train)

    means = x_obvs.mean(dim=0)
    means = means.unsqueeze(dim=0)

    x_obvs = x_obvs.reshape(
        (-1, 1, 28 * 28)
    )  # flatten so we can get a real covariance matrix
    covariance = torch.Tensor(np.cov(x_obvs[:, 0, :], rowvar=False))

    # Save data
    torch.save(x_train, config.x_train_output_path)
    torch.save(y_train, config.y_train_output_path)
    torch.save(x_test, config.x_test_output_path)
    torch.save(y_test, config.y_test_output_path)
    torch.save(covariance, config.covariance_output_path)
    torch.save(means, config.means_output_path)

    with open(config.clip_values_output_path, mode="w") as f:
        json.dump(clip_values, f)


if __name__ == "__main__":
    main()
