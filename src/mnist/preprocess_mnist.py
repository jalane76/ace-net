# -*- coding: utf-8 -*-
from art.utils import load_mnist
import click
import json
import numpy as np
import os
import torch

@click.command()
@click.argument('x_train_output_path', type=click.Path())
@click.argument('y_train_output_path', type=click.Path())
@click.argument('x_test_output_path', type=click.Path())
@click.argument('y_test_output_path', type=click.Path())
@click.argument('covariance_output_path', type=click.Path())
@click.argument('means_output_path', type=click.Path())
@click.argument('clip_values_output_path', type=click.Path())
def main(x_train_output_path, y_train_output_path, x_test_output_path, y_test_output_path, covariance_output_path, means_output_path, clip_values_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    clip_values = {
        'min_pixel_value': min_pixel_value,
        'max_pixel_value': max_pixel_value
    }

    # Swap axes to PyTorch's NCHW format

    x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

    x_obvs = torch.Tensor(x_train)

    means = x_obvs.mean(dim=0)
    means = means.unsqueeze(dim=0)

    x_obvs = x_obvs.reshape((-1, 1, 28 * 28))  # flatten so we can get a real covariance matrix
    covariance = torch.Tensor(np.cov(x_obvs[:, 0, :], rowvar=False))

    # Save data
    torch.save(x_train, x_train_output_path)
    torch.save(y_train, y_train_output_path)
    torch.save(x_test, x_test_output_path)
    torch.save(y_test, y_test_output_path)
    torch.save(covariance, covariance_output_path)
    torch.save(means, means_output_path)

    with open(clip_values_output_path, mode='w') as f:
        json.dump(clip_values, f)

if __name__ == '__main__':
    main()
