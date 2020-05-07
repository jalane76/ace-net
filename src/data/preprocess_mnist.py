# -*- coding: utf-8 -*-
from art.utils import load_mnist
import click
import json
import numpy as np
import os
import torch

@click.command()
@click.argument('output_path', type=click.Path())
def main(output_path):

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

    x_obvs = x_obvs.reshape((-1, 1, 28 * 28))  # flatten so we can get a real covariance matrix
    covariance = torch.Tensor(np.cov(x_obvs[:, 0, :], rowvar=False))

    # Save data
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(x_train, os.path.join(output_path, 'x_train.pt'))
    torch.save(y_train, os.path.join(output_path, 'y_train.pt'))
    torch.save(x_test, os.path.join(output_path, 'x_test.pt'))
    torch.save(y_test, os.path.join(output_path, 'y_test.pt'))
    torch.save(covariance, os.path.join(output_path, 'covariance.pt'))
    torch.save(means, os.path.join(output_path, 'means.pt'))

    with open(os.path.join(output_path, 'clip_values.json'), 'w') as f:
        json.dump(clip_values, f)

if __name__ == '__main__':
    main()
