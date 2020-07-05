# -*- coding: utf-8 -*-
import click
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm, trange


@click.command()
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(x_test_filepath, y_test_filepath, x_test_adv_filepath, output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    H = W = 28
    x_test = torch.load(x_test_filepath)
    y_test = torch.load(y_test_filepath)
    x_test_adv = torch.load(x_test_adv_filepath).numpy()

    # swap axes so they look right
    x_test = np.swapaxes(x_test, 2, 3).astype(np.float32)
    x_test_adv = np.swapaxes(x_test_adv, 2, 3).astype(np.float32)
    
    num_classes = 10

# Plot the differences between test images their adversarial, evil twins
    num_samples = 10
    for sample_idx in trange(num_samples):
        fig, axes = plt.subplots(2, num_samples, sharex=True, sharey=True)
        fig.set_figheight(4.0 * 2)
        fig.set_figwidth(4.0 * num_samples)
        for sample_idx in range(num_samples):
            sample_axis = axes[0, sample_idx]
            sample_axis.imshow(x_test[sample_idx, 0, :, :], aspect='equal', interpolation='nearest')

            evil_twin_axis = axes[1, sample_idx]
            evil_twin_axis.imshow(x_test_adv[sample_idx, 0, :, :], aspect='equal', interpolation='nearest')
        fig.savefig(output_path)

if __name__ == '__main__':
    main()