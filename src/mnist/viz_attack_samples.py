#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import click
from common.config import load_config
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm, trange


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.path.isfile(config.output_path):
        click.confirm(f"Overwrite {config.output_path}?", abort=True)

    W = config.image_width
    H = config.image_height
    x = torch.load(config.x_filepath)
    y = torch.load(config.y_filepath)
    x_adv = torch.load(config.x_adv_filepath).numpy()

    # swap axes so they look right
    x = np.swapaxes(x, 2, 3).astype(np.float32)
    x_adv = np.swapaxes(x_adv, 2, 3).astype(np.float32)

    num_classes = config.num_classes

    # Plot the differences between test images their adversarial, evil twins
    num_samples = config.num_samples
    for sample_idx in trange(num_samples):
        fig, axes = plt.subplots(2, num_samples, sharex=True, sharey=True)
        fig.set_figheight(4.0 * 2)
        fig.set_figwidth(4.0 * num_samples)
        for sample_idx in range(num_samples):
            sample_axis = axes[0, sample_idx]
            sample_axis.imshow(
                x[sample_idx, 0, :, :], aspect="equal", interpolation="nearest"
            )

            evil_twin_axis = axes[1, sample_idx]
            evil_twin_axis.imshow(
                x_adv[sample_idx, 0, :, :], aspect="equal", interpolation="nearest"
            )
        fig.savefig(config.output_path)


if __name__ == "__main__":
    main()
