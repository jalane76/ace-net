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

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    H = W = 28
    inter_map = torch.load(config.inter_map_filepath).cpu()
    y_size = inter_map.size(1)
    i_size = inter_map.size(2)
    inter_map = inter_map.reshape(H, W, y_size, i_size)

    # swap axes so they look right
    inter_map = np.swapaxes(inter_map.numpy(), 0, 1).astype(np.float32)
    inter_map_val_limit = max(abs(inter_map.min()), inter_map.max())

    interventions = torch.load(config.interventions_filepath).cpu()
    num_alphas = interventions.nelement()
    num_classes = 10

    x_train = torch.load(config.x_train_filepath)
    y_train = torch.load(config.y_train_filepath)

    # Get the averages of the training samples
    # collapse the one hot encodings to classes
    train_classes = [np.where(r == 1)[0][0] for r in y_train]
    train_class_examples = np.zeros((H, W, num_classes))

    train_class_counts = np.zeros(num_classes)
    for idx, c in enumerate(train_classes):
        train_class_examples[:, :, c] += x_train[idx, 0, :, :]
        train_class_counts[c] += 1
    for idx in range(num_classes):
        train_class_examples[:, :, idx] /= train_class_counts[idx]

    # Gotta swap axes so they look right
    train_class_examples = np.swapaxes(train_class_examples, 0, 1).astype(np.float32)

    rows = [f"\u03B1 = {alpha:.2f}" for alpha in interventions]
    cols = [f"y = {c}" for c in range(10)]

    # Plot the interventional map
    total_iters = num_classes * num_alphas
    with tqdm(total_iters) as pbar:
        fig, axes = plt.subplots(num_alphas + 1, num_classes, sharex=True, sharey=True)
        fig.set_figheight(4.0 * (num_alphas + 1))
        fig.set_figwidth(4.0 * num_classes)

        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=60, pad=20)

        for ax, row in zip(axes[1:, 0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=60, verticalalignment="center")
            ax.get_yaxis().set_label_coords(-0.8, 0.5)

        for class_index in range(num_classes):
            if class_index >= y_size:
                break
            class_axis = axes[0, class_index]
            class_axis.imshow(
                train_class_examples[:, :, class_index],
                aspect="equal",
                interpolation="nearest",
            )

            for alpha_index in range(num_alphas):
                alpha_axis = axes[alpha_index + 1, class_index]
                alpha_axis.imshow(
                    inter_map[:, :, class_index, alpha_index],
                    norm=TwoSlopeNorm(
                        vmin=-inter_map_val_limit, vcenter=0, vmax=inter_map_val_limit
                    ),
                    cmap="bwr",
                    aspect="equal",
                    interpolation="nearest",
                )
                pbar.update(1)
        fig.savefig(config.output_path)


if __name__ == "__main__":
    main()
