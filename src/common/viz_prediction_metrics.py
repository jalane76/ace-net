#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Sourced from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
import click
from common.config import load_config
import itertools
import json
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

    metrics = {}
    with open(config.metrics_filepath, "r") as f:
        metrics = json.load(f)

    accuracy = metrics.get("Accuracy")
    misclass = 1 - accuracy
    cm = np.array(metrics.get("Confusion Matrix"))

    cmap = plt.get_cmap("Blues")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(config.title)
    plt.colorbar()

    target_names = [str(t) for t in range(10)]
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    if config.normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if config.normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if config.normalize:
            plt.text(
                j,
                i,
                f"{cm[i, j]:0.4f}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(f"Predicted label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}")
    plt.savefig(config.output_path)


if __name__ == "__main__":
    main()
