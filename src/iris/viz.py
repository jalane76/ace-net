#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import click
from common.config import load_config
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.listdir(config.output_path):
        click.confirm(f"Overwrite files?", abort=True)

    ace = torch.load(config.ace_filepath).cpu()
    interventions = torch.load(config.interventions_filepath).cpu()

    feature_name = {
        0: "SepalLength",
        1: "SepalWidth",
        2: "PetalLength",
        3: "PetalWidth",
        4: "Species",
    }

    col = {0: "b", 1: "g", 2: "r", 3: "c"}

    titles = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    for output_index in range(ace.shape[1]):
        plt.figure()
        for input_index in range(ace.shape[0]):
            plt.title(titles[output_index])
            plt.xlabel("Intervention Value(alpha)")
            plt.ylabel("Causal Attributions(ACE)")

            # Baseline is np.mean(expectation_do_x)
            plt.plot(
                interventions,
                ace[input_index, output_index, :],
                label=feature_name[input_index],
                color=col[input_index],
            )
            plt.legend()
            # Plotting vertical lines to indicate regions
            if output_index == 0:
                plt.plot(
                    np.array([0.2916666567325592] * 1000),
                    np.linspace(-3, 3, 1000),
                    "--",
                )

            if output_index == 1:
                plt.plot(
                    np.array([0.2916666567325592] * 1000),
                    np.linspace(-3, 3, 1000),
                    "--",
                )
                plt.plot(
                    np.array([0.6874999403953552] * 1000),
                    np.linspace(-3, 3, 1000),
                    "--",
                )
            if output_index == 2:
                plt.plot(
                    np.array([0.6874999403953552] * 1000),
                    np.linspace(-3, 3, 1000),
                    "--",
                )
        plot_name = f"{titles[output_index]}.png"
        plt.savefig(os.path.join(config.output_path, plot_name))
        print(f"Saved {plot_name}")


if __name__ == "__main__":
    main()
