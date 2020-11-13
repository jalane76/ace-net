#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import click
from common.config import load_config
import json
import numpy as np
import os
from scipy.signal import argrelextrema
import torch
from tqdm import trange


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.path.isfile(config.metrics_output_path):
        click.confirm(f"Overwrite {config.metrics_output_path}?", abort=True)

    ace = torch.load(config.ace_filepath).cpu().numpy()

    num_monotonic = 0
    num_nonmonotonic = 0
    extrema_dict = {}

    for x in range(ace.shape[0]):
        for y in range(ace.shape[1]):
            alphas = ace[x, y, :]
            diff = np.diff(alphas)
            if np.all(diff >= 0) or np.all(diff <= 0):
                num_monotonic += 1
            else:
                num_nonmonotonic += 1
                maxima = argrelextrema(alphas, np.greater)[0]
                minima = argrelextrema(alphas, np.less)[0]
                extrema_count = len(maxima) + len(minima)
                if extrema_count in extrema_dict.keys():
                    extrema_dict[extrema_count] += 1
                else:
                    extrema_dict[extrema_count] = 1

    metrics = {
        "Monotonic": num_monotonic,
        "Non-monotonic": num_nonmonotonic,
        "Extrema": extrema_dict,
    }

    with open(config.metrics_output_path, "w") as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


if __name__ == "__main__":
    main()
