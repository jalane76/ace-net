# -*- coding: utf-8 -*-
import click
import json
import numpy as np
import os
from scipy.signal import argrelextrema
import torch
from tqdm import trange

@click.command()
@click.argument('ace_filepath', type=click.Path(exists=True))
@click.argument('metrics_output_path', type=click.Path())
def main(ace_filepath, metrics_output_path):
    ace = torch.load(ace_filepath).cpu().numpy()

    num_monotonic = 0
    num_nonmonotonic = 0
    extrema_dict = {}

    for x in range(ace.shape[0]):
        for y in range(ace.shape[1]):
            alphas = ace[x,  y, :]
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
        'Monotonic': num_monotonic,
        'Non-monotonic': num_nonmonotonic,
        'Extrema': extrema_dict
    }

    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()