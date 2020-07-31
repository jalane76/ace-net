# -*- coding: utf-8 -*-
import click
import json
import numpy as np
import os
import torch

@click.command()
@click.argument('ace_filepath', type=click.Path(exists=True))
@click.argument('interventions_filepath', type=click.Path(exists=True))
@click.argument('max_ace_output_path', type=click.Path())
def main(ace_filepath, interventions_filepath, max_ace_output_path):
    ace = torch.load(ace_filepath).cpu()
    interventions = torch.load(interventions_filepath)

    max_ace = torch.zeros((ace.shape[0], ace.shape[1]))
    for x in range(ace.shape[0]):
        for y in range(ace.shape[1]):
            max_idx = np.argmax(ace[x, y, :])
            max_ace[x, y] = interventions[max_idx]
    print(max_ace)
    torch.save(max_ace, max_ace_output_path)

if __name__ == '__main__':
    main()
