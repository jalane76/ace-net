# -*- coding: utf-8 -*-
import click
import json
import numpy as np
import os
from scipy import interpolate
import torch
from tqdm import trange

@click.command()
@click.argument('ace_filepath', type=click.Path(exists=True))
@click.argument('interventions_filepath', type=click.Path(exists=True))
@click.argument('x_filepath', type=click.Path(exists=True))
@click.argument('effects_map_output_path', type=click.Path())
def main(ace_filepath, interventions_filepath, x_filepath, effects_map_output_path):
    ace = torch.load(ace_filepath).cpu()
    interventions = torch.load(interventions_filepath)
    X = torch.load(x_filepath)
    
    num_classes = 10

    # Reshape images
    X = X.reshape((X.shape[0], 784))

    # Interpolation function storage
    interp_funcs = np.empty((X.shape[0], X.shape[1]), dtype=np.object)
    effects_map = torch.zeros((X.shape[0], X.shape[1], num_classes))

    for n in trange(X.shape[0]):
    #for n in trange(1):
        x = X[n, :]

        #for idx in diff.nonzero()[0]:
        for idx in range(x.shape[0]):
            for c in range(num_classes):
                if not interp_funcs[idx, c]:
                    interp_funcs[idx, c] = interpolate.interp1d(interventions, ace[idx, c, :])
                f = interp_funcs[idx, c]
                effects_map[n, idx, c] = torch.as_tensor(f(x[idx]))
   
    # Save data
    torch.save(effects_map, effects_map_output_path)

if __name__ == '__main__':
    main()