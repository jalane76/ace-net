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
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_filepath', type=click.Path(exists=True))
@click.argument('y_test_adv_filepath', type=click.Path(exists=True))
@click.argument('metrics_output_path', type=click.Path())
def main(ace_filepath, interventions_filepath, x_test_filepath, y_test_filepath, x_test_adv_filepath, y_test_adv_filepath, metrics_output_path):
    ace = torch.load(ace_filepath).cpu()
    interventions = torch.load(interventions_filepath)
    x_test = torch.load(x_test_filepath)
    y_test = torch.load(y_test_filepath)
    x_test_adv = torch.load(x_test_adv_filepath)
    y_test_adv = torch.load(y_test_adv_filepath)
    
    num_classes = 10

    # Reshape images
    x_test = x_test.reshape((x_test.shape[0], 784))
    x_test_adv = x_test_adv.reshape((x_test_adv.shape[0], 784))

    # Collapse one-hot encodings
    y_test = [np.where(r == 1)[0][0] for r in y_test]
    y_test_adv = [np.where(r == 1)[0][0] for r in y_test_adv]
    
    y_predicted_adv = np.zeros_like(y_test_adv)

    for n in trange(x_test.shape[0]):
        x = x_test[n, :]
        x_adv = x_test_adv[n, :]
        
        diff = x - x_adv

        effect_map = torch.zeros_like(ace[:, :, 0])

        for idx in diff.nonzero()[0]:
            for c in range(num_classes):
                f = interpolate.interp1d(interventions, ace[idx, c, :])
                effect_map[idx, c] = torch.as_tensor(f(x_adv[idx]))
        effect_map = torch.sum(effect_map, dim=0)
        y_predicted_adv[n] = np.argmax(effect_map.numpy())

    metrics = {
        'Accuracy': str(np.sum(y_test_adv == y_predicted_adv) / len(y_test_adv))
    }

    # Save data    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()