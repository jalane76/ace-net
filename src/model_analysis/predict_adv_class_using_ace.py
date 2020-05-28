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

    # Interpolation function storage
    interp_funcs = np.empty((x_test.shape[0], x_test.shape[1]), dtype=np.object)

    for n in trange(x_test.shape[0]):
    #for n in trange(1):
        x = x_test[n, :]
        x_adv = x_test_adv[n, :]
        
        #diff = x - x_adv

        effect_map = torch.zeros_like(ace[:, :, 0])

        #for idx in diff.nonzero()[0]:
        for idx in range(x_adv.shape[0]):
            for c in range(num_classes):
                if not interp_funcs[idx, c]:
                    interp_funcs[idx, c] = interpolate.interp1d(interventions, ace[idx, c, :])
                f = interp_funcs[idx, c]
                effect_map[idx, c] = torch.as_tensor(f(x_adv[idx]))
        effect_map = torch.sum(effect_map, dim=0)
        y_predicted_adv[n] = np.argmax(effect_map.numpy())

    per_class_accuracy = {}
    for c in range(num_classes):
        per_class_accuracy[c] = {
            'Correct': 0,
            'Total': 0,
            'ClassBin': [0] * num_classes
        }
    for y, pred_y in zip(y_test_adv, y_predicted_adv):
        correct = 0
        if y == pred_y:
            correct = 1
        per_class_accuracy[y]['Correct'] += correct
        per_class_accuracy[y]['Total'] += 1
        per_class_accuracy[y]['ClassBin'][pred_y] += 1
    
    for key, value in per_class_accuracy.items():
        value['Accuracy'] = value['Correct'] / value['Total']

    metrics = {
        'Total Accuracy': str(np.sum(y_test_adv == y_predicted_adv) / len(y_test_adv)),
        'Per Class Accuracy': per_class_accuracy
    }

    # Save data    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()