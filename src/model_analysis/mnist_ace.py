# -*- coding: utf-8 -*-
import click
import json
from models.mnist_model import MnistModel
import model_analysis.ace as ace
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('covariance_filepath', type=click.Path(exists=True))
@click.argument('means_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(model_filepath, clip_values_filepath, covariance_filepath, means_filepath, output_path):
    model = torch.load(model_filepath)
    cov = torch.load(covariance_filepath)
    mean = torch.load(means_filepath)

    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))

    num_alphas = 3
    interventions = torch.Tensor(np.linspace(*clip_values, num_alphas))
    ie = ace.interventional_expectation(model, mean, cov, interventions, epsilon=0.000001, method='hessian_diag', progress=True)
    avg_ce = ace.average_causal_effect(ie)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Remove names for now since named tensors aren't serializable (maybe named tensors are too much trouble)
    # Also detach grads so they are not saved.
    ie = ie.rename(None).detach()
    avg_ce = avg_ce.rename(None).detach()

    torch.save(ie, os.path.join(output_path, 'interventional_expectations.pt'))
    torch.save(avg_ce, os.path.join(output_path, 'average_causal_effects.pt'))
    torch.save(torch.Tensor(interventions), os.path.join(output_path, 'interventions.pt'))

if __name__ == '__main__':
    main()
