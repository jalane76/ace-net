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
@click.argument('interventional_expectations_output_path', type=click.Path())
@click.argument('average_causal_effects_output_path', type=click.Path())
@click.argument('interventions_output_path', type=click.Path())
def main(model_filepath, clip_values_filepath, covariance_filepath, means_filepath, interventional_expectations_output_path, average_causal_effects_output_path, interventions_output_path):
    model = torch.load(model_filepath)
    cov = torch.load(covariance_filepath)
    mean = torch.load(means_filepath)

    # Flatten the mean
    mean = mean.reshape(1, -1)

    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))

    num_alphas = 10
    interventions = torch.Tensor(np.linspace(*clip_values, num_alphas))
    ie = ace.interventional_expectation(model, mean, cov, interventions, epsilon=0.000001, method='hessian_diag', progress=True)
    avg_ce = ace.average_causal_effect(ie)
    
    # Remove names for now since named tensors aren't serializable
    # Also detach grads so they are not saved.
    ie = ie.rename(None).detach()
    avg_ce = avg_ce.rename(None).detach()

    torch.save(ie, interventional_expectations_output_path)
    torch.save(avg_ce, average_causal_effects_output_path)
    torch.save(torch.Tensor(interventions), interventions_output_path)

if __name__ == '__main__':
    main()
