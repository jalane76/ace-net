# -*- coding: utf-8 -*-
from iris.models.model_from_paper import neural_network
from iris.models.model_from_paper import wrap_model_activation
import common.ace as ace
import click
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('covariance_filepath', type=click.Path(exists=True))
@click.argument('means_filepath', type=click.Path(exists=True))
@click.argument('interventional_expectations_output_path', type=click.Path())
@click.argument('average_causal_effects_output_path', type=click.Path())
@click.argument('interventions_output_path', type=click.Path())
def main(model_filepath, covariance_filepath, means_filepath, interventional_expectations_output_path, average_causal_effects_output_path, interventions_output_path):
    model = torch.load(model_filepath)
    wrapped_model = wrap_model_activation(model)  # Iris uses a softmax activation, but I didn't want to build that assumption into the ace calculation.
    cov = torch.load(covariance_filepath)
    mean = torch.load(means_filepath)

    interventions = torch.Tensor(np.linspace(0, 1, 1000))
    ie = ace.interventional_expectation(wrapped_model, mean, cov, interventions, epsilon=0.000001, method='hessian_diag', progress=True)
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
