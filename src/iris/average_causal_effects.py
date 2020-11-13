#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import click
import common.ace as ace
from common.config import load_config
from iris.models.model_from_paper import neural_network
from iris.models.model_from_paper import wrap_model_activation
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if (
        os.path.isfile(config.interventional_expectations_output_path)
        or os.path.isfile(config.average_causal_effects_output_path)
        or os.path.isfile(config.interventions_output_path)
    ):
        click.confirm(f"Overwrite files?", abort=True)

    model = torch.load(config.model_filepath)
    wrapped_model = wrap_model_activation(
        model
    )  # Iris uses a softmax activation, but I didn't want to build that assumption into the ace calculation.
    cov = torch.load(config.covariance_filepath)
    mean = torch.load(config.means_filepath)

    interventions = torch.Tensor(np.linspace(0, 1, 1000))
    ie = ace.interventional_expectation(
        wrapped_model,
        mean,
        cov,
        interventions,
        epsilon=config.epsilon,
        method=config.method,
        progress=config.show_progress,
    )
    avg_ce = ace.average_causal_effect(ie)

    # Also detach grads so they are not saved.
    ie = ie.detach()
    avg_ce = avg_ce.detach()

    torch.save(ie, config.interventional_expectations_output_path)
    torch.save(avg_ce, config.average_causal_effects_output_path)
    torch.save(torch.Tensor(interventions), config.interventions_output_path)


if __name__ == "__main__":
    main()
