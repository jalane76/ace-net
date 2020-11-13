#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import click
import common.ace as ace
from common.config import load_config
import json
import mnist.models
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.path.isfile(config.interventional_expectations_output_path):
        click.confirm(
            f"Overwrite {config.interventional_expectations_output_path}?", abort=True
        )
    if os.path.isfile(config.average_causal_effects_output_path):
        click.confirm(
            f"Overwrite {config.average_causal_effects_output_path}?", abort=True
        )
    if os.path.isfile(config.interventions_output_path):
        click.confirm(f"Overwrite {config.interventions_output_path}?", abort=True)

    model = torch.load(config.model_filepath)
    cov = torch.load(config.covariance_filepath)
    mean = torch.load(config.means_filepath)

    # Flatten the mean
    mean = mean.reshape(1, -1)

    clip_values = {}
    with open(config.clip_values_filepath, "r") as f:
        clip_values = json.load(f)
    clip_values = (
        clip_values.get("min_pixel_value"),
        clip_values.get("max_pixel_value"),
    )

    num_alphas = 10
    interventions = torch.Tensor(np.linspace(*clip_values, num_alphas))
    ie = ace.interventional_expectation(
        model,
        mean,
        cov,
        interventions,
        epsilon=0.000001,
        method="hessian_diag",
        progress=True,
    )
    avg_ce = ace.average_causal_effect(ie)

    # Remove names for now since named tensors aren't serializable
    # Also detach grads so they are not saved.
    ie = ie.rename(None).detach()
    avg_ce = avg_ce.rename(None).detach()

    torch.save(ie, config.interventional_expectations_output_path)
    torch.save(avg_ce, config.average_causal_effects_output_path)
    torch.save(torch.Tensor(interventions), config.interventions_output_path)


if __name__ == "__main__":
    main()
