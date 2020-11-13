#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
from common.ace_attack import ace_attack
from common.config import load_config
import click
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

    if os.path.isfile(config.x_adv_output_path):
        click.confirm(f"Overwrite {config.x_adv_output_path}?", abort=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load data
    x = torch.load(config.x_filepath)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    y = torch.load(config.y_filepath)
    ace = torch.load(config.ace_filepath).cpu()
    interventions = torch.load(config.interventions_filepath).cpu()

    model = torch.load(config.model_filepath)

    clip_values = {}
    with open(config.clip_values_filepath, "r") as f:
        clip_values = json.load(f)
    clip_values = (
        clip_values.get("min_pixel_value"),
        clip_values.get("max_pixel_value"),
    )

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=model.criterion,
        optimizer=model.optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )  # TODO: move these parameters to config

    # Target the attacks to a particular class
    y_adv = torch.zeros_like(torch.from_numpy(y))
    y_adv[:, 0] = 1.0

    # Generate attacks
    x_adv = ace_attack(
        ace,
        interventions,
        torch.from_numpy(x),
        target_classes=y_adv,
        norm=2,
        budget=5.0,
    )  # TODO: move these parameters to config

    # Unflatten for saving
    x_adv = x_adv.reshape(x_shape)
    torch.save(x_adv, config.x_adv_output_path)


if __name__ == "__main__":
    main()
