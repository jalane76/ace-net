#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
import click
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

    if os.path.isfile(config.x_adv_output_path):
        click.confirm(f"Overwrite {config.x_adv_output_path}?", abort=True)

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x = torch.load(config.x_filepath)
    x_shape = x.shape
    y = torch.load(config.y_filepath)

    # Flatten test set
    x = x.reshape(x.shape[0], -1)

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

    # Generate attacks
    attack = FastGradientMethod(
        classifier=classifier, eps=0.2
    )  # TODO: move these parameters to config
    x_adv = attack.generate(x=x)

    # Reshape adversarial examples back to original test data shape
    x_adv = torch.from_numpy(x_adv.reshape(x_shape))
    torch.save(x_adv, config.x_adv_output_path)


if __name__ == "__main__":
    main()
