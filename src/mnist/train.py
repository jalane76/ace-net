#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
import click
from common.config import load_config
import json
import logging
import mnist.models
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from common.utils import get_model_from_module


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.path.isfile(config.model_output_path):
        click.confirm(f"Overwrite {config.model_output_path}?", abort=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load data
    x = torch.load(config.x_filepath)
    y = torch.load(config.y_filepath)

    # Flatten training set
    x = x.reshape(x.shape[0], -1)

    clip_values = {}
    with open(config.clip_values_filepath, "r") as f:
        clip_values = json.load(f)
    clip_values = (
        clip_values.get("min_pixel_value"),
        clip_values.get("max_pixel_value"),
    )

    model = get_model_from_module(mnist.models, config.model_class_name)

    if not model:
        sys.exit(f"Could not load provided model {config.model_class_name}")

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=model.criterion,
        optimizer=model.optimizer,
        input_shape=(784),
        nb_classes=10,
    )  # TODO: move these parameters to config

    # Train classifier
    classifier.fit(x, y, batch_size=config.batch_size, nb_epochs=config.num_epochs)

    # Save data
    torch.save(model, config.model_output_path)


if __name__ == "__main__":
    main()
