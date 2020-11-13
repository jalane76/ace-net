#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
import click
from common.config import load_config
import json
import mnist.models
import numpy as np
import os
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import common.utils as utils


@click.command()
@click.argument("config_filepath", type=click.Path(exists=True))
def main(config_filepath):

    config = load_config(config_filepath)

    if os.path.isfile(config.metrics_output_path):
        click.confirm(f"Overwrite {config.metrics_output_path}?", abort=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load data
    x = torch.load(config.x_filepath)
    y = torch.load(config.y_filepath)

    # Flatten
    x = x.reshape(x.shape[0], -1)

    model = torch.load(config.trained_model_filepath)

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

    # Evaluate the classifier on benign data
    predictions = classifier.predict(x)

    # Convert one-hots to numbers for metrics
    y = utils.one_hot_to_num(y)
    predictions = utils.one_hot_to_num(predictions)
    accuracy = {
        "Accuracy": metrics.accuracy_score(y, predictions),
        "Confusion Matrix": metrics.confusion_matrix(y, predictions).tolist(),
    }

    # Save data
    with open(config.metrics_output_path, "w") as f:
        json.dump(
            accuracy,
            f,
            ensure_ascii=False,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


if __name__ == "__main__":
    main()
