# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
import click
import json
import mnist.models
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import get_model_from_module

@click.command()
@click.argument('x_filepath', type=click.Path(exists=True))
@click.argument('y_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('trained_model_filepath', type=click.Path(exists=True))
@click.argument('metrics_output_path', type=click.Path())
def main(x_filepath, y_filepath, clip_values_filepath, trained_model_filepath, metrics_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x = torch.load(x_filepath)
    y = torch.load(y_filepath)

    # Flatten
    x = x.reshape(x.shape[0], -1)

    model = torch.load(trained_model_filepath)

    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=model.criterion,
        optimizer=model.optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    # Evaluate the classifier on benign data
    predictions = classifier.predict(x)
    accuracy = {
        'Accuracy': np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    }

    # Save data    
    with open(metrics_output_path, 'w') as f:
        json.dump(accuracy, f)

if __name__ == '__main__':
    main()