# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
import click
import json
from models.mnist_model import MnistModel
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

@click.command()
@click.argument('max_ace_filepath', type=click.Path(exists=True))
@click.argument('means_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('optimizer_filepath', type=click.Path(exists=True))
@click.argument('metrics_output_path', type=click.Path())
def main(max_ace_filepath, means_filepath, clip_values_filepath, model_filepath, optimizer_filepath, metrics_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    max_ace = torch.load(max_ace_filepath)
    means = torch.load(means_filepath)

    # Reshape max ACE as input to the model
    max_ace = max_ace.permute(1, 0)
    max_ace = max_ace.unsqueeze(1)
    max_ace = max_ace.reshape(-1, 1, 28, 28)

    max_avg_ace = max_ace + means

    model = torch.load(model_filepath)
    optimizer = torch.load(optimizer_filepath)

    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))
    
    criterion = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    # Evaluate the classifier on the max ACE inputs
    max_predictions = classifier.predict(max_ace)

    max_dict = {}
    for idx in range(max_predictions.shape[0]):
        c = np.argmax(max_predictions[idx])
        max_dict[idx] = '{} == {}: {}'.format(idx, c, str(idx == c))

    # Evaluate the classifier on the max ACE inputs + the training mean
    max_avg_predictions = classifier.predict(max_avg_ace)
    
    max_avg_dict = {}
    for idx in range(max_avg_predictions.shape[0]):
        c = np.argmax(max_avg_predictions[idx])
        max_avg_dict[idx] = '{} == {}: {}'.format(idx, c, str(idx == c))

    metrics = {
        'Max ACE': max_dict,
        'Max Avg ACE': max_avg_dict
    }
    
    # Save data    
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()