# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
from common.ace_attack import ace_attack
import click
import json
import mnist,models
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

@click.command()
@click.argument('x_filepath', type=click.Path(exists=True))
@click.argument('y_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('ace_filepath', type=click.Path(exists=True))
@click.argument('interventions_filepath', type=click.Path(exists=True))
@click.argument('attacks_output_filepath', type=click.Path())
@click.argument('metrics_output_path', type=click.Path())
def main(x_filepath, y_filepath, clip_values_filepath, model_filepath, ace_filepath, interventions_filepath, attacks_output_filepath, metrics_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x = torch.load(x_filepath)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    y = torch.load(y_filepath)
    ace = torch.load(ace_filepath).cpu()
    interventions = torch.load(interventions_filepath).cpu()

    model = torch.load(model_filepath)

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

    # Target the attacks to a particular class
    y_adv = torch.zeros_like(torch.from_numpy(y))
    y_adv[:, 0] = 1.0

    # Generate attacks
    x_adv = ace_attack(ace, interventions, torch.from_numpy(x), target_classes=y_adv, norm=2, budget=5.0)
    y_adv = y_adv.numpy()

    # Evaluate the classifier on adversarial data
    predictions = classifier.predict(x_adv)

    count_dict = {}
    for c in range(9):
        count = np.sum(np.argmax(predictions, axis=1) == c)
        count_dict[str(c)] = str(count)

    accuracy = {
        'Accuracy': np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1)) / len(y),
        'Target Accuracy': np.sum(np.argmax(predictions, axis=1) == np.argmax(y_adv, axis=1)) / len(y_adv),
        'Class Counts': count_dict
    }

    # Unflatten for saving
    x_adv = x_adv.reshape(x_shape)
    torch.save(x_adv, attacks_output_filepath)
    
    with open(metrics_output_path, 'w') as f:
        json.dump(accuracy, f)

if __name__ == '__main__':
    main()