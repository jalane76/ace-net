# -*- coding: utf-8 -*-
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
import click
import json
import mnist.models
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
@click.argument('x_adv_output_path', type=click.Path())
def main(x_filepath, y_filepath, clip_values_filepath, model_filepath, x_adv_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x = torch.load(x_filepath)
    x_shape = x.shape
    y = torch.load(y_filepath)

    # Flatten test set
    x = x.reshape(x.shape[0], -1)

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

    # Generate attacks
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    x_adv = attack.generate(x=x)

    # Reshape adversarial examples back to original test data shape
    x_adv = x_adv.reshape(x_shape)
    torch.save(x_adv, x_adv_output_path)

if __name__ == '__main__':
    main()