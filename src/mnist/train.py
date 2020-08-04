# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
import click
import inspect
import json
import mnist.models
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

@click.command()
@click.argument('x_filepath', type=click.Path(exists=True))
@click.argument('y_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_class_name')
@click.argument('model_output_path', type=click.Path())
@click.option('--batch_size', default=64, show_default=True)
@click.option('--num_epochs', default=15, show_default=True)
def main(x_filepath, y_filepath, clip_values_filepath, model_class_name, model_output_path, batch_size, num_epochs):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x = torch.load(x_filepath)
    y = torch.load(y_filepath)

    # Flatten training set
    x = x.reshape(x.shape[0], -1)
    
    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))

    model = None
    for name, cls in inspect.getmembers(mnist.models):
        if name == '__builtins__':
            continue
        print('{}: {}'.format(name, cls))
        if name == model_class_name:
            model = cls()
            break

    if not model:
        sys.exit("Could not load provided model {}".format(model_class_name))

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=model.criterion,
        optimizer=model.optimizer,
        input_shape=(784),
        nb_classes=10
    )

    # Train classifier
    classifier.fit(x, y, batch_size=batch_size, nb_epochs=num_epochs)

    # Save data
    torch.save(model, model_output_path)

if __name__ == '__main__':
    main()