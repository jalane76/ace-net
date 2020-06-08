# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
import click
import json
from models.mnist_model import MnistModel
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

@click.command()
@click.argument('x_train_filepath', type=click.Path(exists=True))
@click.argument('y_train_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_output_path', type=click.Path())
@click.argument('optimizer_output_path', type=click.Path())
def main(x_train_filepath, y_train_filepath, clip_values_filepath, model_output_path, optimizer_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x_train = torch.load(x_train_filepath)
    y_train = torch.load(y_train_filepath)

    # Flatten training set
    x_train = x_train.reshape(x_train.shape[0], -1)
    
    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))

    model = MnistModel()

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(784),
        nb_classes=10
    )

    # Train classifier
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=15)

    # Save data
    torch.save(model, model_output_path)
    torch.save(optimizer, optimizer_output_path)

if __name__ == '__main__':
    main()