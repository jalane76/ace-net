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
@click.argument('output_path', type=click.Path())
def main(x_train_filepath, y_train_filepath, clip_values_filepath, output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x_train = torch.load(x_train_filepath)
    y_train = torch.load(y_train_filepath)
    
    clip_values = {}
    with open(clip_values_filepath, 'r') as f:
        clip_values = json.load(f)
    clip_values = (clip_values.get('min_pixel_value'), clip_values.get('max_pixel_value'))

    model = MnistModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=clip_values,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10
    )

    # Train classifier
    classifier.fit(x_train, y_train, batch_size=64, nb_epochs=5)

    # Save data
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(model, os.path.join(output_path, 'mnist_art_model.pt'))
    torch.save(optimizer, os.path.join(output_path, 'mnist_art_optimizer.pt'))

if __name__ == '__main__':
    main()