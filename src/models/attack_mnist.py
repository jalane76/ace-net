# -*- coding: utf-8 -*-
from art.attacks import FastGradientMethod
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
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('optimizer_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_output_path', type=click.Path())
@click.argument('metrics_output_path', type=click.Path())
def main(x_test_filepath, y_test_filepath, clip_values_filepath, model_filepath, optimizer_filepath, x_test_adv_output_path, metrics_output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x_test = torch.load(x_test_filepath)
    x_test_shape = x_test.shape
    y_test = torch.load(y_test_filepath)

    # Flatten test set
    x_test = x_test.reshape(x_test.shape[0], -1)

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

    # Generate attacks
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    # Evaluate the classifier on adversarial data
    predictions = classifier.predict(x_test_adv)
    accuracy = {
        'Accuracy': np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    }

    # Reshape adversarial examples back to original test data shape
    x_test_adv = x_test_adv.reshape(x_test_shape)
    torch.save(x_test_adv, x_test_adv_output_path)
    
    with open(metrics_output_path, 'w') as f:
        json.dump(accuracy, f)

if __name__ == '__main__':
    main()