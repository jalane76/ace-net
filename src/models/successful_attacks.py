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

def convert_on_hot(logits):
    result = np.zeros_like(logits)
    result[np.argmax(logits)] = 1.0
    return result

@click.command()
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('optimizer_filepath', type=click.Path(exists=True))
@click.argument('x_test_success_output_path', type=click.Path())
@click.argument('y_test_success_output_path', type=click.Path())
@click.argument('x_test_adv_success_output_path', type=click.Path())
@click.argument('y_test_adv_success_output_path', type=click.Path())
def main(
    x_test_filepath,
    y_test_filepath,
    x_test_adv_filepath,
    clip_values_filepath,
    model_filepath,
    optimizer_filepath,
    x_test_success_output_path,
    y_test_success_output_path,
    x_test_adv_success_output_path,
    y_test_adv_success_output_path
):


    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x_test = torch.load(x_test_filepath)
    y_test = torch.load(y_test_filepath)
    x_test_adv = torch.load(x_test_adv_filepath)

    # Remember shapes and flatten
    x_test_shape = x_test.shape
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test_adv_shape = x_test_adv.shape
    x_test_adv = x_test_adv.reshape(x_test_adv.shape[0], -1)

    num_classes = 10

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

    # Evaluate the classifier on benign data
    predictions = classifier.predict(x_test)
    predictions = np.apply_along_axis(convert_on_hot, 1, predictions)  # convert argmax to one-hot
    pred_bool = (predictions == y_test).all(axis=1)  # create a bool array with true flags when prediction is equal to the true class

    # Evaluate the classifier on adversarial data
    attacks = classifier.predict(x_test_adv)
    attacks = np.apply_along_axis(convert_on_hot, 1, attacks)  # convert argmax to one-hot
    attack_bool = (attacks == y_test).all(axis=1)  # create a bool array with true flags when attack prediction is equal to the true class
    
    keep_bool = np.logical_and(pred_bool, np.logical_not(attack_bool))  # create bool array with true flags when benign prediction was correct and adversarial prediction was incorrect
    
    x_test_success = x_test[keep_bool, :]
    y_test_success = y_test[keep_bool, :]
    x_test_adv_success = x_test_adv[keep_bool, :]
    y_test_adv_success = attacks[keep_bool, :]

    # Reshape data back to original
    x_test = x_test.reshape(-1, *x_test_shape[1:])
    x_test_adv = x_test_adv.reshape(-1, *x_test_adv_shape[1:])
    
    torch.save(x_test_success, x_test_success_output_path)
    torch.save(y_test_success, y_test_success_output_path)
    torch.save(x_test_adv_success, x_test_adv_success_output_path)
    torch.save(y_test_adv_success, y_test_adv_success_output_path)

if __name__ == '__main__':
    main()