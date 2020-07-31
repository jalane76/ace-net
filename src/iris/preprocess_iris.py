# -*- coding: utf-8 -*-
import click
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('x_train_output_path', type=click.Path())
@click.argument('y_train_output_path', type=click.Path())
@click.argument('covariance_output_path', type=click.Path())
@click.argument('means_output_path', type=click.Path())
def main(input_filepath, x_train_output_path, y_train_output_path, covariance_output_path, means_output_path):

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = pd.read_csv(input_filepath)
    dataset = pd.get_dummies(dataset, columns=['species']) # One Hot Encoding
    values = list(dataset.columns.values)

    y = dataset[values[-3:]]
    y = np.array(y, dtype='float32')
    x = dataset[values[:-3]]
    x = np.array(x, dtype='float32')

    # Shuffle Data
    indices = np.random.choice(len(x), len(x), replace=False)
    x_values = x[indices]

    scaler = MinMaxScaler()

    test_size = 30
    x_train = x_values[:-test_size]
    x_train = scaler.fit_transform(x_train)

    x_values = scaler.transform(x_values)
    y_values = y[indices]

    x_tensor = torch.Tensor(x_values)
    y_tensor = torch.Tensor(y_values)

    # Calculate observational statistics
    covariance = torch.Tensor(np.cov(x_values, rowvar=False))
    means = torch.Tensor(np.mean(x_values, axis=0))

    # Save data
    torch.save(x_tensor, x_train_output_path)
    torch.save(y_tensor, y_train_output_path)
    torch.save(covariance, covariance_output_path)
    torch.save(means, means_output_path)

if __name__ == '__main__':
    main()
