# -*- coding: utf-8 -*-
import click
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(input_filepath, output_path):

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

    # Save data
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(x_tensor, os.path.join(output_path, 'x_train.pt'))
    torch.save(y_tensor, os.path.join(output_path, 'y_train.pt'))

if __name__ == '__main__':
    main()
