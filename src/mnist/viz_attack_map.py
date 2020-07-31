# -*- coding: utf-8 -*-
import click
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm, trange


@click.command()
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_filepath', type=click.Path(exists=True))
@click.argument('y_test_adv_filepath', type=click.Path(exists=True))
@click.argument('x_test_map_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_map_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(x_test_filepath, y_test_filepath, x_test_adv_filepath, y_test_adv_filepath, x_test_map_filepath, x_test_adv_map_filepath, output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    H = W = 28
    num_classes = 10

    x_test = torch.load(x_test_filepath)
    y_test = torch.load(y_test_filepath)
    x_test_adv = torch.load(x_test_adv_filepath)
    y_test_adv = torch.load(y_test_adv_filepath)
    x_test_map = torch.load(x_test_map_filepath)
    x_test_adv_map = torch.load(x_test_adv_map_filepath)

    x_test = x_test.reshape(x_test.shape[0], H, W)
    x_test_adv = x_test_adv.reshape(x_test_adv.shape[0], H, W)
    x_test_map = x_test_map.reshape(x_test_map.shape[0], H, W, num_classes).numpy()
    x_test_adv_map = x_test_adv_map.reshape(x_test_adv_map.shape[0], H, W, num_classes).numpy()

    # swap axes so they look right
    x_test = np.swapaxes(x_test, 1, 2)
    x_test_adv = np.swapaxes(x_test_adv, 1, 2)
    x_test_map = np.swapaxes(x_test_map, 1, 2)
    x_test_adv_map = np.swapaxes(x_test_adv_map, 1, 2)

    # collapse one-hot encodings 
    y_test = [np.where(r == 1)[0][0] for r in y_test]
    y_test_adv = [np.where(r == 1)[0][0] for r in y_test_adv]

    # Plot an input sample, it's adversary, and their respective causal effects maps
    plot_index = 5
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.set_figheight(4.0 * 2)
    fig.set_figwidth(4.0 * 3)
    
    sample_axis = axes[0, 0]
    sample_axis.imshow(x_test[plot_index, :, :], aspect='equal', interpolation='nearest')
    sample_axis.title.set_text('Actual Class: {}'.format(y_test[plot_index]))

    evil_twin_axis = axes[0, 1]
    evil_twin_axis.imshow(x_test_adv[plot_index, :, :], aspect='equal', interpolation='nearest')
    evil_twin_axis.title.set_text('Adversarial Class: {}'.format(y_test_adv[plot_index]))

    diff_axis = axes[0, 2]
    diff_image = x_test_adv[plot_index, :, :] - x_test[plot_index, :, :]
    diff_axis.imshow(diff_image, aspect='equal', interpolation='nearest')
    diff_axis.title.set_text('Difference')
    
    sample_map_axis = axes[1, 0]
    sample_map_axis.imshow(x_test_map[plot_index, :, :, y_test[plot_index]], norm=TwoSlopeNorm(vmin=-1.0 , vcenter=0, vmax=1.0), cmap='bwr', aspect='equal', interpolation='nearest')
    sample_map_axis.title.set_text('Causal Effect Map')

    evil_twin_map_axis = axes[1, 1]
    evil_twin_map_axis.imshow(x_test_adv_map[plot_index, :, :, y_test_adv[plot_index]], norm=TwoSlopeNorm(vmin=-1.0 , vcenter=0, vmax=1.0), cmap='bwr', aspect='equal', interpolation='nearest')
    evil_twin_map_axis.title.set_text('Adversarial Causal Effect Map')

    diff_map_axis = axes[1, 2]
    diff_map_image = x_test_adv_map[plot_index, :, :, y_test_adv[plot_index]] - x_test_map[plot_index, :, :, y_test[plot_index]]
    diff_map_axis.imshow(diff_map_image, norm=TwoSlopeNorm(vmin=-1.0 , vcenter=0, vmax=1.0), cmap='bwr', aspect='equal', interpolation='nearest')
    diff_map_axis.title.set_text('Difference Causal Effect Map')
    
    fig.savefig(output_path)

if __name__ == '__main__':
    main()