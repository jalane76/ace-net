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
@click.argument('output_path', type=click.Path())
def main(x_test_filepath, y_test_filepath, x_test_adv_filepath, output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    H = W = 28
    x_test = torch.load(x_test_filepath)
    y_test = torch.load(y_test_filepath)
    x_test_adv = torch.load(x_test_adv_filepath)

    # swap axes so they look right
    x_test = np.swapaxes(x_test, 2, 3).astype(np.float32)
    x_test_adv = np.swapaxes(x_test_adv, 2, 3).astype(np.float32)
    
    num_classes = 10

    # Get the averages of the testing samples
    test_classes = [np.where(r == 1)[0][0] for r in y_test]
    test_class_examples = np.zeros((H, W, num_classes))

    test_class_counts = np.zeros(num_classes)
    for idx, c in enumerate(test_classes):
        test_class_examples[:, :, c] += x_test[idx, 0, :, :]
        test_class_counts[c] += 1
    for idx in range(num_classes):
        test_class_examples[:, :, idx] /= test_class_counts[idx]
        
    # Get the averages of the adversarial samples
    test_adv_class_examples = np.zeros((H, W, num_classes))

    test_adv_class_counts = np.zeros(num_classes)
    for idx, c in enumerate(test_classes):
        test_adv_class_examples[:, :, c] += x_test_adv[idx, 0, :, :]
        test_adv_class_counts[c] += 1
    for idx in range(num_classes):
        test_adv_class_examples[:, :, idx] /= test_adv_class_counts[idx]

    # Get the averages of the differences
    x_diff = x_test_adv - x_test
    diff_examples = np.zeros((H, W, num_classes))
    diff_counts = np.zeros(num_classes)
    for idx, c in enumerate(test_classes):
        diff_examples[:, :, c] += x_diff[idx, 0, :, :]
        diff_counts[c] += 1
    for idx in range(num_classes):
        diff_examples[:, :, idx] /= diff_counts[idx]

# Plot the differences between test images their adversarial, evil twins
    total_iters = num_classes * 3
    with tqdm(total_iters) as pbar:
        fig, axes = plt.subplots(3, num_classes, sharex=True, sharey=True)
        fig.set_figheight(4.0 * 3)
        fig.set_figwidth(4.0 * num_classes)
        for col_idx in range(num_classes):
            sample_axis = axes[0, col_idx]
            sample_axis.imshow(test_class_examples[:, :, col_idx], aspect='equal', interpolation='nearest')

            evil_twin_axis = axes[1, col_idx]
            evil_twin_axis.imshow(test_adv_class_examples[:, :, col_idx], aspect='equal', interpolation='nearest')
            
            diff_axis = axes[2, col_idx]
            diff_axis.imshow(diff_examples[:, :, col_idx], norm=TwoSlopeNorm(vmin=-1.0 , vcenter=0, vmax=1.0), cmap='bwr', aspect='equal', interpolation='nearest')
            
            pbar.update(1)
        fig.savefig(output_path)

if __name__ == '__main__':
    main()