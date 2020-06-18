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
@click.argument('ace_filepath', type=click.Path(exists=True))
@click.argument('interventions_filepath', type=click.Path(exists=True))
@click.argument('x_train_filepath', type=click.Path(exists=True))
@click.argument('y_train_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(ace_filepath, interventions_filepath, x_train_filepath, y_train_filepath, output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    H = W = 28
    ace = torch.load(ace_filepath).cpu()
    y_size = ace.size(1)
    i_size = ace.size(2)
    ace = ace.reshape(H, W, y_size, i_size)

    # swap axes so they look right
    ace = np.swapaxes(ace.numpy(), 0, 1).astype(np.float32)
    ace_val_limit = max(abs(ace.min()), ace.max())

    interventions = torch.load(interventions_filepath).cpu()
    num_alphas = interventions.nelement()
    num_classes = 10

    x_train = torch.load(x_train_filepath)
    y_train = torch.load(y_train_filepath)

    # Get the averages of the training samples
    # collapse the one hot encodings to classes
    train_classes = [np.where(r == 1)[0][0] for r in y_train]
    train_class_examples = np.zeros((H, W, num_classes))

    train_class_counts = np.zeros(num_classes)
    for idx, c in enumerate(train_classes):
        train_class_examples[:, :, c] += x_train[idx, 0, :, :]
        train_class_counts[c] += 1
    for idx in range(num_classes):
        train_class_examples[:, :, idx] /= train_class_counts[idx]

    # Gotta swap axes so they look right
    train_class_examples = np.swapaxes(train_class_examples, 0, 1).astype(np.float32)

    # Plot the ACEs
    total_iters = num_classes * num_alphas
    with tqdm(total_iters) as pbar:
        fig, axes = plt.subplots(num_alphas + 1, num_classes, sharex=True, sharey=True)
        fig.set_figheight(4.0 * (num_alphas + 1))
        fig.set_figwidth(4.0 * num_classes)
        for class_index in range(num_classes):
            if class_index >= y_size:
                break
            class_axis = axes[0, class_index]
            class_axis.imshow(train_class_examples[:, :, class_index], aspect='equal', interpolation='nearest')
            
            for alpha_index in range(num_alphas):
                alpha_axis = axes[alpha_index + 1, class_index]
                alpha_axis.imshow(ace[:, :, class_index, alpha_index], norm=TwoSlopeNorm(vmin=-ace_val_limit , vcenter=0, vmax=ace_val_limit), cmap='bwr', aspect='equal', interpolation='nearest')
                pbar.update(1)
        fig.savefig(output_path)

if __name__ == '__main__':
    main()