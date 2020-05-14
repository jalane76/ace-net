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
@click.argument('ie_filepath', type=click.Path(exists=True))
@click.argument('interventions_filepath', type=click.Path(exists=True))
@click.argument('x_train_filepath', type=click.Path(exists=True))
@click.argument('y_train_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(ace_filepath, ie_filepath, interventions_filepath, x_train_filepath, y_train_filepath, output_path):

    H = W = 28
    ace = torch.load(ace_filepath).cpu()
    y_size = ace.size(1)
    i_size = ace.size(2)
    ace = ace.reshape(H, W, y_size, i_size)
    # swap axes so they look right
    ace = np.swapaxes(ace.numpy(), 0, 1).astype(np.float32)
    ace_val_limit = max(abs(ace.min()), ace.max())

    ie = torch.load(ie_filepath).cpu()
    ie = ie.reshape(H, W, y_size, i_size)
    # swap axes so they look right
    ie = np.swapaxes(ie.numpy(), 0, 1).astype(np.float32)
    ie_val_limit = max(abs(ie.min()), ie.max())

    interventions = torch.load(interventions_filepath).cpu()
    num_alphas = interventions.nelement()
    x_train = torch.load(x_train_filepath)
    y_train = torch.load(y_train_filepath)
    
    # collapse the one hot encodings to classes
    classes = [np.where(r == 1)[0][0] for r in y_train]
    num_classes = 10
    class_examples = np.zeros((H, W, num_classes))

    for idx, c in enumerate(classes):
        class_examples[:, :, c] += x_train[idx, 0, :, :]

    class_examples /= 6000

    # Gotta swap axes so they look right
    class_examples = np.swapaxes(class_examples, 0, 1).astype(np.float32)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Plot the ACEs
    total_iters = num_classes * num_alphas * 2
    with tqdm(total_iters) as pbar:
        fig, axes = plt.subplots(num_alphas + 1, num_classes, sharex=True, sharey=True)
        fig.set_figheight(4.0 * (num_alphas + 1))
        fig.set_figwidth(4.0 * num_classes)
        for class_index in range(num_classes):
            if class_index >= y_size:
                break
            class_axis = axes[0, class_index]
            class_axis.imshow(class_examples[:, :, class_index], aspect='equal', interpolation='nearest')
            
            for alpha_index in range(num_alphas):
                alpha_axis = axes[alpha_index + 1, class_index]
                alpha_axis.imshow(ace[:, :, class_index, alpha_index], norm=TwoSlopeNorm(vmin=-ace_val_limit , vcenter=0, vmax=ace_val_limit), cmap='bwr', aspect='equal', interpolation='nearest')
                pbar.update(1)
        plot_name = 'mnist_ace.png'
        fig.savefig(os.path.join(output_path, plot_name))

        # Plot the IEs 
        fig, axes = plt.subplots(num_alphas + 1, num_classes, sharex=True, sharey=True)
        fig.set_figheight(4.0 * (num_alphas + 1))
        fig.set_figwidth(4.0 * num_classes)
        for class_index in range(num_classes):
            if class_index >= y_size:
                break
            class_axis = axes[0, class_index]
            class_axis.imshow(class_examples[:, :, class_index], aspect='equal', interpolation='nearest')
            
            for alpha_index in range(num_alphas):
                alpha_axis = axes[alpha_index + 1, class_index]
                alpha_axis.imshow(ie[:, :, class_index, alpha_index], norm=TwoSlopeNorm(vmin=-ace_val_limit , vcenter=0, vmax=ace_val_limit), cmap='bwr', aspect='equal', interpolation='nearest')
                pbar.update(1)
        plot_name = 'mnist_ie.png'
        fig.savefig(os.path.join(output_path, plot_name))

if __name__ == '__main__':
    main()