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
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('x_test_adv_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(ace_filepath, ie_filepath, interventions_filepath, x_train_filepath, y_train_filepath, x_test_filepath, y_test_filepath, x_test_adv_filepath, output_path):

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

    ie = torch.load(ie_filepath).cpu()
    ie = ie.reshape(H, W, y_size, i_size)
    # swap axes so they look right
    ie = np.swapaxes(ie.numpy(), 0, 1).astype(np.float32)
    ie_val_limit = max(abs(ie.min()), ie.max())

    interventions = torch.load(interventions_filepath).cpu()
    num_alphas = interventions.nelement()
    x_train = torch.load(x_train_filepath)
    y_train = torch.load(y_train_filepath)
    x_test = torch.load(x_test_filepath)
    y_test = torch.load(y_test_filepath)
    x_test_adv = torch.load(x_test_adv_filepath)

    # swap axes so they look right
    x_test = np.swapaxes(x_test, 2, 3).astype(np.float32)
    x_test_adv = np.swapaxes(x_test_adv, 2, 3).astype(np.float32)
    
    num_classes = 10

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

    # Gotta swap axes so they look right
    train_class_examples = np.swapaxes(train_class_examples, 0, 1).astype(np.float32)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Plot the ACEs
    total_iters = num_classes * num_alphas * 2
    with tqdm(total_iters, desc='ACE and IE') as pbar:
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
            class_axis.imshow(train_class_examples[:, :, class_index], aspect='equal', interpolation='nearest')
            
            for alpha_index in range(num_alphas):
                alpha_axis = axes[alpha_index + 1, class_index]
                alpha_axis.imshow(ie[:, :, class_index, alpha_index], norm=TwoSlopeNorm(vmin=-ace_val_limit , vcenter=0, vmax=ace_val_limit), cmap='bwr', aspect='equal', interpolation='nearest')
                pbar.update(1)
        plot_name = 'mnist_ie.png'
        fig.savefig(os.path.join(output_path, plot_name))


# Plot the differences between test images their adversarial, evil twins
    total_iters = num_classes * 3
    with tqdm(total_iters, desc='Diff samples') as pbar:
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
        plot_name = 'attack_diff.png'
        fig.savefig(os.path.join(output_path, plot_name))

if __name__ == '__main__':
    main()