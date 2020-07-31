# -*- coding: utf-8 -*-
import click
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm, trange

def plot_accuracies(df, output_path):

    norms = df['Norm'].unique()
    target_classes = df['Target Class'].unique()

    cols = ['Norm: {}'.format(col) for col in norms]
    rows = ['{}'.format(row) for row in target_classes]

    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(10, 10), sharex='col', sharey=True)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=15)
    
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, fontsize=20, verticalalignment='center')
        ax.get_yaxis().set_label_coords(-0.3, 0.5)

    for col, norm in enumerate(norms):
        for row, target in enumerate(target_classes):
            plot_df = df[(df['Norm'] == norm) & (df['Target Class'] == target)]
            
            axes[row, col].plot(plot_df['Budget'], plot_df['Accuracy'], label='Benign Accuracy')
            axes[row, col].plot(plot_df['Budget'], plot_df['Target Accuracy'], label='Target Accuracy')
            handles, labels = axes[row, col].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center')

    #fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'accuracies.png'))

def plot_class_counts(df, output_path):

    norms = df['Norm'].unique()
    target_classes = df['Target Class'].unique()

    cols = ['Norm: {}'.format(col) for col in norms]
    rows = ['{}'.format(row) for row in target_classes]

    bar_width = 0.05
    x = np.arange(len(target_classes))
    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(10, 10), sharex='col', sharey=True)

    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=15)
    
    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, rotation=0, fontsize=20, verticalalignment='center')
        ax.get_yaxis().set_label_coords(-0.3, 0.5)

    for col, norm in enumerate(norms):
        for row, target in enumerate(target_classes):
            plot_df = df[(df['Norm'] == norm) & (df['Target Class'] == target)]

            num_bars = plot_df['Budget'].nunique()
            for idx, budget in enumerate(sorted(plot_df['Budget'].unique())):
                filtered_df = plot_df[plot_df['Budget'] == budget]
                filtered_df = filtered_df.loc[:,'Class 0 Count':'Class 9 Count']
                                
                bar_loc = x - (num_bars * bar_width / 2) + idx * bar_width
                bar_vals = list(filtered_df.iloc[0])
                axes[row, col].bar(bar_loc, bar_vals, bar_width, color=list(mcolors.TABLEAU_COLORS)[idx])

            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(target_classes)

    fig.savefig(os.path.join(output_path, 'class_counts.png'))

def plot_examples(df, img_size, data_filepath, output_path):

    W, H = img_size
    norms = df['Norm'].unique()
    target_classes = df['Target Class'].unique()

    rows = ['{}'.format(row) for row in target_classes]

    for norm in norms:

        fig, axes = plt.subplots(nrows=len(rows), ncols=1, figsize=(10, 10), sharex='col', sharey=True)
        axes[0].set_title(norm, fontsize=15)
        
        for ax, row, target in zip(axes, rows, target_classes):
            ax.set_ylabel(row, rotation=0, fontsize=20, verticalalignment='center')

            plot_df = df[(df['Norm'] == norm) & (df['Target Class'] == target)]

            plot = np.zeros((W, H * plot_df['Budget'].nunique()))
            num_ticks = np.arange(plot_df['Budget'].nunique())
            for idx, budget in enumerate(sorted(plot_df['Budget'].unique())):
                path = os.path.join(data_filepath, 'attacks_target-{}_norm-{}_budget-{}.pt'.format(target, norm.strip('L_'), budget))
                img = torch.load(path)[0, :]
                img = img.reshape(W, H)
                img = np.swapaxes(img.numpy(), 0, 1).astype(np.float32)
                
                plot[:, H * idx : (H * (idx + 1))] = img

            ax.imshow(plot, aspect='equal', interpolation='nearest')
            ax.set_xticks(num_ticks * W + W / 2)
            ax.set_xticklabels('{:.2f}'.format(b) for b in sorted(plot_df['Budget'].unique()))

        fig.savefig(os.path.join(output_path, 'examples_norm-{}.png'.format(norm)))

@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(data_filepath, output_path):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    H = W = 28
    num_classes = 10
    df = pd.read_csv(os.path.join(data_filepath, 'unknown_to_known_permutations_results.csv'), float_precision='round_trip')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plot_accuracies(df, output_path)
    plot_class_counts(df, output_path)
    plot_examples(df, (W, H), data_filepath, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

if __name__ == '__main__':
    main()