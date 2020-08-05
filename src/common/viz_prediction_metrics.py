# Sourced from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

# -*- coding: utf-8 -*-
import click
import itertools
import json
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm, trange


@click.command()
@click.argument('metrics_filepath', type=click.Path(exists=True))
@click.option('--title', default="Confusion Matrix")
@click.option('--normalize', default=False)
@click.argument('output_path', type=click.Path())
def main(metrics_filepath, title, normalize, output_path):

    metrics = {}
    with open(metrics_filepath, 'r') as f:
        metrics = json.load(f)

    accuracy = metrics.get('Accuracy')
    misclass = 1 - accuracy
    cm = np.array(metrics.get('Confusion Matrix'))

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    target_names = [str(t) for t in range(10)]
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(output_path)

if __name__ == '__main__':
    main()