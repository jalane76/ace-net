# -*- coding: utf-8 -*-
from art.classifiers import PyTorchClassifier
from attacks.ace_attack import ace_attack
import click
import csv
import json
from models.mnist_model import MnistModel
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

def run_attack(ace, interventions, inputs, original_classes, target_classes, norm, budget, classifier, results_output_filepath):
    target_class = np.argmax(target_classes[0]).item()

    attacks = ace_attack(ace, interventions, inputs, target_classes=target_classes, norm=norm, budget=budget)
    predictions = classifier.predict(attacks)

    torch.save(attacks, os.path.join(results_output_filepath, 'attacks_target-{}_norm-{}_budget-{}.pt'.format(target_class, norm, budget)))
    torch.save(predictions, os.path.join(results_output_filepath, 'predictions_target-{}_norm-{}_budget-{}.pt'.format(target_class, norm, budget)))

    results_dict = {}
    for c in range(10):
        count = np.sum(np.argmax(predictions, axis=1) == c)
        results_dict['Class {} Count'.format(str(c))] = str(count)

    results_dict['Accuracy'] = np.sum(np.argmax(predictions, axis=1) == np.argmax(original_classes, axis=1)) / len(original_classes)
    results_dict['Target Accuracy'] = np.sum(np.argmax(predictions, axis=1) == np.argmax(target_classes.numpy(), axis=1)) / len(target_classes.numpy())
    results_dict['Norm'] = 'L_{}'.format(norm)
    results_dict['Budget'] = str(budget)
    results_dict['Target Class'] = target_class
    return results_dict

@click.command()
@click.argument('x_test_filepath', type=click.Path(exists=True))
@click.argument('y_test_filepath', type=click.Path(exists=True))
@click.argument('clip_values_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('optimizer_filepath', type=click.Path(exists=True))
@click.argument('ace_filepath', type=click.Path(exists=True))
@click.argument('interventions_filepath', type=click.Path(exists=True))
@click.argument('results_output_filepath', type=click.Path())
def main(x_test_filepath, y_test_filepath, clip_values_filepath, model_filepath, optimizer_filepath, ace_filepath, interventions_filepath, results_output_filepath):

    seed = 45616451
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    x_test = torch.load(x_test_filepath)
    x_test_shape = x_test.shape
    x_test = torch.tensor(x_test.reshape(x_test.shape[0], -1))
    y_test = torch.load(y_test_filepath)
    ace = torch.load(ace_filepath).cpu()
    interventions = torch.load(interventions_filepath).cpu()

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

    target_classes = range(10)
    L0_budgets = np.linspace(1, 121, num=10)
    L1_budgets = np.linspace(1, 100, num=10)
    L2_budgets = np.linspace(.1, 20, num=10)
    Linf_budgets = np.linspace(.1, 1.0, num=10)
    total_iters = len(target_classes) * (len(L0_budgets) + len(L1_budgets) + len(L2_budgets) + len(Linf_budgets))

    if not os.path.exists(results_output_filepath):
        os.makedirs(results_output_filepath)

    with open(os.path.join(results_output_filepath, 'unknown_to_known_permutations_results.csv'), mode='w') as csv_file:
        csv_header = ['Class {} Count'.format(str(c)) for c in target_classes] + ['Accuracy', 'Target Accuracy', 'Norm', 'Budget', 'Target Class']
        writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        writer.writeheader()

        with tqdm(total=total_iters) as pbar:
            for target_class in target_classes:
                # Target the attacks to a particular class
                y_test_adv = torch.zeros_like(torch.from_numpy(y_test))
                y_test_adv[:, target_class] = 1.0

                # L0 norm
                norm = 0
                for budget in L0_budgets:
                    results = run_attack(ace, interventions, x_test, y_test, y_test_adv, norm, budget, classifier, results_output_filepath)
                    writer.writerow(results)
                    pbar.update(1)

                # L1 norm
                norm = 1
                for budget in L1_budgets:
                    results = run_attack(ace, interventions, x_test, y_test, y_test_adv, norm, budget, classifier, results_output_filepath)
                    writer.writerow(results)
                    pbar.update(1)

                # L2 norm
                norm = 2
                for budget in L2_budgets:
                    results = run_attack(ace, interventions, x_test, y_test, y_test_adv, norm, budget, classifier, results_output_filepath)
                    writer.writerow(results)
                    pbar.update(1)

                # Linf norm
                norm = float('inf')
                for budget in Linf_budgets:
                    results = run_attack(ace, interventions, x_test, y_test, y_test_adv, norm, budget, classifier, results_output_filepath)
                    writer.writerow(results)
                    pbar.update(1)

if __name__ == '__main__':
    main()