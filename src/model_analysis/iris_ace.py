# -*- coding: utf-8 -*-
import model_analysis.ace as ace
import click
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class neural_network(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(neural_network, self).__init__()
        self.hidden = nn.Linear(num_input,num_hidden)
        self.out = nn.Linear(num_hidden,num_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.out(x)

@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('covariance_filepath', type=click.Path(exists=True))
@click.argument('means_filepath', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(model_filepath, covariance_filepath, means_filepath, output_path):
    model = torch.load(model_filepath)
    cov = torch.load(covariance_filepath)
    mean = torch.load(means_filepath)

    interventions = np.linspace(0, 1, 1000)
    ie = ace.interventional_expectation(model, mean, cov, interventions, epsilon=0.000001, method='hessian_diag', progress=True)
    avg_ce = ace.average_causal_effect(ie)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Remove names for now since named tensors aren't serializable
    ie = ie.rename(None)
    avg_ce = avg_ce.rename(None)
    
    torch.save(ie, os.path.join(output_path, 'interventional_expectations.pt'))
    torch.save(avg_ce, os.path.join(output_path, 'average_causal_effects.pt'))
    torch.save(torch.Tensor(interventions), os.path.join(output_path, 'interventions.pt'))

if __name__ == '__main__':
    main()
