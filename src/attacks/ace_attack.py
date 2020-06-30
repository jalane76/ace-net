import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange

def ace_attack(ace, interventions, inputs, predicted_classes=None, target_classes=None, norm=0, budget=100):
    ''' 
        Create attack by perturbing the given input using the provided ACE and interventions.

        :param ace: A tensor containing the average causal effects calculated for the model, if given.
        :type ace: Named torch.Tensor with dimension names ('X', 'Y', 'I')

        :param interventions: Interventional values that were used to perturb the model used to calculate the ACE.
        :type interventions: torch.Tensor with shape (number_of_interventional_values)

        :param input: The input tensor to perturb as an attack.  The first dimension of the tensor is the batch size.
        :type input: torch.Tensor

        :param predicted_class: A Tensor containing the classes that the model currently predicts the inputs to be.  If None the attack will not take the ACE calculations for the predicted class (the 'Y' dimension in the ACE tensor) into account.  The first dimension is the batch size.
        :type predicted_class: torch.Tensor with the same shape as the model output.

        :param target_class: A Tensor containing the classes that the model should classify the completed attacks as.  If None the attack will not be targeted.  The first dimension is the batch size.
        :type target_class: torch.Tensor with the same shape as the model output.

        :param norm: The norm used to judge the difference between the input and the attack.
        :type norm: Valid values are from (0, 1, 2, 'inf').

        :param budget: The value that the difference calculated using the given norm should remain below when calculating the attack.
        :type budget: A number > 0

        :return: A tensor containing the attack consisting of the perturbed input
        :rtype: torch.Tensor with the same shape as the input tensor
    '''
    
    inputs_shape = inputs.shape
    attacks = inputs.clone()

    # Flatten the inputs and attacks
    inputs = inputs.reshape(inputs_shape[0], -1)
    attacks = attacks.reshape(inputs_shape[0], -1)

    for input_index in range(inputs_shape[0]):
        inp = inputs[input_index, :]
        attack = attacks[input_index, :]

        
    return attacks.reshape(inputs_shape)