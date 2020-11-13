import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm, trange

# TODO: put in utilities (look in successful_attacks.py too)
def convert_one_hot_to_num(one_hot):
    return torch.argmax(one_hot).item()


# TODO: put in utilities (waiting because it will repro almost everything)
def get_indices_from_index(tensor, index):
    strides = list(tensor.stride())
    indices = []
    residue = index
    for idx, val in enumerate(strides):
        indices.append((residue // val).item())
        if idx < len(strides) - 1:
            residue = residue % val
    return tuple(indices)


def targeted_attack_from_known_class(
    ace, interventions, inp, predicted_class, target_class, norm, budget
):
    raise NotImplementedError("Targeted attack from known class not yet implemented!")


def untargeted_attack_from_known_class(
    ace, interventions, inp, predicted_class, norm, budget
):
    raise NotImplementedError("Untargeted attack from known class not yet implemented!")


def targeted_attack_from_unknown_class(
    ace, interventions, inp, target_class, norm, budget
):
    target_class = convert_one_hot_to_num(target_class)
    mask = torch.ones_like(inp)
    attack = inp.clone()

    while (
        torch.norm(inp - attack, norm) < budget and torch.max(mask).item() == 1.0
    ):  # TODO: fix later to break before we bust budget
        aces = ace[:, target_class, :] * mask.unsqueeze(1)
        pixel_idx, intervention_idx = get_indices_from_index(aces, torch.argmax(aces))

        attack[pixel_idx] = interventions[intervention_idx]
        mask[pixel_idx] = 0.0

    return attack


def untargeted_attack_from_unknown_class(ace, interventions, inp, norm, budget):
    raise NotImplementedError(
        "Untargeted attack from unkown class not yet implemented!"
    )


def ace_attack(
    ace,
    interventions,
    inputs,
    predicted_classes=None,
    target_classes=None,
    norm=0,
    budget=100,
):
    """
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
    """

    inputs_shape = inputs.shape
    attacks = inputs.clone()

    # Flatten the inputs and attacks
    inputs = inputs.reshape(inputs_shape[0], -1)
    attacks = attacks.reshape(inputs_shape[0], -1)

    for input_index in range(inputs_shape[0]):
        inp = inputs[input_index, :]

        predicted_class = (
            predicted_classes[input_index, :] if predicted_classes is not None else None
        )
        target_class = (
            target_classes[input_index, :] if target_classes is not None else None
        )

        if predicted_class is not None and target_class is not None:
            attack = targeted_attack_from_known_class(
                ace, interventions, inp, predicted_class, target_class, norm, budget
            )
        elif predicted_class is not None and target_class is None:
            attack = untargeted_attack_from_known_class(
                ace, interventions, inp, predicted_class, norm, budget
            )
        elif predicted_class is None and target_class is not None:
            attack = targeted_attack_from_unknown_class(
                ace, interventions, inp, target_class, norm, budget
            )
        else:
            attack = untargeted_attack_from_unknown_class(
                ace, interventions, inp, norm, budget
            )

        attacks[input_index] = attack

    return attacks.reshape(inputs_shape)
