#!/usr/bin/python

"""
Model producer.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids
"""

import torch
from typing import Mapping, Any, Tuple
from nflows.flows import Flow
from torch.nn import Module


def make_model(
    lower: float,
    upper: float,
    bijector: Mapping[str, Any],
    base: Mapping[str, Any],
    coord_trans: Mapping[str, Any],
    # energy_layer: Mapping[str, Any]
) -> Tuple[Flow, Module]:
    """Constructs a IDP model, with various configuration options.

    The model is implemented as follows:
    1. We draw N conformers randomly from a base distribution.
       All conformers have the same backbone but different dihedral angles.
    2. We jointly transform the dihedral angles with a flow (a nflow bijector).

    Args:
      lower: float, the lower ranges of the angle.
      upper: float, the upper ranges of the angle.
      bijector: configures the bijector that transforms angles. Expected to
        have the following keys:
        * 'constructor': a callable that creates the bijector.
        * 'kwargs': keyword arguments to pass to the constructor.
      base: configures the base distribution. Expected to have the following keys:
        * 'constructor': a callable that creates the base distribution.
        * 'kwargs': keyword arguments to pass to the constructor.

    Returns:
      A particle model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model = base['constructor'](
        **base['kwargs'])
    bij = bijector['constructor'](
        angles=base_model.torsion_angles,
        lower=lower,
        upper=upper,
        **bijector['kwargs']).to(device)

    model = Flow(bij, base_model)

    trans = coord_trans['constructor'](
        mol=base_model.mol,
        angles=base_model.torsion_angles)
    # energy = energy_layer['constructor']()
    # mol=base_model.mol)

    # energy_fn = Sequential(trans, energy)

    return model, trans
