#!/usr/bin/python

"""
Conditioner for IDP models.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids
"""

# import functools
from typing import Callable, Mapping, Any
import torch
from torch import Tensor
from torch.nn import (Linear, Module)


def circular(x: Tensor,
             lower: float,
             upper: float,
             num_frequencies: int) -> Tensor:
    """Maps angles to points on the unit circle.

    The mapping is such that the interval [lower, upper] is mapped to a full
    circle starting and ending at (1, 0). For num_frequencies > 1, the mapping
    also includes higher frequencies which are multiples of 2 pi/(lower-upper)
    so that [lower, upper] wraps around the unit circle multiple times.

    Args:
        x: array of shape [..., D].
        lower: lower limit, angles equal to this will be mapped to (1, 0).
        upper: upper limit, angles equal to this will be mapped to (1, 0).
        num_frequencies: number of frequencies to consider in the embedding.

    Returns:
        An array of shape [..., 2*num_frequencies*D].
    """
    base_frequency = 2. * torch.pi / (upper - lower)
    frequencies = base_frequency * torch.arange(1, num_frequencies+1)
    angles = frequencies * (x[..., None] - lower)
    # Reshape from [..., D, num_frequencies] to [..., D*num_frequencies].
    angles = angles.reshape(x.shape[:-1] + (-1,))
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.concat([cos, sin], axis=-1)


# def _reshape_last(x: Tensor, ndims: int, new_shape: Sequence[int]) -> Tensor:
#     """Reshapes the last `ndims` dimensions of `x` to shape `new_shape`."""
#     if ndims <= 0:
#         raise ValueError(
#             f'Number of dimensions to reshape must be positive, got {ndims}.')
#     return torch.reshape(x, x.shape[:-ndims] + tuple(new_shape))


# def make_equivariant_conditioner(
#     num_bijector_params: int,
#     lower: float,
#     upper: float,
#     embedding_size: int,
#     conditioner_constructor: Callable[..., Any],
#     conditioner_kwargs: Mapping[str, Any],
#     num_frequencies: int,
# ) -> Sequential:
#     """Make a conditioner for the coupling flow."""
#     # This conditioner assumes that the input is of shape [..., N, D1]. It returns
#     # an output of shape [..., N, D2, K], where:
#     #   D2 = `shape_transformed[-1]`
#     #   K = `num_bijector_params`
#     conditioner = conditioner_constructor(**conditioner_kwargs)
#     return Sequential([
#         functools.partial(
#             circular, lower=lower, upper=upper,
#             num_frequencies=num_frequencies),
#         Linear(embedding_size),
#         conditioner,
#         Linear(num_bijector_params),
#         functools.partial(
#             _reshape_last, ndims=1, new_shape=(num_bijector_params)),
#     ])


class Conditioner(Module):
    """Make a conditioner for the coupling flow."""

    def __init__(self,
                 num_bijector_params: int,
                 lower: float,
                 upper: float,
                 angles: Tensor,
                 embedding_size: int,
                 conditioner_constructor: Callable[..., Any],
                 conditioner_kwargs: Mapping[str, Any],
                 num_frequencies: int):
        """
        This conditioner assumes that the input is of shape [..., N, D1]. 
        It returns an output of shape [..., N, K], where:
        K = `num_bijector_params`
        """
        super().__init__()
        self.linear1 = Linear(in_features=angles.shape[-1],
                              out_features=embedding_size)
        self.conditioner = conditioner_constructor(**conditioner_kwargs)
        self.linear2 = Linear(in_features=embedding_size,
                              out_features=num_bijector_params)
        self.circular_kwards = dict(
            lower=lower,
            upper=upper,
            num_frequencies=num_frequencies
        )

    def forward(self, input: Tensor):
        out = circular(input, **self.circular_kwards)
        out = self.linear1(out)
        out = self.conditioner(out, out)
        out = self.linear2(out)
        return out
