#!/usr/bin/python

"""
Model producer.
Code adapted from https://github.com/deepmind/flows_for_atomic_solids
"""

from logging import Logger
from typing import Mapping, Any, Tuple
from torch.nn import Module, Identity
from nflows.distributions.base import Distribution
from nflows.transforms import Transform
from nflows.utils import torchutils


class Flow(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform: Transform, distribution: Distribution, embedding_net=None):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        if embedding_net is not None:
            assert isinstance(embedding_net, Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform.inverse(
            inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        noise = self._distribution.sample(
            num_samples, context=embedded_context)

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(
                samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=embedded_context
        )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(
                samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(
                logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform.inverse(
            inputs, context=self._embedding_net(context))
        return noise


def make_model(
    lower: float,
    upper: float,
    bijector: Mapping[str, Any],
    base: Mapping[str, Any],
    coord_trans: Mapping[str, Any],
    logger: Logger,
    device='cuda'
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
    base_model = base['constructor'](
        **base['kwargs'],
        logger=logger,
        device=device)
    bij = bijector['constructor'](
        angles=base_model.torsion_angles,
        lower=lower,
        upper=upper,
        logger=logger,
        device=device,
        **bijector['kwargs']).to(device)

    model = Flow(bij, base_model)

    trans = coord_trans['constructor'](
        mol=base_model.mol,
        angles=base_model.torsion_angles,
        logger=logger,
        device=device)

    return model, trans
