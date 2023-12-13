"""Utility functions for coupling filter definition."""

from typing import Tuple

import numpy as np
import scipy.stats as sts
from numpy.typing import NDArray

from .basis import Basis


def difference_of_gammas(
    ws: int,
    upper_percentile: float = 0.99,
    inhib_a: float = 1.0,
    excit_a: float = 2.0,
    inhib_b: float = 2.0,
    excit_b: float = 2.0,
) -> NDArray:
    r"""Generate coupling filter as a Gamma pdf difference.

    Parameters
    ----------
    ws:
        The window size of the filter.
    upper_percentile:
        Upper bound of the gamma range as a percentile. The gamma function
        will be evaluated over the range [0, ppf(upper_percentile)].
    inhib_a:
        The `a` constant for the gamma pdf of the inhibitory part of the filter.
    excit_a:
        The `a` constant for the gamma pdf of the excitatory part of the filter.
    inhib_b:
        The `b` constant for the gamma pdf of the inhibitory part of the filter.
    excit_b:
        The `a` constant for the gamma pdf of the excitatory part of the filter.

    Notes
    -----
    The probability density function of a gamma distribution is parametrized as
    follows$^1$,
    $$
        p(x;\; a, b) = \frac{b^a x^{a-1} e^{-x}}{\Gamma(a)},
    $$
    where $\Gamma(a)$ refers to teh gamma function, see$^1$.

    Returns
    -------
    filter:
        The coupling filter.

    References
    ----------
    1. [SciPy Docs - "scipy.stats.gamma"](https://docs.scipy.org/doc/
    scipy/reference/generated/scipy.stats.gamma.html)
    """
    gm_inhibition = sts.gamma(a=inhib_a, scale=1 / inhib_b)
    gm_excitation = sts.gamma(a=excit_a, scale=1 / excit_b)

    # calculate upper bound for the evaluation
    xmax = max(gm_inhibition.ppf(upper_percentile), gm_excitation.ppf(upper_percentile))
    # equi-spaced sample covering the range
    x = np.linspace(0, xmax, ws)

    # compute difference of gammas & normalize
    gamma_diff = gm_excitation.pdf(x) - gm_inhibition.pdf(x)
    gamma_diff = gamma_diff / np.linalg.norm(gamma_diff, ord=2)

    return gamma_diff


def regress_filter(
    coupling_filter_bank: NDArray, basis: Basis
) -> Tuple[NDArray, NDArray]:
    """Approximate scipy.stats.gamma based filters with basis function.

    Find the ols weights for representing the filters in terms of basis functions.
    This is done to re-use the nsl.glm.simulate method.

    Parameters
    ----------
    coupling_filter_bank:
        The coupling filters. Shape (n_neurons, n_neurons, window_size)
    basis:
        The basis function to instantiate.

    Returns
    -------
    eval_basis:
        The basis matrix, shape (window_size, n_basis_funcs)
    weights:
        The weights for each neuron. Shape (n_neurons, n_neurons, n_basis_funcs)
    """
    n_neurons, _, ws = coupling_filter_bank.shape
    eval_basis = basis.evaluate(np.linspace(0, 1, ws))

    # Reshape the coupling_filter_bank for vectorized least-squares
    filters_reshaped = coupling_filter_bank.reshape(-1, ws)

    # Solve the least squares problem for all filters at once
    # (vecotrizing the features)
    weights = np.linalg.lstsq(eval_basis, filters_reshaped.T, rcond=None)[0]

    # Reshape back to the original dimensions
    weights = weights.T.reshape(n_neurons, n_neurons, -1)

    return eval_basis, weights
