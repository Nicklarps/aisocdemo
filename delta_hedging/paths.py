"""Simulated stock paths: exact geometric Brownian motion.

Model (PDF section 2.1):   dS = mu S dt + sigma S dW

Applying Ito's lemma to ln S gives  d ln S = (mu - sigma^2/2) dt + sigma dW,
which integrates exactly over any step dt (PDF section 2.2):

    S_{t+dt} = S_t * exp[(mu - sigma^2/2) dt + sigma sqrt(dt) Z],   Z ~ N(0, 1)

Because this is the exact solution and not an Euler approximation, there is
no discretisation error in the paths however large dt is. The -sigma^2/2
term is the "volatility drag": leave it out and E[S_T] comes out wrong.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_gbm"]


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, rng):
    """Simulate GBM paths on an even time grid.

    Parameters
    ----------
    S0 : float        starting price
    mu : float        real-world drift of the price (per year)
    sigma : float     realised volatility of the world (per year)
    T : float         horizon in years
    n_steps : int     number of time steps; the grid has n_steps + 1 points
    n_paths : int     number of independent paths
    rng : numpy.random.Generator, e.g. ``np.random.default_rng(42)``

    Returns
    -------
    S : ndarray, shape (n_paths, n_steps + 1), with S[:, 0] == S0.
    """
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z  # Ito drift correction
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(log_inc, axis=1))
    return S
