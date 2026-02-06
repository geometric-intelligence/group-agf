import numpy as np
from escnn.group import *


def group_fourier(group, template):
    """Compute the group Fourier transform of the template.

    For each irrep rho, compute the Fourier coefficient:
        hat x [rho] = sum_{g in G} x[g] * rho(g).conj().T

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    template : np.ndarray, shape=[group.order()]
        The template to compute the Fourier transform of.

    Returns
    -------
    fourier_coefs : list of np.ndarray, each of shape=[irrep.size, irrep.size]
        A list of (matrix) Fourier coefficients of template at each irrep.
    """
    irreps = group.irreps()
    fourier_coefs = []
    for irrep in irreps:
        coef = sum([template[i_g] * irrep(g).conj().T for i_g, g in enumerate(group.elements)])
        fourier_coefs.append(coef)
    return fourier_coefs


def group_fourier_inverse(group, fourier_coefs):
    """Compute the inverse group Fourier transform.

    Using the formula:
    x(g) = 1/|G| * sum_{rho in irreps} dim(rho) * Tr(rho(g) * hat x[rho])

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    fourier_coefs : list of np.ndarray, each of shape=[irrep.size, irrep.size]
        The (matrix) Fourier coefficients of template at each irrep.

    Returns
    -------
    signal : np.ndarray, shape=[group.order()]
        The inverse Fourier transform: a signal over the group.
    """
    irreps = group.irreps()

    def _inverse_at_element(g):
        return (
            1
            / group.order()
            * sum(
                [
                    irrep.size * np.trace(irrep(g) @ fourier_coefs[i])
                    for i, irrep in enumerate(irreps)
                ]
            )
        )

    return np.array([_inverse_at_element(g) for g in group.elements])
