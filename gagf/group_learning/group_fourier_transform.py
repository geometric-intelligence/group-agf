from escnn.group import *
import numpy as np


def compute_group_fourier_coef(group, template, irrep):
    """Compute the Fourier coefficient of template x at irrep rho.
    
    hat x [rho] = sum_{g in G} x[g] * rho(g).conj().T

    Formula from the Group-AGF paper.
    
    Parameters
    ----------
    group : Group
        The group (escnn object)
    template : np.ndarray, shape=[group.order()]
        The template to compute the Fourier coefficient of.
    irrep : IrreducibleRepresentation
        The irrep (escnn object).

    Returns
    -------
    _ : np.ndarray, shape=[irrep.size, irrep.size]
        The (matrix) Fourier coefficient of template x at irrep rho.
    """
    return sum([template[i_g] * irrep(g).conj().T for i_g, g in enumerate(group.elements)])

def compute_group_fourier_transform(group, template):
    """Compute the group Fourier transform of the template.
    
    Parameters
    ----------
    group : Group
        The group (escnn object)
    template : np.ndarray, shape=[group.order()]
        The template to compute the Fourier transform of.
    
    Returns
    -------
    _: list of np.ndarray, each of shape=[irrep.size, irrep.size]
        A list of (matrix) Fourier coefficients of template at each irrep.
        Since each irrep has a different size (dimension), the (matrix) Fourier
        coefficients have different shapes: the list cannot be concatenated
        into a single array.
    """
    irreps = group.irreps()
    fourier_coefs = []
    for irrep in irreps:
        fourier_coef = compute_group_fourier_coef(group, template, irrep)
        fourier_coefs.append(fourier_coef)
    return fourier_coefs

def compute_group_inverse_fourier_element(group, fourier_transform, g):
    """Compute the inverse Fourier transform at element g.
    
    Using the formula (Wikipedia):
    x(g) = 1/|G| * sum_{rho in irreps} dim(rho) * Tr(rho(g) * hat x[rho])

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    fourier_transform : list of np.ndarray, each of shape=[irrep.size, irrep.size]
        The (matrix) Fourier coefficients of template at each irrep.
    g : GroupElement (escnn object)
        The element of the group to compute the inverse Fourier transform at.

    Returns
    -------
    _ : np.ndarray, shape=[group.order()]
        The inverse Fourier transform at element g.
    """
    irreps = group.irreps()
    inverse_fourier_element = 1/group.order() * sum(
        [irrep.size * np.trace(irrep(g) @ fourier_transform[i]) 
        for i, irrep in enumerate(irreps)])
    return inverse_fourier_element

def compute_group_inverse_fourier_transform(group, fourier_transform):
    """Compute the inverse Fourier transform.
    
    Parameters
    ----------
    group : Group (escnn object)
        The group.
    fourier_transform : list of np.ndarray, each of shape=[irrep.size, irrep.size]
        The (matrix) Fourier coefficients of template at each irrep.
        
    Returns
    -------
    _ : np.ndarray, shape=[group.order()]
        The inverse Fourier transform: a signal over the group.
    """
    return np.array([
        compute_group_inverse_fourier_element(
            group, fourier_transform, g) for g in group.elements])

def compute_group_power_spectrum(group, template):
    """Compute the (group) power spectrum of the template.

    For each irrep rho, the power is given by:
    ||hat x(rho)||_rho = dim(rho) * Tr(hat x(rho)^dagger * hat x(rho))
    where hat x(rho) is the (matrix) Fourier coefficient of template x at irrep rho.

    Parameters
    ----------
    group : Group (escnn object)
        The group.
    template : np.ndarray, shape=[group.order()]
        The template to compute the power spectrum of.

    Returns
    -------
    _ : np.ndarray, shape=[len(group.irreps())]
        The power spectrum of the template.
    """
        
    irreps = group.irreps()

    power_spectrum = np.zeros(len(irreps))
    for i, irrep in enumerate(irreps):
        fourier_coef = compute_group_fourier_coef(group, template, irrep)
        power_spectrum[i] = irrep.size * np.trace(fourier_coef.conj().T @ fourier_coef)  # TODO: check if this is correct 
        # print(f"Power of {irrep.name}: {power_spectrum[i]}, type: {type(power_spectrum[i])}")
    return np.array(power_spectrum)

