import numpy as np
from escnn.group import Octahedral

import src.fourier as fourier
import src.template as template


def test_group_fourier_inverse_is_identity():
    """Test that group_fourier followed by group_fourier_inverse reconstructs the original."""
    group = Octahedral()

    tpl = template.fixed_group(group, fourier_coef_diag_values=[100.0, 20.0, 0.0, 0.0, 0.0])

    fourier_coefs = fourier.group_fourier(group, tpl)
    reconstructed = fourier.group_fourier_inverse(group, fourier_coefs)

    assert np.allclose(tpl, reconstructed, atol=1e-10), (
        f"Inversion failed! max diff: {np.max(np.abs(tpl - reconstructed))}"
    )


def test_group_fourier_coefs_shape():
    """Test that group_fourier returns one coefficient matrix per irrep."""
    group = Octahedral()
    tpl = template.fixed_group(group, fourier_coef_diag_values=[100.0, 20.0, 0.0, 0.0, 0.0])

    fourier_coefs = fourier.group_fourier(group, tpl)

    assert len(fourier_coefs) == len(group.irreps())
    for coef, irrep in zip(fourier_coefs, group.irreps()):
        assert coef.shape == (irrep.size, irrep.size)


if __name__ == "__main__":
    test_group_fourier_inverse_is_identity()
    test_group_fourier_coefs_shape()
