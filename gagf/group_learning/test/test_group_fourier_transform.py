import numpy as np
import pytest

from escnn.group import Octahedral

from gagf.group_learning.datasets import generate_fixed_group_template
from gagf.group_learning.group_fourier_transform import (
    compute_group_fourier_transform,
    compute_group_inverse_fourier_transform,
)


# def test_fourier_transform_of_fixed_template():
#     group = Octahedral()
#     seed = 42

#     # Generate template with nontrivial spectrum
#     template = generate_fixed_group_template(
#         group, seed=seed, powers=[100.0, 20.0, 0.0, 0.0, 0.0]
#     )

#     # Forward Fourier transform
#     fourier_transform = compute_group_fourier_transform(group, template)

#     for i, ft in enumerate(fourier_transform):
#         print(f"Fourier transform element {i}:")
#         print(ft)
#     raise Exception("Stop here to check the Fourier transform.")


def test_fourier_inverse_is_identity():
    group = Octahedral()
    seed = 42

    # Generate template with nontrivial spectrum
    template = generate_fixed_group_template(group, seed=seed, fourier_coef_diag_values=[100.0, 20.0, 0.0, 0.0, 0.0])

    # Forward Fourier transform
    fourier_transform = compute_group_fourier_transform(group, template)

    # Inverse Fourier transform
    reconstructed = compute_group_inverse_fourier_transform(group, fourier_transform)

    # Perform Fourier transform of the reconstructed template
    fourier_transform_reconstructed = compute_group_fourier_transform(group, reconstructed)

    # Check that the original and reconstructed template are close
    assert np.allclose(template, reconstructed, atol=1e-10), \
        f"Inversion failed! max diff: {np.max(np.abs(template - reconstructed))}"
    print(f'diff: {(np.abs(template - reconstructed))}')

    # Check that the Fourier transform of the reconstructed template is close to the original Fourier transform
    print(f'fourier_transform: {[ft.shape for ft in fourier_transform]}')
    print(f'fourier_transform_reconstructed: {[ft.shape for ft in fourier_transform_reconstructed]}')
    assert len(fourier_transform) == len(fourier_transform_reconstructed), \
        f"Length mismatch: {len(fourier_transform)} vs {len(fourier_transform_reconstructed)}"
    for i, (ft, ft_rec) in enumerate(zip(fourier_transform, fourier_transform_reconstructed)):
        assert np.allclose(ft, ft_rec, atol=1e-10), \
            f"Fourier transform failed at index {i}! max diff: {np.max(np.abs(ft - ft_rec))}"
        print(f'diff at index {i}: {np.max(np.abs(ft - ft_rec))}')
    #print(f'diff: {(np.abs(fourier_transform - fourier_transform_reconstructed))}')

    raise Exception("Stop here to check the max diff.")

if __name__ == "__main__":
    test_fourier_inverse_is_identity()
