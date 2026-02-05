import numpy as np
from escnn.group import Octahedral

from group_agf.binary_action_learning.templates import fixed_group_template
from group_agf.binary_action_learning.power import GroupPower


def test_power_custom_template():
    group = Octahedral()
    irrep_sizes = [irrep.size for irrep in group.irreps()]
    print("Irrep sizes:", irrep_sizes)
    seed = 42
    powers = [0., 20.0, 20.0, 100.0, 0.0]  # on irreps [1, 3, 3, 2, 1]
    fourier_coef_diag_values = [np.sqrt(group.order()*p / dim**2) for p, dim in zip(powers, irrep_sizes)]
    template = fixed_group_template(group, fourier_coef_diag_values=fourier_coef_diag_values)

    gp = GroupPower(template, group)
    power = gp.power
    expected_powers = powers

    print("Computed power spectrum:", power)
    print("Expected powers:", expected_powers)
    print("Max diff:", np.max(np.abs(power - expected_powers)))
    assert np.allclose(
        power, expected_powers
    ), "Power spectrum does not match expected values"
