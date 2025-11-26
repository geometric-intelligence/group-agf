import numpy as np
from escnn.group import Octahedral
from gagf.group_learning.datasets import generate_fixed_group_template
from gagf.group_learning.power import GroupPower

def test_power_custom_template():
    group = Octahedral()
    print("Irrep sizes:", [irrep.size for irrep in group.irreps()])
    seed = 42
    powers = [100., 20., 0., 0., 0.] # on irreps [1, 3, 3, 2, 1]
    template = generate_fixed_group_template(group, seed=seed, powers=powers)

    gp = GroupPower(template, group)
    power = gp.power
    expected_powers = [
        irrep.size ** 2 * power**2 / group.order() for irrep, power in zip(group.irreps(), powers)
    ]

    print("Computed power spectrum:", power)
    print("Expected powers:", expected_powers)
    print("Max diff:", np.max(np.abs(power - expected_powers)))
    assert np.allclose(power, expected_powers), "Power spectrum does not match expected values"


