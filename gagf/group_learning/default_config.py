import numpy as np

# Dataset Parameters
group_name = ["znz_znz"]  # , 'octahedral', 'cyclic', 'dihedral', 'znz_znz' 'A5']
group_n = [5]  # n in Dn [3, 4, 5] we are doingn D6
# TODO: don't include this image_length here, since it should only be for the znz_znz group.
image_length = [5]  # , 10, 15] # length of one side of the square image patch

powers = {
    "znz_znz": None,
    "cyclic": [0.0, 1000.0, 200.0, 300.0, 0.0, 0.0], # only uses 6 bc real and imag irreps are merged in escnn
    "dihedral": [0.0, 3000.0,  2000.0, 1000.0],  # [0.0, 3200.0,  50.0, 400.0, 1600.0, 500.0],
    # "dihedral": [0.0, 3600.0,  400.0, 1200.0, 2000.0, 900.0],  # [0.0, 3200.0,  50.0, 400.0, 1600.0, 500.0],
    "octahedral": [0.0, 400.0, 100., 900.0, 1000.0],  # [1, 3, 3, 2, 1]   # [1, 3, 3, 2, 1] #[0.0, 200.0, 10., 850.0, 1000.0] at 10000 for 18gbbmou
    "A5": [0.0, 1300.0, 100., 2000.0, 600.],  # [1, 3, 5, 3, 4]
}

fourier_coef_diag_values = {
    "znz_znz": None,
    "cyclic": [
        np.sqrt(24*p / dim**2) for p, dim in zip(powers["cyclic"], [1, 2, 2, 2, 2, 1])
    ],
    "dihedral": [
        np.sqrt(10*p / dim**2) for p, dim in zip(powers["dihedral"], [1, 1, 2, 2])
    ],
    # "dihedral": [
    #     np.sqrt(12*p / dim**2) for p, dim in zip(powers["dihedral"], [1, 1, 2, 2, 1, 1])
    # ],
    "octahedral": [
        np.sqrt(24*p / dim**2) for p, dim in zip(powers["octahedral"], [1, 3, 3, 2, 1])
    ],
    "A5":[
        [
            np.sqrt(60*p / dim**2) for p, dim in zip(powers["A5"], [1, 3, 5, 3, 4])
        ] for i in powers["A5"]
    ]
}

# Model Parameters
hidden_factor = [50]  # hidden size = hidden_factor * group_size

# Learning Parameters
seed = [10]  # [10, 20, 30, 40]  # , 30, 40, 50] #, 60, 70, 80, 90, 100]
init_scale = [1e-2]  # [1e-5, 1e-6, 1e-4]  # originally 1e-2 for cnxcn .. 1e-6 for dihedral, 1e-4 for cn x cn
lr = [0.01]  # , 0.0001]  # originaly 0.01. 0.00001 for dihedral6, 0.01 for cn x cn 0.00001 for D5
mom = [0.9]  # originaly 0.9
optimizer_name = [
    "PerNeuronScaledSGD"
]  # , 'SGD' Adam', "Custom'" "PerNeuronScaledSGD"]

# Training parameters
epochs = [15000]  # , 20000, 30000] #, 10000] #, 10000, 20000, 30000], 50000 for cn x cn 10000 for D5
verbose_interval = [100]  # 100
checkpoint_interval = [5000]
batch_size = [128]  #    128, 256]

# plotting parameters
power_logscale = False

# Change these if you want to resume training from a checkpoint
resume_from_checkpoint = False
checkpoint_epoch = 10000
checkpoint_run_name_to_load = "y8iuevxm"

# znz_znz specific parameters
# mnist_digit = [4]
# frequencies_to_learn = [3, 6, 9]  # number of frequencies to learn in the template

dataset_fraction = {
    "cyclic": 1.0,
    "znz_znz": 1.0,
    "dihedral": 1.0,
    "octahedral": 0.6,
    "A5": 0.6,
}

# git_root_path = setcwd.get_root_dir()
# model_save_dir = "/tmp/nmiolane/"
model_save_dir = "/tmp/adele/"



