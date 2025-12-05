import numpy as np

# Dataset Parameters
group_name = ["octahedral"] #, "A5"]  # , 'octahedral', 'cyclic', 'dihedral', 'znz_znz' 'A5']
group_n = [6]  # n in Dn [3, 4, 5]

i_powers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #, 1] #, 2, 3, 4]

powers = {
    "dihedral": [0.0, 0.0, 0.0, 0.0, 0.0],
    "octahedral": [# [1, 3, 3, 2, 1] 
        [0.0, 2000.0, 0., 0.0, 0.0] , # 1:100, 3: 900
         [0.0, 1600.0, 0.0, 0.0, 0.0] ,  # 2:400, 3: 900
         [0.0, 1200.0, 0.0, 0.0, 0.0] ,  # 3:900, 3:900
         [0.0, 800.0, 0.0, 0.0, 0.0] ,  # 3:900, 3:900
         [0.0, 400.0, 0.0, 0.0, 0.0] ,  # 3:900, 3:900
         [0.0, 0.0, 0.0, 2000.0, 0.0] ,  # 3:900, 3:900
         [0.0, 0.0, 0.0, 1600.0, 0.0] ,  # 3:900, 3:900
         [0.0, 0.0, 0.0, 1200.0, 0.0] ,  # 3:900, 3:900
         [0.0, 0.0, 0.0, 800.0, 0.0] ,  # 3:900, 3:900
         [0.0, 0.0, 0.0, 400.0, 0.0] ,  # 3:900, 3:900
        ],  
    "A5": [# [1, 3, 5, 3, 4]
       [0.0, 1800.0, 0., 1800.0, 0.],  # 3:900, 3:900
       [0.0, 900.0, 0.0, 0.0, 1600.],  #3:900, 4:1600
       [0.0, 0.0, 2500.0, 900.0, 0.0] ,  # 5:2500, 3:900
       [0.0, 0.0, 2500.0, 0.0, 1600.0] ,  # 5:2500, 4:1600
    ],
}

fourier_coef_diag_values = {
    "dihedral": [0.0, 0.0, 0.0, 0.0, 0.0],
    "octahedral": [
        [
            np.sqrt(24*p / dim**2) for p, dim in zip(powers["octahedral"][i], [1, 3, 3, 2, 1])
        ] for i in i_powers
    ],
    # "A5":[
    #     [
    #         np.sqrt(60*p / dim**2) for p, dim in zip(powers["A5"][i], [1, 3, 5, 3, 4])
    #     ] for i in i_powers
    # ]
}

# Model Parameters
hidden_factor = [20] #, 30, 40, 50]  # hidden size = hidden_factor * group_size

# Learning Parameters
seed = [10]
init_scale = [1e-3]#, 1e-4, 1e-5, 1e-6]
lr = [0.0001] #, 0.00001]
mom = [0.9]
optimizer_name = ["PerNeuronScaledSGD"]
epochs = [1000] #, 50000]
verbose_interval = 100
checkpoint_interval = 200000
batch_size = [128]  #    128, 256]

# plotting parameters
power_logscale = False

# Change these if you want to resume training from a checkpoint
resume_from_checkpoint = False
checkpoint_epoch = 50000
checkpoint_run_name_to_load = "run_0nnx3ive"

# znz_znz specific parameters
image_length = [5]

i_dataset_fractions = [0]
dataset_fraction = {
    "cyclic": 1.0,
    "znz_znz": 0.4,
    "dihedral": 1.0,
    "octahedral": [1.],
    "A5": [1.0], # [0.2, 0.3, 0.4, 0.5, 0.6]
}

model_save_dir = "/tmp/nmiolane/"
# model_save_dir = "/tmp/adele/"
