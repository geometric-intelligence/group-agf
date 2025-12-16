import numpy as np

# Dataset Parameters
group_name = "cnxcn" #, "A5"]  # , 'octahedral', 'cn', 'dihedral', 'cnxcn' 'A5']
group_n = [6]  # n in Dn [3, 4, 5]
template_type = "one_hot" # "one_hot", "irrep_construction"]

powers = {
    "cn": [[0, 12.5, 10, 7.5, 5, 2.5]],
    "cnxcn": [[0, 12, 10, 8, 6, 4]],
    "dihedral": [[0.0, 5.0, 0.0, 7.0, 0.0, 0.0]],  # D6: [1,1,2,2,1,1], D5: [1,1,2,2], D3: [1,1,2]
    "octahedral": [# [1, 3, 3, 2, 1] 
        [0.0, 2000.0, 0., 0.0, 0.0],
        ],  
    "A5": [# [1, 3, 5, 3, 4]
       [0.0, 1800.0, 0., 1800.0, 0.],  # 3:900, 3:900
       [0.0, 900.0, 0.0, 0.0, 1600.],  #3:900, 4:1600
       [0.0, 0.0, 2500.0, 900.0, 0.0] ,  # 5:2500, 3:900
       [0.0, 0.0, 2500.0, 0.0, 1600.0] ,  # 5:2500, 4:1600
    ],
}

# Model Parameters
hidden_factor = [30] #20, 30, 40, 50]  # hidden size = hidden_factor * group_size

# Learning Parameters
seed = [10]
init_scale = {
    "cn": [1e-2],
    "cnxcn": [1e-2],
    "dihedral": [1e-6],
    "octahedral": [1e-3],
    "A5": [1e-3],
}
lr = {
    "cn": [0.01],
    "cnxcn": [0.01],
    "dihedral": [0.01],
    "octahedral": [0.0001],
    "A5": [0.0001],
}

mom = [0.9]
optimizer_name = ["PerNeuronScaledSGD"]
epochs = [1000] #, 50000]
verbose_interval = 100
checkpoint_interval = 200000
batch_size = [128]  #    128, 256]

# plotting parameters
power_logscale = False

# Change these if you want to resume training from a checkpoint
resume_from_checkpoint = True
checkpoint_epoch = 5000

# cnxcn specific parameters
image_length = [5]

dataset_fraction = {
    "cn": 1.0,
    "cnxcn": 1.0,
    "dihedral": 1.0,
    "octahedral": 1.0,
    "A5": 1.0, # [0.2, 0.3, 0.4, 0.5, 0.6]
}

# model_save_dir = "/tmp/nmiolane/"
model_save_dir = "/tmp/adele/"
