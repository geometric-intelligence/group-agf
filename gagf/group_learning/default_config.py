# Dataset Parameters
group_name = ["octahedral"]  # , 'octahedral', 'cyclic', 'dihedral', 'znz_znz' 'A5']
group_n = [6]  # n in Dn [3, 4, 5]
# TODO: don't include this image_length here, since it should only be for the znz_znz group.
image_length = [5]  # , 10, 15] # length of one side of the square image patch


# Model Parameters
hidden_factor = [50] # hidden size = hidden_factor * group_size

# Learning Parameters
seed = [10] #[10, 20, 30, 40]  # , 30, 40, 50] #, 60, 70, 80, 90, 100]
init_scale = [1e-6] #[1e-5, 1e-6, 1e-4]  # originally 1e-2
lr = [0.0001]  # , 0.001]  # originaly 0.01
mom = [0.9]  # originaly 0.9
optimizer_name = ["PerNeuronScaledSGD"]  # , 'SGD' Adam', "Custom'" "PerNeuronScaledSGD"]

# Training parameters
epochs = [100000]  # , 20000, 30000] #, 10000] #, 10000, 20000, 30000]
verbose_interval = [1] # 100
batch_size = [128]  #    128, 256]

# plotting parameters
power_logscale = True

# znz_znz specific parameters
mnist_digit = [4]
frequencies_to_learn = [3, 6, 9]  # number of frequencies to learn in the template

dataset_fraction = {
    "cyclic": 1.,
    "znz_znz": 0.4,
    "dihedral": 1.,
    "octahedral": 0.3,
    "A5": 0.6,
}

# git_root_path = setcwd.get_root_dir()
model_save_dir = "/tmp/nmiolane/"
# model_save_dir = "/tmp/adele/"

