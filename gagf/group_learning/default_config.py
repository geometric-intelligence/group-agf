# Dataset Parameters
import os

p = [5, 10, 12]
mnist_digit = [4]
dataset_fraction = [0.3, 0.4] # fraction of the total dataset to train on
group = ['znz_znz'] #, 'dihedral']

# Learning Parameters
seed = [10]
init_scale = [1e-2, 1e-1]  # originally 1e-2
lr = [0.01, 0.001, 0.001]  # originaly 0.01
mom = [0.9] # originaly 0.9

# Training parameters
epochs = [5000, 10000, 20000, 30000] 
verbose_interval = [100]
batch_size = [128, 256] 
frequencies_to_learn = [3, 6, 9]  # number of frequencies to learn in the template

# git_root_path = setcwd.get_root_dir()
model_save_dir = "/tmp/adele/"


