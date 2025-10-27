# Dataset Parameters
import os

p = [5, 10, 15]
mnist_digit = [4]
dataset_fraction = [0.2, 0.3] # fraction of the total dataset to train on
group = ['znz_znz'] #, 'dihedral']
template_type = ['mnist']  # 'mnist' or 'znz_znz_fixed' or dihedral_fixed

# Learning Parameters
seed = [10]
n_frequencies_to_learn = [6]
init_scale = [1e-2, 1e-1]  # originally 1e-2
lr = [0.01, 0.001, 0.001]  # originaly 0.01
mom = [0.9] # originaly 0.9

# Training parameters
epochs = [5000, 10000, 20000] 
verbose_interval = [100]
batch_size = [128, 256] 

# git_root_path = setcwd.get_root_dir()
model_save_dir = "/tmp/adele/"


