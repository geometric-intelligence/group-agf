# Dataset Parameters
import os

dataset_fraction = [0.3, 0.4] # fraction of the total dataset to train on
group_name = ['dihedral'] #, 'dihedral', 'znz_znz']
mnist_digit = [4]
dihedral_order_n = [4]  # n in Dn [3, 4, 5]
image_length = [5, 10, 15] # length of one side of the square image patch
signal_length_1d = [6] # only works for 6 rn, because only d3 implemented

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


