# Dataset Parameters

p = [10]
mnist_digit = [4]
dataset_fraction = [0.2] # fraction of the total dataset to train on
group = ['znz_znz', 'dihedral']

# Learning Parameters
seed = [10]
n_frequencies_to_learn = [6]
hidden_size = 6 * n_frequencies_to_learn
init_scale = [1e-2]
lr, mom = 0.001, 0.9 # originaly 0.01 and 0.9

# Training parameters
epochs = 500 
verbose_interval = 100
batch_size = 128  

