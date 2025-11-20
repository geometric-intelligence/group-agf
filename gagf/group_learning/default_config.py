# Dataset Parameters
group_name = ['dihedral', 'octahedral'] #, 'cyclic', 'dihedral', 'znz_znz']
group_n = [6]  # n in Dn [3, 4, 5]
image_length = [5] #, 10, 15] # length of one side of the square image patch

# Model Parameters
hidden_size = [None] #, 10, 15] # number of hidden units in the neural network

# Learning Parameters
seed = [10, 20] #, 30, 40, 50] #, 60, 70, 80, 90, 100]
init_scale = [1e-3, 1e-4, 1e-5]  # originally 1e-2
lr = [0.01, 0.001] #, 0.001]  # originaly 0.01
mom = [0.9] # originaly 0.9
optimizer_name = ['PerNeuronScaledSGD'] #, 'SGD' Adam', "Custom'"]

# Training parameters
epochs = [10000] #, 20000, 30000] #, 10000] #, 10000, 20000, 30000] 
verbose_interval = [100]
batch_size = [128] #    , 256] 

# znz_znz specific parameters
mnist_digit = [4]
frequencies_to_learn = [3, 6, 9]  # number of frequencies to learn in the template
dataset_fraction = [1.0] #, 0.4] # fraction of the total dataset to train on

# git_root_path = setcwd.get_root_dir()
model_save_dir = "/tmp/nmiolane/"


