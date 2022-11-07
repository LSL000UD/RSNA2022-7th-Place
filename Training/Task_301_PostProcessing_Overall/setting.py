import numpy as np
import path

# Task Name
task_name = __file__.replace('\\', '/').split('/')[-2]

# GPU
list_GPU_ids = [0]

# Path
path_output = f"{path.path_root}/Models"
path_training_data = f"{path.path_root}/TrainingData/{task_name}"
path_training_data_info = f"{path.path_root}/TrainingData/{task_name}.pkl"
print(f"Loading dataset from {path_training_data}")
print(f"Loading dataset info from {path_training_data_info}")
# Model
input_num = 1
in_ch = 2
out_ch = 1
list_ch = [-1, 16, 32, 64, 128]

# Data loader
input_size = (96, 96, 96)  # (Z, Y, X)
train_bs = 4
num_works = 2
prefetch_factor = 2
pin_memory = True
persistent_workers = True


# Training
training_folds = ['no_val']
continue_training = False
continue_with_model_weight_only = False

max_iter = 20000
iter_per_epoch = 500
total_epoch = max_iter // iter_per_epoch
list_save_in_n_iter = (-999, 10000-1, 15000-1)
list_val_epoch = list(np.linspace(0, total_epoch - 1, total_epoch).astype(np.int64))
lrs = [3e-4, 3e-4]

