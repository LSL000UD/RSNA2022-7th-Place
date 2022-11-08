import pickle
import numpy as np
import os
import sys
import path


# Task Name
task_name = __file__.replace('\\', '/').split('/')[-2]

path_root = path.path_root

# Path
path_nnunet = f"{path_root}/nnUnet"
path_nnunet_raw_data = f"{path_nnunet}/nnUNet_raw_data"
path_nnunet_preprocessing_output_dir = f"{path_nnunet}/nnUNet_preprocessed"
path_nnunet_model_dir = f"{path_nnunet}/Models"

# Pre-Processing
do_pre_processing = 'YES'
assert do_pre_processing == 'YES' or do_pre_processing == 'NO'

# Plan

plan_stage = 0
batch_size = 2
base_num_features = 32
patch_size = np.array([160, 128, 128])
num_pool_per_axis = [5, 5, 5]
pool_op_kernel_sizes = [[2, 2, 2]] * 5
conv_kernel_sizes = [[3, 3, 3]] * 6
do_dummy_2D_data_aug = False


# Training
fold = 'all'
GPU_id = 0
network = '3d_fullres'
network_trainer = 'nnUNetTrainerV2'
previous_plan = None

dataset_directory = f"{path_nnunet_preprocessing_output_dir}/{task_name}"                          # Use previous data
plans_file = f"{path_nnunet_preprocessing_output_dir}/{task_name}/nnUNetPlansv2.1_plans_3D.pkl"    # Use previous plan
batch_dice = True

continue_training = False

default_num_threads = 4
max_num_epochs = 1000
fg_sample_radio = 0.5


optimizer_type = 'SGD'


# ---------------------------- Modify from original nnunet --------------------------- #
# Set environ
os.environ['nnUNet_raw_data_base'] = path_nnunet
os.environ['nnUNet_preprocessed'] = path_nnunet_preprocessing_output_dir
os.environ['RESULTS_FOLDER'] = path_nnunet_model_dir
os.environ['nnUnet_do_pre_processing'] = do_pre_processing
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
os.environ['nnUNet_default_num_threads'] = str(default_num_threads)
os.environ['nnUNet_max_num_epochs'] = str(max_num_epochs)
os.environ['nnUNet_fg_sample_radio'] = str(fg_sample_radio)
os.environ['nnUNet_optimizer_type'] = optimizer_type

print(f"==> nnunet environ:")
nnunet_environ_keys = [
    'nnUNet_raw_data_base',
    'nnUNet_preprocessed',
    'RESULTS_FOLDER',
    'nnUnet_do_pre_processing',
    'CUDA_VISIBLE_DEVICES',
    'nnUNet_default_num_threads',
    'nnUNet_max_num_epochs',
    'nnUNet_fg_sample_radio',
    'nnUNet_optimizer_type'
]
for key in nnunet_environ_keys:
    print(f"        ----> {key:30s}: {os.environ.get(key)}")

# Add src to path
src_path = os.path.abspath(os.path.join(__file__, '../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

