import os

# ---------------------------- Modify from original nnunet --------------------------- #
default_num_threads = int(os.environ.get('nnUNet_default_num_threads'))
# default_num_threads = 4
# ---------------------------- Modify from original nnunet --------------------------- #


RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis
# separately (with NN)