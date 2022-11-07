# -*- encoding: utf-8 -*-
import random
import pickle
import SimpleITK as sitk
import numpy as np
import skimage.measure


class OnlineEvaluation:
    def __init__(self,
                 source_dir,
                 dataset_info,
                 ):
        pass

    def __call__(self, network, device):
        # No validation, just pick the model with best train loss
       return -111