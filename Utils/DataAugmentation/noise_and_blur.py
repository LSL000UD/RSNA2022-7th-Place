# -*- encoding: utf-8 -*-
import cv2
import random
import numpy as np
from Utils.DataAugmentation.utilities import check_dtype_and_get_clip_range


__all__ = ['RandomGaussianNoise', 'random_gaussian_noise']


# Gaussian Noise
class RandomGaussianNoise:
    def __init__(self,
                 mean=0,
                 var_range=(0, 0.01),
                 p=0.5,
                 clip_range=None,
                 return_var=False,
                 list_target_indexes=None):
        self.mean = mean
        self.var_range = var_range
        self.p = p
        self.return_var = return_var
        self.clip_range = clip_range
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return random_gaussian_noise(list_images,
                                     mean=self.mean,
                                     var_range=self.var_range,
                                     p=self.p,
                                     clip_range=self.clip_range,
                                     return_var=self.return_var,
                                     list_target_indexes=self.list_target_indexes
                                     )


def random_gaussian_noise(list_images, mean=0, var_range=(0, 0.01), clip_range=None, list_target_indexes=None,
                          return_var=False, p=0.5):

    var = 0
    if random.random() <= p:
        var = random.uniform(var_range[0], var_range[1])
        noise = np.random.normal(mean, var, size=list_images[0].shape).astype(np.float32)

        if list_target_indexes is None:
            list_target_indexes = range(len(list_images))

        for image_i in list_target_indexes:
            ori_dtype = list_images[image_i].dtype
            final_clip_range = check_dtype_and_get_clip_range(
                ori_dtype,
                clip_range,
                support_dtypes=(np.uint8, np.float16, np.int16, np.float32)
            )

            list_images[image_i] = np.float32(list_images[image_i])
            list_images[image_i] = list_images[image_i] + noise

            list_images[image_i] = np.clip(list_images[image_i], a_min=final_clip_range[0], a_max=final_clip_range[1])
            list_images[image_i] = list_images[image_i].astype(ori_dtype)

    if return_var:
        return list_images, var
    else:
        return list_images
