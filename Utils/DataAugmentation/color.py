# -*- encoding: utf-8 -*-
import cv2
import torchvision.transforms as TF
from PIL import Image as PILImage
import random
import numpy as np
from .utilities import check_dtype_and_get_clip_range

__all__ = ['RandomBrightnessAddictiveGray', 'random_brightness_addictive_gray',
           'RandomContrastGray', 'random_contrast_gray',
           'RandomGammaGray', 'random_gamma_gray',
           ]


class RandomBrightnessAddictiveGray:
    def __init__(self,
                 mu,
                 sigma,
                 shift_clip_range=None,
                 shift_distribution='uniform',
                 clip_range=None,
                 list_target_indexes=None,
                 same_shift_for_all_images=True,
                 return_shift=False,
                 p=0.5):
        self.mu = mu
        self.sigma = sigma
        self.shift_clip_range = shift_clip_range
        self.shift_distribution = shift_distribution
        self.clip_range = clip_range
        self.list_target_indexes = list_target_indexes
        self.same_shift_for_all_images = same_shift_for_all_images
        self.return_shift = return_shift
        self.p = p

    def __call__(self, list_images):
        return random_brightness_addictive_gray(list_images,
                                                self.mu,
                                                self.sigma,
                                                self.shift_clip_range,
                                                self.shift_distribution,
                                                self.clip_range,
                                                self.list_target_indexes,
                                                self.same_shift_for_all_images,
                                                self.return_shift,
                                                self.p
                                                )


def random_brightness_addictive_gray(list_images,
                                     mu,
                                     sigma,
                                     shift_clip_range=None,
                                     shift_distribution='uniform',
                                     clip_range=None,
                                     list_target_indexes=None,
                                     same_shift_for_all_images=True,
                                     return_shift=False,
                                     p=0.5
                                     ):
    list_shift = []
    if np.random.random() <= p:
        if list_target_indexes is None:
            list_target_indexes = range(len(list_images))

        # Get shift of brightness
        shift = None
        for image_i in list_target_indexes:
            if (shift is None) or (not same_shift_for_all_images):
                if shift_distribution == 'uniform':
                    shift = random.uniform(-sigma, sigma)
                elif shift_distribution == 'norm':
                    shift = np.random.normal(mu, sigma)
                else:
                    raise ValueError('Unknown shift distribution type')

                if shift_clip_range is not None:
                    shift = min(shift_clip_range[1], max(shift, shift_clip_range[0]))

            list_shift.append(shift)

            # Do shifting
            ori_dtype = list_images[image_i].dtype
            final_clip_range = check_dtype_and_get_clip_range(
                ori_dtype,
                clip_range,
                support_dtypes=(np.uint8, np.float16, np.int16, np.float32)
            )

            # To float32 first to prevent numerical overflow
            list_images[image_i] = np.float32(list_images[image_i]) + shift
            list_images[image_i] = np.clip(list_images[image_i], a_min=final_clip_range[0], a_max=final_clip_range[1])
            list_images[image_i] = list_images[image_i].astype(ori_dtype)

    if return_shift:
        return list_images, list_shift
    else:
        return list_images


class RandomContrastGray:
    def __init__(self,
                 contrast_range=(0.75, 1.25),
                 clip_range=None,
                 list_target_indexes=None,
                 same_factor_for_all_images=True,
                 return_factor=False,
                 p=0.5):
        self.contrast_range = contrast_range
        self.clip_range = clip_range
        self.list_target_indexes = list_target_indexes
        self.same_factor_for_all_images = same_factor_for_all_images
        self.return_factor = return_factor
        self.p = p

    def __call__(self, list_images):
        return random_contrast_gray(list_images,
                                    contrast_range=self.contrast_range,
                                    clip_range=self.clip_range,
                                    list_target_indexes=self.list_target_indexes,
                                    same_factor_for_all_images=self.same_factor_for_all_images,
                                    return_factor=self.return_factor,
                                    p=self.p
                                    )


def random_contrast_gray(list_images,
                         contrast_range=(0.75, 1.25),
                         clip_range=None,
                         list_target_indexes=None,
                         same_factor_for_all_images=True,
                         return_factor=False,
                         p=0.5):
    list_factor = []
    if np.random.random() <= p:
        if list_target_indexes is None:
            list_target_indexes = range(len(list_images))

        factor = None
        for image_i in list_target_indexes:
            if (factor is None) or (not same_factor_for_all_images):
                if contrast_range[0] < 1 and contrast_range[1] > 1:
                    if np.random.random() < 0.5:
                        factor = np.random.uniform(contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(1, contrast_range[1])
                else:
                    factor = np.random.uniform(contrast_range[0], contrast_range[1])

            list_factor.append(factor)

            # Do contrast
            ori_dtype = list_images[image_i].dtype
            final_clip_range = check_dtype_and_get_clip_range(
                ori_dtype,
                clip_range,
                support_dtypes=(np.uint8, np.float16, np.int16, np.float32)
            )

            mean_ = np.mean(list_images[image_i])
            image = np.float32(list_images[image_i])

            list_images[image_i] = (image - mean_) * factor + mean_

            list_images[image_i] = np.clip(list_images[image_i], a_min=final_clip_range[0], a_max=final_clip_range[1])
            list_images[image_i] = list_images[image_i].astype(ori_dtype)

    if return_factor:
        return list_images, list_factor
    else:
        return list_images


class RandomGammaGray:
    def __init__(self,
                 gamma_range=(0.75, 1.25),
                 clip_range=None,
                 list_target_indexes=None,
                 same_gamma_for_all_images=True,
                 return_gamma=False,
                 p=0.5):
        self.gamma_range = gamma_range
        self.clip_range = clip_range
        self.list_target_indexes = list_target_indexes
        self.same_gamma_for_all_images = same_gamma_for_all_images
        self.return_gamma = return_gamma
        self.p = p

    def __call__(self, list_images):
        return random_gamma_gray(list_images,
                                 gamma_range=self.gamma_range,
                                 clip_range=self.clip_range,
                                 list_target_indexes=self.list_target_indexes,
                                 same_gamma_for_all_images=self.same_gamma_for_all_images,
                                 return_gamma=self.return_gamma,
                                 p=self.p
                                 )


def random_gamma_gray(list_images,
                      gamma_range=(0.5, 2),
                      clip_range=None,
                      list_target_indexes=None,
                      same_gamma_for_all_images=True,
                      return_gamma=False,
                      p=0.5):
    list_gamma = []
    if np.random.random() <= p:
        if list_target_indexes is None:
            list_target_indexes = range(len(list_images))

        gamma = None
        for image_i in list_target_indexes:
            if (gamma is None) or (not same_gamma_for_all_images):
                if gamma_range[0] < 1 and gamma_range[1] > 1:
                    if np.random.random() < 0.5:
                        gamma = np.random.uniform(gamma_range[0], 1)
                    else:
                        gamma = np.random.uniform(1, gamma_range[1])
                else:
                    gamma = np.random.uniform(gamma_range[0], gamma_range[1])

            list_gamma.append(gamma)

            # Do gamma
            ori_dtype = list_images[image_i].dtype
            final_clip_range = check_dtype_and_get_clip_range(ori_dtype,
                                                              clip_range,
                                                              support_dtypes=(
                                                                  np.uint8, np.float16, np.int16, np.float32))

            list_images[image_i] = np.float32(list_images[image_i])

            min_v = np.min(list_images[image_i])
            max_v = np.max(list_images[image_i])

            range_v = max_v - min_v
            list_images[image_i] = (((list_images[image_i] - min_v) / float(range_v + 1e-7)) ** gamma) * range_v + min_v

            list_images[image_i] = np.clip(list_images[image_i], a_min=final_clip_range[0], a_max=final_clip_range[1])
            list_images[image_i] = list_images[image_i].astype(ori_dtype)

    if return_gamma:
        return list_images, list_gamma
    else:
        return list_images
