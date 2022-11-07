# -*- encoding: utf-8 -*-
import torch
import numpy as np
import random


__all__ = [
           'ToTensor', 'to_tensor',
           'ToDtype', 'to_dtype',
           'clip',
           'RandomChooseAug',
           'CompositeAug']


class ToTensor:
    def __init__(self, list_target_indexes=None):
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return to_tensor(list_images, list_target_indexes=self.list_target_indexes)


def to_tensor(list_images, list_target_indexes=None):
    """
    Trans numpy arrays to torch tensor, image should be (C, H, W) or (C, Z, H, W)
    """
    if list_target_indexes is None:
        list_target_indexes = range(len(list_images))

    for image_i in list_target_indexes:
        list_images[image_i] = torch.from_numpy(list_images[image_i].copy())
    return list_images


class ToDtype:
    def __init__(self, list_dtypes, list_target_indexes=None):
        self.list_dtypes = list_dtypes
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return to_dtype(list_images,
                        list_dtypes=self.list_dtypes,
                        list_target_indexes=self.list_target_indexes
                        )


def to_dtype(list_images, list_dtypes, list_target_indexes=None):
    if list_target_indexes is None:
        list_target_indexes = range(len(list_images))

    for image_i in list_target_indexes:
        list_images[image_i] = list_images[image_i].astype(list_dtypes[image_i])
    return list_images


def clip(list_images, list_ranges, list_target_indexes=None):
    if list_target_indexes is None:
        list_target_indexes = range(len(list_images))

    for image_i in list_target_indexes:
        range_ = list_ranges[image_i]
        list_images[image_i] = np.clip(list_images[image_i], a_min=range_[0], a_max=range_[1])
    return list_images


class CompositeAug:
    def __init__(self, *args):
        self.list_augs = args

    def __call__(self, list_images):
        for i in range(len(self.list_augs)):
            list_images = self.list_augs[i](list_images)

        return list_images


class RandomChooseAug:
    def __init__(self, list_augs, list_weights):
        self.list_augs = list_augs
        self.list_weights = list_weights

        self.cumsum_weight = np.cumsum(list_weights)
        self.cumsum_weight /= self.cumsum_weight[-1]

        assert len(list_augs) == len(list_weights), \
            'len(list_augs) != len(list_weights)'

    def __call__(self, list_images):
        rand_num = random.random()

        aug_index = np.where(self.cumsum_weight >= rand_num)[0][0]
        list_images = self.list_augs[aug_index](list_images)

        return list_images