# -*- encoding: utf-8 -*-
import numpy as np
import random
from Utils.CommonTools.bbox import get_bbox, extend_bbox

__all__ = [
           'RandomCrop', 'random_crop',
           'CenterPadToSize', 'center_pad_to_size']


# Random Crop
class RandomCrop:
    def __init__(self, target_size, return_bbox=False, list_target_indexes=None):
        self.target_size = target_size
        self.return_bbox = return_bbox
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return random_crop(list_images,
                           target_size=self.target_size,
                           return_bbox=self.return_bbox,
                           list_target_indexes=self.list_target_indexes
                           )


def random_crop(list_images, target_size, return_bbox=False, list_target_indexes=None):
    """
    :param list_images: list of images, CxZxHxW or CxHxW
    :param target_size: (Z, H, W) or (H, W)
    :param return_bbox: bool, default=False
    :param list_target_indexes: which image to process, default None
            if = [k1, k2], only process list_images[k1, k2]
            if =None, process all images in list_images
    :return: cropped list_images, crop bbox(if return_bbox=True)
    """

    ori_size = list_images[0].shape[1:]
    dims = len(target_size)

    assert dims == 2 or dims == 3, \
        'Only support 2D or 3D image, current target size is {:s}'.format(str(target_size))

    assert dims == len(ori_size), \
        'Target dimension {:d} != ori dimension{:d}'.format(dims, len(ori_size))

    assert not np.any(np.array(ori_size) < np.array(target_size)), \
        'Ori size {:s} is smaller than new size {:s}'.format(str(ori_size), str(target_size))

    crop_bbox = []
    for axis_i in range(dims):
        b_l = random.randint(0, ori_size[axis_i] - target_size[axis_i])
        e_l = b_l + target_size[axis_i] - 1
        crop_bbox += [b_l, e_l]

    if list_target_indexes is None:
        list_target_indexes = range(len(list_images))

    for image_i in list_target_indexes:
        if dims == 3:
            list_images[image_i] = list_images[image_i][:,
                                   crop_bbox[0]:crop_bbox[1] + 1,
                                   crop_bbox[2]:crop_bbox[3] + 1,
                                   crop_bbox[4]:crop_bbox[5] + 1]
        elif dims == 2:
            list_images[image_i] = list_images[image_i][:,
                                   crop_bbox[0]:crop_bbox[1] + 1,
                                   crop_bbox[2]:crop_bbox[3] + 1]

    if return_bbox:
        return list_images, crop_bbox
    else:
        return list_images


# Center Pad
class CenterPadToSize:
    def __init__(self, target_size, list_pad_value=None, return_list_bbox=False, list_target_indexes=None):
        self.target_size = target_size
        self.list_pad_value = list_pad_value
        self.return_list_bbox = return_list_bbox
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return center_pad_to_size(list_images,
                                  target_size=self.target_size,
                                  list_pad_value=self.list_pad_value,
                                  return_list_bbox=self.return_list_bbox,
                                  list_target_indexes=self.list_target_indexes

                                  )


def center_pad_to_size(list_images, target_size, list_pad_value=None, return_list_bbox=False, list_target_indexes=None):
    ori_size = list_images[0].shape[1:]
    dims = len(target_size)

    assert dims == 2 or dims == 3, \
        'Only support 2D or 3D image, current target size is {:s}'.format(str(target_size))

    assert dims == len(ori_size), \
        'Target dimension {:d} != ori dimension{:d}'.format(dims, len(ori_size))

    list_pad_l = []
    list_bbox = []
    for axis_i in range(dims):
        pad_l = target_size[axis_i] - ori_size[axis_i]

        pad_l = max(0, pad_l)
        pad_l_1 = pad_l // 2
        pad_l_2 = pad_l - pad_l_1

        list_pad_l += [pad_l_1, pad_l_2]
        list_bbox += [pad_l_1, pad_l_1 + ori_size[axis_i] - 1]

    # Set pad value
    if list_pad_value is None:
        list_pad_value = [0] * len(list_images)

    # Set process image indexes
    if list_target_indexes is None:
        list_target_indexes = range(len(list_images))

    for image_i in list_target_indexes:
        if dims == 3:
            list_images[image_i] = np.pad(list_images[image_i],
                                          ((0, 0), (list_pad_l[0], list_pad_l[1]), (list_pad_l[2], list_pad_l[3]),
                                           (list_pad_l[4], list_pad_l[5])),
                                          mode='constant',
                                          constant_values=list_pad_value[image_i])
        elif dims == 2:
            list_images[image_i] = np.pad(list_images[image_i],
                                          ((0, 0), (list_pad_l[0], list_pad_l[1]), (list_pad_l[2], list_pad_l[3])),
                                          mode='constant',
                                          constant_values=list_pad_value[image_i])

    if return_list_bbox:
        return list_images, list_bbox
    else:
        return list_images
