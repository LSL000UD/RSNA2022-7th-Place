# -*- encoding: utf-8 -*-
import numpy as np
import random
import cv2
import SimpleITK as sitk
import warnings

__all__ = ['RandomFlip', 'random_flip',
           'RandomDummyRotateAndScale3d', 'random_dummy_rotate_and_scale_3d']


class RandomFlip:
    def __init__(self, list_p_per_axis=(0., 0.5, 0.5, 0.5), p=0.5, list_target_indexes=None):
        if list_p_per_axis[0] > 0:
            warnings.warn(
                'list_p_per_axis[0] is {:3.2f} while this axis often be the channel'.format(list_p_per_axis[0]))

        self.list_p_per_axis = list_p_per_axis
        self.p = p
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return random_flip(list_images,
                           list_p_per_axis=self.list_p_per_axis,
                           p=self.p,
                           list_target_indexes=self.list_target_indexes
                           )


def random_flip(list_images, list_p_per_axis=(0, 0.5, 0.5, 0.5), p=0.5, list_target_indexes=None):
    """
    :param list_images:
    :param list_p_per_axis: probabilities of (C, Z, H, W), or (C, H, W)
    :param p:overall probability of flip
    :param list_target_indexes:
    :return:
    """
    if random.random() <= p:
        assert len(list_p_per_axis) == len(list_images[0].shape), \
            'Image shape {:s} need {:d} p_per_axis'.format(str(list_images[0].shape), len(list_images[0].shape))
        if list_p_per_axis[0] > 0:
            warnings.warn(
                'list_p_per_axis[0] is {:3.2f} while this axis often be the channel'.format(list_p_per_axis[0]))

        if list_target_indexes is None:
            list_target_indexes = range(len(list_images))

        list_flip_axis = []
        for axis_i in range(len(list_p_per_axis)):
            if random.random() <= list_p_per_axis[axis_i]:
                list_flip_axis.append(axis_i)

        for image_i in list_target_indexes:
            list_images[image_i] = np.flip(list_images[image_i], list_flip_axis)

    return list_images


# Dummy rotate and scale 3D using cv2
class RandomDummyRotateAndScale3d:
    def __init__(self, list_interp,
                 list_constant_value,
                 list_specific_angle=None,
                 angle_range=(0, 360),
                 list_specific_scale=None,
                 scale_range=(0.75, 1.25),
                 p=0.5,
                 list_target_indexes=None):
        self.list_interp = list_interp
        self.list_constant_value = list_constant_value
        self.list_specific_angle = list_specific_angle
        self.angle_range = angle_range
        self.list_specific_scale = list_specific_scale
        self.scale_range = scale_range
        self.p = p
        self.list_target_indexes = list_target_indexes

    def __call__(self, list_images):
        return random_dummy_rotate_and_scale_3d(list_images,
                                                list_interp=self.list_interp,
                                                list_constant_value=self.list_constant_value,
                                                list_specific_angle=self.list_specific_angle,
                                                angle_range=self.angle_range,
                                                list_specific_scale=self.list_specific_scale,
                                                scale_range=self.scale_range,
                                                p=self.p,
                                                list_target_indexes=self.list_target_indexes
                                                )


# Dummy rotate and scale 3D using cv2
def random_dummy_rotate_and_scale_3d(list_images,
                                     list_interp,
                                     list_constant_value,
                                     list_specific_angle=None,
                                     angle_range=(0, 360),
                                     list_specific_scale=None,
                                     scale_range=(0.75, 1.25),
                                     p=0.5,
                                     list_target_indexes=None):
    if random.random() <= p:
        # Randomly choose scale factor and angle
        if list_specific_angle is None:
            _angle = random.uniform(angle_range[0], angle_range[1])
        else:
            _angle = random.sample(list_specific_angle, 1)[0]

        if list_specific_scale is None:
            _scale = random.uniform(scale_range[0], scale_range[1])
        else:
            _scale = random.sample(list_specific_scale, 1)[0]

        # Rotate and scale
        if list_target_indexes is None:
            list_target_indexes = range(len(list_images))

        for image_i in list_target_indexes:
            _dummy_rotate_and_scale_3d(list_images[image_i],
                                       angle=_angle,
                                       scale=_scale,
                                       interp=list_interp[image_i],
                                       constant_value=list_constant_value[image_i])

    return list_images


def _dummy_rotate_and_scale_3d(image, angle, scale, interp, constant_value):
    for chan_i in range(image.shape[0]):
        for slice_i in range(image.shape[1]):
            rows, cols = image[chan_i, slice_i, :, :].shape
            rotation_matrix = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, scale=scale)
            image[chan_i, slice_i, :, :] = cv2.warpAffine(image[chan_i, slice_i, :, :],
                                                          rotation_matrix,
                                                          (cols, rows),
                                                          borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=constant_value,
                                                          flags=interp)
