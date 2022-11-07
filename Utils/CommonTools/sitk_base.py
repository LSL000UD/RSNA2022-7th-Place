# -*- encoding: utf-8 -*-
import math
import time

import SimpleITK as sitk
import numpy as np


def copy_nii_info(source_image, target_image):
    target_image.SetSpacing(source_image.GetSpacing())
    target_image.SetDirection(source_image.GetDirection())
    target_image.SetOrigin(source_image.GetOrigin())

    return target_image


def get_nii_info(image_nii):
    image_info = {
        'spacing': image_nii.GetSpacing(),
        'direction': image_nii.GetDirection(),
        'origin': image_nii.GetOrigin(),
        'size':image_nii.GetSize()

    }
    return image_info


def set_nii_info(source_image, image_info):
    source_image.SetSpacing(image_info['spacing'])
    source_image.SetDirection(image_info['direction'])
    source_image.SetOrigin(image_info['origin'])

    return source_image


# Todo : find filters in sitk to replace all shit below
def sitk_get_affine_matrix(direction, origin, spacing):
    affine_matrix = np.zeros((4, 4), np.float32)
    affine_matrix[0:3, 0:3] = np.reshape(direction, [3, 3])
    affine_matrix[0:3, 3] = origin[:]
    affine_matrix[3, :] = [0, 0, 0, 1]

    spacing_matrix = np.zeros((4, 4), np.float32)
    spacing_matrix[0, 0] = spacing[0]
    spacing_matrix[1, 1] = spacing[1]
    spacing_matrix[2, 2] = spacing[2]
    spacing_matrix[3, 3] = 1

    affine_matrix = np.array(np.matrix.dot(affine_matrix, spacing_matrix))

    return affine_matrix


def sitk_location2coordinate(location, affine_matrix):
    return list(np.matrix.dot(affine_matrix, np.reshape(location, (4, 1))).reshape([4]))


# todo : need check, maybe this has bug when input non-LPS data
def get_origin_from_center(center_origin, affine_matrix, new_size, new_spacing):
    simple_direction = np.array(sitk_location2coordinate([1, 1, 1, 1], affine_matrix)) - np.array(
        sitk_location2coordinate([0, 0, 0, 1], affine_matrix))
    simple_direction = simple_direction / (np.abs(simple_direction) + 1e-8)
    new_origin = np.array(center_origin) - np.array([simple_direction[0] * (new_size[0] / 2) * new_spacing[0],
                                                     simple_direction[1] * (new_size[1] / 2) * new_spacing[1],
                                                     simple_direction[2] * (new_size[2] / 2) * new_spacing[2],
                                                     ])

    return list(new_origin)


def resample(image,
             new_spacing,
             new_origin=None,
             new_size=None,
             new_direction=None,
             center_origin=None,
             interp=sitk.sitkNearestNeighbor,
             dtype=sitk.sitkInt16,
             constant_value=-1024):

    ori_spacing = image.GetSpacing()
    ori_origin = image.GetOrigin()
    ori_dircection = image.GetDirection()
    ori_size = image.GetSize()

    if new_direction is None:
        new_direction = ori_dircection

    if new_size is None:
        new_size = [math.ceil(ori_spacing[i] * ori_size[i] / new_spacing[i]) for i in range(3)]

    if center_origin is None:
        if new_origin is None:
            new_origin = ori_origin
    else:
        affine_matrix = sitk_get_affine_matrix(ori_dircection,
                                               ori_origin,
                                               ori_spacing)

        if center_origin == 'auto':
            center_origin = image[ori_size[0] // 2:ori_size[0] // 2 + 1,
                            ori_size[1] // 2:ori_size[1] // 2 + 1,
                            ori_size[2] // 2:ori_size[2] // 2 + 1].GetOrigin()

        new_origin = get_origin_from_center(center_origin, affine_matrix, new_size, new_spacing)

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetInterpolator(interp)
    resample_filter.SetOutputOrigin(new_origin)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetOutputDirection(new_direction)
    resample_filter.SetDefaultPixelValue(constant_value)
    resample_filter.SetOutputPixelType(dtype)
    resampled_image = resample_filter.Execute(image)
    return resampled_image
