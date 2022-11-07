# -*- encoding: utf-8 -*-
import numpy as np

try:
    import torch
except:
    pass


def get_bbox(mask, to_slice=False):
    dims = len(mask.shape)

    bbox = [-1, -1] * dims
    for axis_i in np.argsort(mask.shape):
        list_axis = []
        for i in range(0, dims):
            if i != axis_i:
                list_axis.append(i)

        exist_l = np.where(np.any(mask, axis=tuple(list_axis)) > 0)[0]

        if len(exist_l) == 0:
            return None

        b_l = np.min(exist_l)
        e_l = np.max(exist_l)

        bbox[2 * axis_i] = b_l
        bbox[2 * axis_i + 1] = e_l

        list_slicer = []
        for i in list_axis:
            list_slicer.append(slice(0, mask.shape[i], None))
        list_slicer.insert(axis_i, slice(b_l, e_l + 1, None))
        mask = mask[tuple(list_slicer)]

    if to_slice:
        slicer = []
        for i in range(dims):
            slicer.append(slice(bbox[i * 2], bbox[i * 2 + 1] + 1, None))
        return tuple(slicer)
    else:
        return bbox


def extend_bbox(bbox, max_shape, list_extend_length, spacing, approximate_method=np.ceil):
    """
    :param approximate_method: np.ceil or np.round or np.int
    :param bbox: input bbox
    :param max_shape: in case bbox out of image after extending
    :param list_extend_length: length of each side to extend
    :param spacing: extend_length/spacing = real extend pixel
    :return:
    """
    dimensions = len(max_shape)

    extended_bbox = []
    for i in range(dimensions):
        extend_pixel = np.int(approximate_method(list_extend_length[i] / spacing[i]))

        extended_bbox.append(max(0, bbox[2 * i] - extend_pixel))
        extended_bbox.append(min(max_shape[i]-1, bbox[2 * i + 1] + extend_pixel))

    return extended_bbox