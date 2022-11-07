# -*- encoding: utf-8 -*-
import math
import numpy as np


def generate_patch_by_sliding_window(input_size, patch_size, stride):
    dimensions = len(input_size)

    assert dimensions == 2 or dimensions == 3, \
        'Only support 2D or 3D input !'

    assert len(input_size) == len(patch_size) == len(stride), \
        'Dimensions of input_size, patch_size, stride should be the same, current is ({:d}, {:d}, {:d})'.format(
            len(input_size), len(patch_size), len(stride)
        )

    assert np.all(np.array(input_size) >= np.array(patch_size)), \
        'Input size should bigger than output size !'

    list_coords = []
    for d_i in range(dimensions):
        input_l = input_size[d_i]
        output_l = patch_size[d_i]
        stride_l = stride[d_i]

        if input_l == output_l:
            coords = [0]
        else:
            num_step = math.ceil((input_l - output_l) / stride_l) + 1
            new_stride = (input_l - output_l) / (num_step - 1)
            coords = [int((i * new_stride)) for i in range(num_step)]

        list_coords.append(coords)

    output = []
    if dimensions == 2:
        for h_ in list_coords[0]:
            for w_ in list_coords[1]:
                output_h = patch_size[0]
                output_w = patch_size[1]
                output.append([h_, h_ + output_h - 1, w_, w_ + output_w - 1])
    elif dimensions == 3:
        for z_ in list_coords[0]:
            for h_ in list_coords[1]:
                for w_ in list_coords[2]:
                    output_z = patch_size[0]
                    output_h = patch_size[1]
                    output_w = patch_size[2]
                    output.append([z_, z_ + output_z - 1, h_, h_ + output_h - 1, w_, w_ + output_w - 1])
    else:
        raise Exception('Wrong dimensions : {:d} ! '.format(dimensions))

    return output

