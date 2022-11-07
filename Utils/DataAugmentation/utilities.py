# -*- encoding: utf-8 -*-
import numpy as np


__all__ = ['check_dtype_and_get_clip_range']


def check_dtype_and_get_clip_range(dtype, clip_range, support_dtypes):
    assert dtype in support_dtypes, \
        '==> {:s} is not in support dtypes : {:s}'.format(str(dtype), str(support_dtypes))

    if dtype == np.uint8:
        final_clip_range = (0, 255)
    elif dtype == np.int16:
        final_clip_range = (-32768, 32767)
    else:
        final_clip_range = (-1e+8, 1e+8)

    if clip_range is not None:
        final_clip_range = (max(final_clip_range[0], clip_range[0]), min(final_clip_range[1], clip_range[1]))

    return final_clip_range
