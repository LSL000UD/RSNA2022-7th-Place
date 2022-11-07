import numpy as np
import torch
import torch.nn.functional as F

import skimage.morphology

from Utils.CommonTools.connected_components import keep_largest_cc_binary


def keep_largest_cervical_cc(seg, image_spacing):
    output = seg.copy()
    b_mask = np.uint8(seg > 0)

    # Down sampling for faster processing
    target_spacing = (2.0, 2.0, 2.0)
    scale_factor = [image_spacing[k] / target_spacing[k] for k in range(3)]

    # Resampling using torch
    b_mask = torch.from_numpy(b_mask[np.newaxis, np.newaxis])
    b_mask = F.interpolate(b_mask, size=None, scale_factor=scale_factor, mode='nearest')
    b_mask = b_mask.numpy()[0, 0]

    for i in range(3):
        b_mask = skimage.morphology.binary_dilation(b_mask)

    # Resampling back
    b_mask = torch.from_numpy(np.uint8(b_mask[np.newaxis, np.newaxis]))
    b_mask = F.interpolate(b_mask, size=output.shape, scale_factor=None, mode='nearest')
    b_mask = b_mask.numpy()[0, 0]
    b_mask = keep_largest_cc_binary(b_mask)

    output[b_mask < 1] = 0

    return output
