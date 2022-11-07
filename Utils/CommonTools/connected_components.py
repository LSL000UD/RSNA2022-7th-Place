# -*- encoding: utf-8 -*-
import skimage.measure
import numpy as np
from Utils.CommonTools.bbox import get_bbox


def get_top_n_max_connected_components_3D(mask, top_n, merge_top_n=False):
    # Crop bbox for speeding
    bbox = get_bbox(mask)
    if bbox is None:
        return None

    roi_bz, roi_ez, roi_by, roi_ey, roi_bx, roi_ex = bbox[:]
    patch_mask = mask[roi_bz:roi_ez+1, roi_by:roi_ey+1, roi_bx:roi_ex+1]

    # Top n props
    instance_label = skimage.measure.label(patch_mask)
    # instance_label = cc3d.connected_components(patch_mask)   # This is much faster
    props = skimage.measure.regionprops(instance_label)
    list_area = []
    for prop in props:
        list_area.append(prop.area)
    sorted_prop_indexes = list(np.argsort(list_area)[::-1])

    list_mask = []
    list_bbox = []
    list_size = []
    if merge_top_n:
        merge_mask = np.zeros(mask.shape, np.uint8)

    for index_i in range(min(top_n, len(props))):
        prop = props[sorted_prop_indexes[index_i]]

        bbox = prop.bbox
        bbox = [bbox[0], bbox[3]-1, bbox[1], bbox[4]-1, bbox[2], bbox[5]-1]
        bz, ez, by, ey, bx, ex = bbox[:]

        list_bbox.append([bz+roi_bz, ez+roi_bz, by+roi_by, ey+roi_by, bx+roi_bx, ex+roi_bx])
        list_size.append(prop.area)

        cur_mask = np.zeros(mask.shape, np.uint8)
        cur_mask[roi_bz+bz:roi_bz+ez+1, roi_by+by:roi_by+ey+1, roi_bx+bx:roi_bx+ex+1][instance_label[bz:ez+1, by:ey+1, bx:ex+1] == prop.label] = 1
        list_mask.append(cur_mask)

        if merge_top_n:
            merge_mask[roi_bz+bz:roi_bz+ez+1, roi_by+by:roi_by+ey+1, roi_bx+bx:roi_bx+ex+1][instance_label[bz:ez+1, by:ey+1, bx:ex+1] == prop.label] = 1

    if merge_top_n:
        return list_mask, merge_mask, list_bbox, list_size
    else:
        return list_mask, list_bbox, list_size


def get_largest_cc(mask):
    output = get_top_n_max_connected_components_3D(mask, 1, merge_top_n=False)
    if output is None:
        return None
    else:
        list_mask, list_bbox, list_size = output
        return list_mask[0], list_bbox[0], list_size[0]


def keep_largest_cc_binary(mask):
    final_output = np.zeros(mask.shape, np.bool_)

    # Crop bbox for speeding
    mask = mask > 0
    bbox = get_bbox(mask)
    if bbox is None:
        return final_output

    roi_bz, roi_ez, roi_by, roi_ey, roi_bx, roi_ex = bbox[:]
    patch_mask = mask[roi_bz:roi_ez + 1, roi_by:roi_ey + 1, roi_bx:roi_ex + 1]

    instance_label = skimage.measure.label(patch_mask)
    props = skimage.measure.regionprops(instance_label)

    # Sort by area
    list_area = []
    for prop in props:
        list_area.append(prop.area)
    sorted_prop_indexes = list(np.argsort(list_area)[::-1])

    final_output[roi_bz:roi_ez+1, roi_by:roi_ey+1, roi_bx:roi_ex+1] = instance_label == props[sorted_prop_indexes[0]].label

    return final_output


def keep_largest_cc(mask):
    final_output = np.zeros_like(mask)

    # Crop bbox for speeding
    bbox = get_bbox(mask > 0)
    if bbox is None:
        return final_output

    roi_bz, roi_ez, roi_by, roi_ey, roi_bx, roi_ex = bbox[:]
    patch_mask = mask[roi_bz:roi_ez + 1, roi_by:roi_ey + 1, roi_bx:roi_ex + 1]

    # Keep largest cc for each index
    list_unique_indexes = list(np.unique(patch_mask))
    if 0 in list_unique_indexes:
        list_unique_indexes.remove(0)

    for index in list_unique_indexes:
        largest_cc_binary = keep_largest_cc_binary(patch_mask == index)
        final_output[roi_bz:roi_ez+1, roi_by:roi_ey+1, roi_bx:roi_ex+1][largest_cc_binary > 0] = index

    return final_output