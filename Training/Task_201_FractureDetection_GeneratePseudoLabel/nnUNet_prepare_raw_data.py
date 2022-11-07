import json
from collections import OrderedDict
import skimage.morphology
import skimage.measure
import pandas as pd
import skimage.morphology

from Utils.post_processing import keep_largest_cervical_cc
from Utils.CommonTools.sitk_base import resample
from Utils.CommonTools.dir import try_recursive_mkdir
from Utils.CommonTools.bbox import get_bbox, extend_bbox

import SimpleITK as sitk
import os
import numpy as np

import path
import setting


def get_dataset():
    image_dir = f"{path.path_root}/CompetitionData/Image"
    seg_dir = f"{path.path_root}/CompetitionData/PredictedSegmentation"
    bbox_dir = f"{path.path_root}/CompetitionData/Bbox"

    save_dir = f'{setting.path_nnunet_raw_data}/{setting.task_name}'
    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    try_recursive_mkdir(save_dir_image)
    try_recursive_mkdir(save_dir_label)

    new_spacing = (0.8, 0.4, 0.4)
    extend_length = (5.0, 5.0, 5.0)

    count = 0
    for file in os.listdir(bbox_dir):
        case_id = file.split('.nii.gz')[0]

        # if case_id.find('1.2.826.0.1.3680043.31370') == -1:
        #     continue

        count += 1
        print(f"==> {count}: Processing {case_id}")

        # Reading
        print(f"        ----> Reading")
        image_nii = sitk.ReadImage(f"{image_dir}/{case_id}.nii.gz")
        bbox_nii = sitk.ReadImage(f"{bbox_dir}/{case_id}.nii.gz")
        seg_nii = sitk.ReadImage(f"{seg_dir}/{case_id}.nii.gz")

        image_spacing = image_nii.GetSpacing()[::-1]  # (Z, Y, X)

        # Cropping
        print(f"        ----> Cropping")
        # Crop ROI, extend by 5mm
        seg = sitk.GetArrayFromImage(seg_nii)
        seg = keep_largest_cervical_cc(seg, image_spacing)
        seg_bbox = get_bbox(np.logical_and(seg >= 1, seg <= 7))
        seg_bbox = extend_bbox(seg_bbox, max_shape=seg.shape, list_extend_length=extend_length, spacing=image_spacing,
                               approximate_method=np.ceil)
        bz, ez, by, ey, bx, ex = seg_bbox

        image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
        bbox_nii = bbox_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
        seg_nii = seg_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

        # Resampling
        print(f"        ----> Resampling")
        image_nii = resample(
            image_nii,
            new_spacing=new_spacing[::-1],
            new_origin=None,
            new_size=None,
            new_direction=None,
            center_origin=None,
            interp=sitk.sitkNearestNeighbor,
            dtype=sitk.sitkInt16,
            constant_value=-1024
        )
        seg_nii = resample(
            seg_nii,
            new_spacing=image_nii.GetSpacing(),
            new_origin=image_nii.GetOrigin(),
            new_size=image_nii.GetSize(),
            new_direction=image_nii.GetDirection(),
            center_origin=None,
            interp=sitk.sitkNearestNeighbor,
            dtype=sitk.sitkUInt8,
            constant_value=0
        )
        bbox_nii = resample(
            bbox_nii,
            new_spacing=image_nii.GetSpacing(),
            new_origin=image_nii.GetOrigin(),
            new_size=image_nii.GetSize(),
            new_direction=image_nii.GetDirection(),
            center_origin=None,
            interp=sitk.sitkNearestNeighbor,
            dtype=sitk.sitkUInt8,
            constant_value=0
        )

        # Processing Bbox
        print(f"        ----> Removing bbox out of roi mask")
        bbox = sitk.GetArrayFromImage(bbox_nii)
        seg = sitk.GetArrayFromImage(seg_nii)
        roi_mask = np.logical_and(seg >= 1, seg <= 7)
        roi_mask = skimage.morphology.dilation(roi_mask)
        bbox[roi_mask < 1] = 0
        bbox_nii = sitk.GetImageFromArray(bbox)
        bbox_nii.SetOrigin(image_nii.GetOrigin())
        bbox_nii.SetSpacing(image_nii.GetSpacing())
        bbox_nii.SetDirection(image_nii.GetDirection())

        # Saving
        print(f"        ----> Saving")
        sitk.WriteImage(image_nii, f"{save_dir_image}/{case_id}_0000.nii.gz")
        sitk.WriteImage(seg_nii, f"{save_dir_image}/{case_id}_0001.nii.gz")
        sitk.WriteImage(bbox_nii, f"{save_dir_label}/{case_id}.nii.gz")


def get_dataset_info():
    save_dir = f'{setting.path_nnunet_raw_data}/{setting.task_name}'
    save_dir_label = f"{save_dir}/labelsTr"

    total_class = 2

    # Train case ids
    list_train_ids = []
    for file in os.listdir(save_dir_label):
        if file.find('.nii.gz') > -1:
            list_train_ids.append(file.split('.nii.gz')[0])
    print(f"==> Total {len(list_train_ids)} train cases")

    label_names = {str(k): k for k in range(total_class)}
    print(label_names)

    # Json
    json_dict = OrderedDict()
    json_dict['name'] = setting.task_name
    json_dict['description'] = "fracture segmentation "
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "None"
    json_dict['licence'] = "None"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
        "1": "noNorm"
    }

    json_dict['labels'] = label_names

    json_dict['numTraining'] = len(list_train_ids)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             list_train_ids]

    # No special test
    json_dict['test'] = []

    with open(os.path.join(save_dir, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)


def main():
    get_dataset()
    get_dataset_info()


if __name__ == '__main__':
    main()
