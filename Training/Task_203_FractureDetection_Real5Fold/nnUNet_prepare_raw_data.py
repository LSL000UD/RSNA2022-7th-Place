import json
from collections import OrderedDict
import skimage.morphology
import skimage.measure
import pandas as pd
import SimpleITK as sitk
import os
import pickle
import numpy as np

from Utils.CommonTools.sitk_base import resample
from Utils.CommonTools.dir import try_recursive_mkdir
from Utils.post_processing import keep_largest_cervical_cc
from Utils.CommonTools.bbox import get_bbox, extend_bbox

import path
import setting


def get_dataset():
    image_dir = f"{path.path_root}/CompetitionData/Image"
    seg_dir = f"{path.path_root}/CompetitionData/PredictedSegmentation"
    fracture_dir = f"{path.path_root}/CompetitionData/PredictedFractureV2"

    train_csv = pd.read_csv(f"{path.path_competition_data}/train.csv")

    save_dir = f'{setting.path_nnunet_raw_data}/{setting.task_name}'
    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    try_recursive_mkdir(save_dir_image)
    try_recursive_mkdir(save_dir_label)

    new_spacing = (0.8, 0.4, 0.4)
    extend_length = (5.0, 5.0, 5.0)
    threshold_fracture = 0.2

    count = 0
    for file in os.listdir(fracture_dir):
        case_id = file.split('.nii.gz')[0]
        meta_info_index = list(train_csv['StudyInstanceUID']).index(case_id)

        # Only use positive samples
        if not int(train_csv[f"patient_overall"][meta_info_index]):
            continue

        count += 1
        print(f"==> {count}: Processing {case_id}")

        # Reading
        print(f"        ----> Reading")
        image_nii = sitk.ReadImage(f"{image_dir}/{case_id}.nii.gz")
        fracture_nii = sitk.ReadImage(f"{fracture_dir}/{case_id}.nii.gz")
        seg_nii = sitk.ReadImage(f"{seg_dir}/{case_id}.nii.gz")

        image_spacing = image_nii.GetSpacing()[::-1]  # (Z, Y, X)

        # Cropping
        print(f"        ----> Cropping")
        # Crop ROI, extend by 5mm
        seg = sitk.GetArrayFromImage(seg_nii)
        seg = keep_largest_cervical_cc(seg, image_spacing)
        seg_bbox = get_bbox(np.logical_and(seg >= 1, seg <= 7))
        if seg_bbox is None:
            continue
        seg_bbox = extend_bbox(seg_bbox, max_shape=seg.shape, list_extend_length=extend_length, spacing=image_spacing,
                               approximate_method=np.ceil)
        bz, ez, by, ey, bx, ex = seg_bbox

        image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
        fracture_nii = fracture_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
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
        fracture_nii = resample(
            fracture_nii,
            new_spacing=image_nii.GetSpacing(),
            new_origin=image_nii.GetOrigin(),
            new_size=image_nii.GetSize(),
            new_direction=image_nii.GetDirection(),
            center_origin=None,
            interp=sitk.sitkLinear,
            dtype=sitk.sitkFloat32,
            constant_value=0
        )

        # Processing Bbox
        print(f"        ----> Removing bbox out of roi mask")
        fracture = sitk.GetArrayFromImage(fracture_nii)
        fracture = fracture >= threshold_fracture

        # Remove error fracture prediction
        seg = sitk.GetArrayFromImage(seg_nii)
        roi_mask = skimage.morphology.dilation(seg)
        roi_mask[roi_mask > 7] = 0

        use_this_case_for_training = True
        for C_i in range(1, 8):
            fracture_in_this = int(train_csv[f"C{C_i}"][meta_info_index])
            if not fracture_in_this:
                roi_mask[roi_mask == C_i] = 0

            if fracture_in_this and (not np.any(fracture[roi_mask==C_i])):
                use_this_case_for_training = False
                break

        if not use_this_case_for_training:
            count -= 1
            print(f"       ----> Ignore this case {case_id}")

        else:
            fracture[roi_mask < 1] = 0

            fracture_nii = sitk.GetImageFromArray(np.uint8(fracture))
            fracture_nii.SetOrigin(image_nii.GetOrigin())
            fracture_nii.SetSpacing(image_nii.GetSpacing())
            fracture_nii.SetDirection(image_nii.GetDirection())

            # Saving
            print(f"        ----> Saving")
            sitk.WriteImage(image_nii, f"{save_dir_image}/{case_id}_0000.nii.gz")
            sitk.WriteImage(fracture_nii, f"{save_dir_label}/{case_id}.nii.gz")


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
        "0": "CT"
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
