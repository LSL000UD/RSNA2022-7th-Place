import json
from collections import OrderedDict
import setting


from Utils.CommonTools.sitk_base import resample
from Utils.CommonTools.dir import try_recursive_mkdir
import SimpleITK as sitk
import os
import numpy as np

import path


def get_dataset():
    source_dir = f"{path.path_root}"
    save_dir = f'{setting.path_nnunet_raw_data}/{setting.task_name}'

    save_dir_image = f"{save_dir}/imagesTr"
    save_dir_label = f"{save_dir}/labelsTr"
    try_recursive_mkdir(save_dir_image)
    try_recursive_mkdir(save_dir_label)

    new_spacing = (1.5, 1.5, 1.5)
    count = 0
    for dataset_name in ['CompetitionData', 'TotalSegmentatorData', 'Verse2020Data']:
        image_dir = f"{source_dir}/{dataset_name}/Image"
        seg_dir = f"{source_dir}/{dataset_name}/Segmentation"

        for file in os.listdir(seg_dir):
            if file.find('.nii.gz') == -1:
                continue

            case_id = file.split('.nii.gz')[0]
            count += 1

            # Reading
            print(f"==> Processing {count} {case_id}...")
            image_nii = sitk.ReadImage(f"{image_dir}/{case_id}.nii.gz")
            seg_nii = sitk.ReadImage(f"{seg_dir}/{case_id}.nii.gz")

            # Remove label > 8
            seg = sitk.GetArrayFromImage(seg_nii)
            seg[seg > 8] = 0
            seg_nii = sitk.GetImageFromArray(np.uint8(seg))
            seg_nii.SetOrigin(image_nii.GetOrigin())
            seg_nii.SetSpacing(image_nii.GetSpacing())
            seg_nii.SetDirection(image_nii.GetDirection())

            # Resampling
            print(f"        ----> Resampling")
            image_nii = resample(
                image_nii,
                new_spacing=new_spacing[::-1],
                new_origin=None,
                new_size=None,
                new_direction=None,
                center_origin=None,
                interp=sitk.sitkLinear,
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

            # Saving
            print(f"        ----> Saving")
            sitk.WriteImage(image_nii, f"{save_dir_image}/{case_id}_0000.nii.gz")
            sitk.WriteImage(seg_nii, f"{save_dir_label}/{case_id}.nii.gz")


def get_dataset_info():
    save_dir = f'{setting.path_nnunet_raw_data}/{setting.task_name}'
    save_dir_label = f"{save_dir}/labelsTr"

    # Train case ids
    list_train_ids = []
    for file in os.listdir(save_dir_label):
        if file.find('.nii.gz') > -1:
            list_train_ids.append(file.split('.nii.gz')[0])
    print(f"==> Total {len(list_train_ids)} train cases")

    label_names = {str(k):k for k in range(9)}
    print(label_names)

    # Json
    json_dict = OrderedDict()
    json_dict['name'] = setting.task_name
    json_dict['description'] = "C1-C7 + T1 "
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
