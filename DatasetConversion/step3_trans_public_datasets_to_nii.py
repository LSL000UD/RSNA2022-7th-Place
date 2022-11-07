import SimpleITK as sitk
import numpy as np
import os

from Utils.CommonTools.dir import try_recursive_mkdir
from tqdm import tqdm
import path
from Utils.CommonTools.connected_components import keep_largest_cc, keep_largest_cc_binary


def get_Verse2020():
    source_dir = path.path_Verse2020
    save_dir = f"{path.path_root}/Verse2020Data"
    try_recursive_mkdir(save_dir)
    try_recursive_mkdir(f"{save_dir}/Image")
    try_recursive_mkdir(f"{save_dir}/Segmentation")

    # Get all files
    list_file_info = []
    for sub_dir in ['dataset-verse20training/dataset-01training',
                    'dataset-verse20validation/dataset-02validation',
                    'dataset-verse20test/dataset-03test']:
        rawdata_dir = f"{source_dir}/{sub_dir}/rawdata"
        derivatives_dir = f"{source_dir}/{sub_dir}/derivatives"

        for case_id in os.listdir(rawdata_dir):
            if not os.path.isdir(f"{rawdata_dir}/{case_id}"):
                continue

            image_file = os.listdir(f"{rawdata_dir}/{case_id}")[0]
            image_file = f"{rawdata_dir}/{case_id}/{image_file}"

            for label_file in os.listdir(f"{derivatives_dir}/{case_id}"):
                if label_file.find('.nii.gz') > -1:
                    gt_file = label_file
            gt_file = f"{derivatives_dir}/{case_id}/{gt_file}"

            list_file_info.append(
                {'case_id': case_id,
                 'image': image_file,
                 'gt': gt_file
                 }
            )

    # Reading and saving
    count = 0
    for file_info in tqdm(list_file_info):
        case_id = file_info['case_id']
        image_file = file_info['image']
        gt_file = file_info['gt']

        try:
            gt_nii = sitk.ReadImage(gt_file)
            gt = sitk.GetArrayFromImage(gt_nii)
            gt[gt > 8] = 0

            # Remove some noise
            largest_cc_binary = keep_largest_cc_binary(gt > 0)
            gt[largest_cc_binary < 1] = 0
            gt = keep_largest_cc(gt)  # For each index, eg. C1, C2, C3 ... C7
            if (not np.any(gt == 1)) or (not np.any(gt == 8)):
                continue

            image_nii = sitk.ReadImage(image_file)
        except:
            print(f"== > Read error, ignore this case {gt_file}!")  # ITK only supports orthonormal direction cosines
            continue

        gt_nii = sitk.GetImageFromArray(np.uint8(gt))
        gt_nii.SetOrigin(image_nii.GetOrigin())
        gt_nii.SetDirection(image_nii.GetDirection())
        gt_nii.SetSpacing(image_nii.GetSpacing())

        count += 1
        print(f"{count}:    {case_id}")
        sitk.WriteImage(image_nii, f"{save_dir}/Image/{case_id}.nii.gz")
        sitk.WriteImage(gt_nii, f"{save_dir}/Segmentation/{case_id}.nii.gz")


def get_TotalSegmentator():
    source_dir = path.path_TotalSegmentator
    save_dir = f"{path.path_root}/TotalSegmentatorData"
    try_recursive_mkdir(save_dir)
    try_recursive_mkdir(f"{save_dir}/Image")
    try_recursive_mkdir(f"{save_dir}/Segmentation")

    count = 0
    for case_id in tqdm(os.listdir(source_dir)):
        cur_dir = f"{source_dir}/{case_id}"
        if not os.path.isdir(cur_dir):
            continue

        gt_dir = f"{cur_dir}/segmentations"


        try:
            C1_nii = sitk.ReadImage(f'{gt_dir}/vertebrae_C1.nii.gz')
            C1 = sitk.GetArrayFromImage(C1_nii)
            if not np.any(C1):  # We only use data with C1 exist
                continue

            gt = np.uint8(C1 > 0)
            for C_i in range(2, 8):
                Ci_nii = sitk.ReadImage(f'{gt_dir}/vertebrae_C{C_i}.nii.gz')
                Ci = sitk.GetArrayFromImage(Ci_nii)
                gt[Ci > 0] = C_i
            T1_nii = sitk.ReadImage(f'{gt_dir}/vertebrae_T1.nii.gz')
            T1 = sitk.GetArrayFromImage(T1_nii)
            gt[T1 > 0] = 8

            image_nii = sitk.ReadImage(f"{cur_dir}/ct.nii.gz")
        except:
            print(f"== > Read error, ignore this case {case_id}!")  # ITK only supports orthonormal direction cosines
            continue

        # Remove some noise
        largest_cc_binary = keep_largest_cc_binary(gt > 0)
        gt[largest_cc_binary < 1] = 0
        gt = keep_largest_cc(gt)  # For each index, eg. C1, C2, C3 ... C7, T1
        if (not np.any(gt == 1)) or (not np.any(gt == 8)):
            continue

        gt_nii = sitk.GetImageFromArray(np.uint8(gt))
        gt_nii.SetOrigin(image_nii.GetOrigin())
        gt_nii.SetDirection(image_nii.GetDirection())
        gt_nii.SetSpacing(image_nii.GetSpacing())

        # Save
        count += 1
        print(f"{count}:    {case_id}")
        sitk.WriteImage(image_nii, f"{save_dir}/Image/{case_id}.nii.gz")
        sitk.WriteImage(gt_nii, f"{save_dir}/Segmentation/{case_id}.nii.gz")


def main():
    get_TotalSegmentator()
    get_Verse2020()


if __name__ == '__main__':
    main()
