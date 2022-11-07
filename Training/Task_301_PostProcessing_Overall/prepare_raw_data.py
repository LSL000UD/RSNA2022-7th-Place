import os
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from Utils.CommonTools import sitk_base
from Utils.CommonTools.bbox import *
import pandas as pd

import setting

from Utils.CommonTools.dir import try_recursive_mkdir
from Utils.PreProcessing.resampling import sitk_dummy_3D_resample

import SimpleITK as sitk
import numpy as np
import os
import pickle
import numpy as np

from scipy.optimize import curve_fit
import path
import torch.nn.functional as F


def get_dataset():
    pred_dir = f"{path.path_root}/CompetitionData/PredictedFractureV4_3_1TTA"
    train_csv = pd.read_csv(f"{path.path_competition_data}/train.csv")

    save_dir = setting.path_training_data
    save_dir_image = f"{save_dir}/imagesTr"
    try_recursive_mkdir(save_dir_image)

    new_size = (96, 96, 96)

    count = 0
    for file in os.listdir(pred_dir):
        if file.find('_fracture.nii.gz') == -1:
            continue

        case_id = file.split('_fracture.nii.gz')[0]
        meta_info_index = list(train_csv['StudyInstanceUID']).index(case_id)

        count += 1
        print(f"==> {count}: Processing {case_id}")

        pred_nii = sitk.ReadImage(f"{pred_dir}/{file}")
        seg_nii = sitk.ReadImage(f"{pred_dir}/{case_id}_C1_C7.nii.gz")

        new_spacing = list(np.array(pred_nii.GetSize()[::-1]) / np.array(new_size))

        pred_nii = sitk_base.resample(
            pred_nii,
            new_spacing[::-1],
            new_origin=None,
            new_size=new_size[::-1],
            new_direction=None,
            center_origin=None,
            interp=sitk.sitkLinear,
            dtype=sitk.sitkFloat32,
            constant_value=0
        )
        seg_nii = sitk_base.resample(
            seg_nii,
            new_spacing[::-1],
            new_origin=None,
            new_size=new_size[::-1],
            new_direction=None,
            center_origin=None,
            interp=sitk.sitkNearestNeighbor,
            dtype=sitk.sitkUInt8,
            constant_value=0
        )

        # Saving
        fracture_exist = train_csv[f"patient_overall"][meta_info_index] == 1
        if fracture_exist:
            sitk.WriteImage(pred_nii, f"{save_dir_image}/{case_id}_pred_positive.nii.gz")
            sitk.WriteImage(seg_nii, f"{save_dir_image}/{case_id}_C1_C7_positive.nii.gz")
        else:
            sitk.WriteImage(pred_nii, f"{save_dir_image}/{case_id}_pred.nii.gz")
            sitk.WriteImage(seg_nii, f"{save_dir_image}/{case_id}_C1_C7.nii.gz")


def get_dataset_info():
    path_training_data = setting.path_training_data
    path_training_data_info = setting.path_training_data_info

    dataset_info = []
    for file in os.listdir(f"{path_training_data}/imagesTr"):
        if file.find('_pred') == -1:
            continue

        case_id = file.split('.nii.gz')[0]
        dataset_info.append(case_id)

    dataset_info = {'no_val':dataset_info}
    pickle.dump(dataset_info, open(path_training_data_info, 'wb'))
    print(f"==> Saving dataset info to {path_training_data_info}")
    kkkk = 1


def main():
    get_dataset()
    get_dataset_info()


if __name__ == '__main__':
    main()
