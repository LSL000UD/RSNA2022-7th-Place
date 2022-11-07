import SimpleITK as sitk
import os
from Utils.CommonTools.NiiIO import read_from_DICOM_dir
from Utils.CommonTools.dir import try_recursive_mkdir

import path


def main():
    image_dir = f"{path.path_competition_data}/train_images"
    save_dir = f"{path.path_root}/CompetitionData/Image"
    try_recursive_mkdir(save_dir)

    count = 0
    for case_id in os.listdir(image_dir):
        case_dir = f"{image_dir}/{case_id}"

        image_nii = read_from_DICOM_dir(case_dir)

        count += 1
        print(f"==> Saving {count}:    {case_id}")
        sitk.WriteImage(image_nii, f"{save_dir}/{case_id}.nii.gz")


if __name__ == '__main__':
    main()
