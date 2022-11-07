import SimpleITK as sitk
import numpy as np
import pandas as pd
import pydicom

from Utils.CommonTools.dir import try_recursive_mkdir

import path


def main():
    dicom_dir = path.path_competition_data
    image_dir = f"{path.path_root}/CompetitionData/Image"
    bbox_info = pd.read_csv(f"{path.path_competition_data}/train_bounding_boxes.csv")

    save_dir = f"{path.path_root}/CompetitionData/Bbox"
    try_recursive_mkdir(save_dir)

    dict_bbox_info = {}
    for i in range(bbox_info.shape[0]):
        StudyInstanceUID = bbox_info['StudyInstanceUID'][i]
        x = bbox_info['x'][i]
        y = bbox_info['y'][i]
        width = bbox_info['width'][i]
        height = bbox_info['height'][i]
        slice_number = bbox_info['slice_number'][i]

        if StudyInstanceUID not in dict_bbox_info.keys():
            dict_bbox_info[StudyInstanceUID] = []

        dict_bbox_info[StudyInstanceUID].append({
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'slice_number': slice_number,
        })

    count = 0

    for case_id in dict_bbox_info.keys():
        count += 1
        print(f"==> Processing {count}:    {case_id}")

        image_nii = sitk.ReadImage(f"{image_dir}/{case_id}.nii.gz")
        gt = np.zeros(image_nii.GetSize()[::-1], np.uint8)

        for slice_info in dict_bbox_info[case_id]:
            x = slice_info['x']
            y = slice_info['y']
            width = slice_info['width']
            height = slice_info['height']
            slice_number = slice_info['slice_number']

            bx = int(round(x))
            ex = int(round(x + width))
            by = int(round(y))
            ey = int(round(y + height))

            # Get Slice index in Nii image
            slice_z_coord = float(pydicom.read_file(f'{dicom_dir}/train_images/{case_id}/{slice_number}.dcm').ImagePositionPatient[-1])
            index = None
            for i in range(image_nii.GetSize()[-1]):
                cur_index_z_coord = image_nii[0:1, 0:1, i:i+1].GetOrigin()[-1]
                if abs(cur_index_z_coord - slice_z_coord) <= 0.01:
                    index = i
                    break
            # Draw bbox
            gt[index, by:ey + 1, bx:ex + 1] = 1

        gt_nii = sitk.GetImageFromArray(gt)
        gt_nii.SetOrigin(image_nii.GetOrigin())
        gt_nii.SetDirection(image_nii.GetDirection())
        gt_nii.SetSpacing(image_nii.GetSpacing())

        print(f'        ----> Saving ')
        sitk.WriteImage(gt_nii, f"{save_dir}/{case_id}.nii.gz")


if __name__ == '__main__':
    main()
