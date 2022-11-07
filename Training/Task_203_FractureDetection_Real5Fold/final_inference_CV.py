import math
import os
import pickle
import sys
import time
import SimpleITK as sitk
import numpy as np
import torch
import path
import skimage.morphology
import threading
import gc

from Utils.CommonTools.bbox import get_bbox, extend_bbox
from Utils.post_processing import keep_largest_cervical_cc
from Utils.Inference.nnunet_inference import NNUnetCTPredictor
from Utils.CommonTools.dir import try_recursive_mkdir
from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info
from Utils.CommonTools.NiiIO import read_from_DICOM_dir
from Utils.PreProcessing.resampling import sitk_dummy_3D_resample


class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def resampling(self, ct_nii):

        ori_spacing = ct_nii.GetSpacing()[::-1]  # to z,y,x
        ori_size = ct_nii.GetSize()[::-1]
        new_spacing = self.plan['plans_per_stage'][self.plan_stage]['current_spacing']

        # For faster inference, but may reduce accuracy
        new_size = [int(math.ceil(ori_size[0] * ori_spacing[0] / 0.8)), 224, 224]

        new_spacing[0] = 0.8
        new_spacing[1] = ori_size[1] * ori_spacing[1] / 224.
        new_spacing[2] = ori_size[2] * ori_spacing[2] / 224.

        do_resampling = np.any(np.abs(np.array(ori_spacing) - np.array(new_spacing)) > self.resampling_tolerance)
        if do_resampling:
            ct_nii = sitk_dummy_3D_resample(
                ct_nii,
                new_spacing=new_spacing[::-1],
                new_size=new_size[::-1],
                interp_xy=self.resampling_mode,
                interp_z=sitk.sitkNearestNeighbor,
                out_dtype=self.resampling_dtype,
                constant_value=self.resampling_constance_value
            )
        else:
            print(f'==> No necessary to do resampling ori {ori_spacing}, new: {new_spacing}')

        return ct_nii


class DICOMReader(threading.Thread):
    def __init__(self, func=read_from_DICOM_dir, args=()):
        super(DICOMReader, self).__init__()

        self.func = func
        self.args = args

        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        return self.result


class FractureDetector:
    def __init__(self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi = extend_roi

        self.params = {
            'alpha': [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024],
            'beta': [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36],
            'min_score': [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
            'max_score': [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999]
        }

        self.results = {}

    def get_c1_c7_bbox(self, pred, image_spacing):
        c1_c7_bbox = get_bbox(pred > 0)
        if c1_c7_bbox is None:
            return None
        c1_c7_bbox = extend_bbox(
            c1_c7_bbox,
            max_shape=pred.shape,
            list_extend_length=self.extend_roi,
            spacing=image_spacing,
            approximate_method=np.ceil
        )
        return c1_c7_bbox

    def predict_stage1(self, ct_nii):
        # Resampling
        time_start = time.time()
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = self.predictor_stage1.resampling(ct_nii)
        print(f"                  Resampling use: {time.time() - time_start}")

        # Pre_processing
        time_start = time.time()
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage1.pre_processing(image)
        print(f"                  Pre_processing use: {time.time() - time_start}")

        # Model forward
        time_start = time.time()
        pred = self.predictor_stage1.sliding_window_inference(image)
        print(f"                  Model forward use: {time.time() - time_start}")

        # Post processing
        time_start = time.time()
        pred = np.argmax(pred, axis=0)
        pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
        pred[pred > 7] = 0
        print(f"                  Post processing use: {time.time() - time_start}")

        # Resampling back
        time_start = time.time()
        pred_nii = sitk.GetImageFromArray(np.uint8(pred))
        pred_nii = copy_nii_info(ct_nii, pred_nii)
        pred_nii = resample(
            pred_nii,
            new_spacing=ori_nii_info['spacing'],
            new_origin=ori_nii_info['origin'],
            new_size=ori_nii_info['size'],
            new_direction=ori_nii_info['direction'],
            center_origin=None,
            interp=sitk.sitkNearestNeighbor,
            dtype=sitk.sitkUInt8,
            constant_value=0
        )

        pred = sitk.GetArrayFromImage(pred_nii)
        print(f"                  Resampling back use: {time.time() - time_start}")
        return pred

    def predict_stage2(self, ct_nii):

        # Resampling
        time_start = time.time()
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = self.predictor_stage2.resampling(ct_nii)
        print(f"                  Resampling use: {time.time() - time_start}")

        # Pre_processing
        time_start = time.time()
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage2.pre_processing(image)
        print(f"                  Pre_processing use: {time.time() - time_start}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{image.shape}")
        # Model forward
        time_start = time.time()
        pred = self.predictor_stage2.sliding_window_inference(image)
        print(f"                  Model forward use: {time.time() - time_start}")

        # Post processing
        time_start = time.time()
        pred = pred[1]  # 0 for background, 1 for foreground
        print(f"                  Post processing use: {time.time() - time_start}")

        # Resampling back
        time_start = time.time()
        pred_nii = sitk.GetImageFromArray(pred)
        pred_nii = copy_nii_info(ct_nii, pred_nii)
        pred_nii = resample(
            pred_nii,
            new_spacing=ori_nii_info['spacing'],
            new_origin=ori_nii_info['origin'],
            new_size=ori_nii_info['size'],
            new_direction=ori_nii_info['direction'],
            center_origin=None,
            interp=sitk.sitkLinear,
            dtype=sitk.sitkFloat32,
            constant_value=0.
        )
        pred = sitk.GetArrayFromImage(pred_nii)
        print(f"                  Resampling back use: {time.time() - time_start}")

        return pred

    def get_score(self, pred_c1_c7, pred_fracture):
        output = np.zeros(8, np.float32)  # Overall, C1-C7

        if (pred_c1_c7 is not None) and (pred_fracture is not None):
            # Overall, C1-C7
            for C_i in range(8):
                if C_i == 0:
                    roi_fracture = pred_fracture[np.logical_and(pred_fracture >= self.params['alpha'][C_i], pred_c1_c7 > 0)]
                else:
                    roi_fracture = pred_fracture[np.logical_and(pred_fracture >= self.params['alpha'][C_i], pred_c1_c7 == C_i)]

                if len(roi_fracture) == 0:
                    output[C_i] = self.params['min_score'][C_i]
                else:
                    output[C_i] = max(self.params['min_score'][C_i],
                                      min(self.params['max_score'][C_i],
                                          np.percentile(roi_fracture, 100 * self.params['beta'][C_i])
                                          )
                                      )
        else:
            for C_i in range(8):
                output[C_i] = self.params['min_score'][C_i]
        return output

    @staticmethod
    def read_DICOM_multi_thread(list_DICOM_dirs):
        list_thread = []
        list_outputs = []

        for DICOM_dir in list_DICOM_dirs:
            cur_thread = DICOMReader(func=read_from_DICOM_dir, args=(DICOM_dir, ))
            cur_thread.start()
            list_thread.append(cur_thread)

        for cur_thread in list_thread:
            list_outputs.append(cur_thread.get_result())
        list_thread.clear()

        return list_outputs

    def predict(self,
                list_test_files,
                num_thread=4,
                output_dir=None
                ):
        try_recursive_mkdir(output_dir)

        overall_time_start = time.time()
        num_split = math.ceil(len(list_test_files) / num_thread)
        for split_i in range(num_split):
            cur_test_files = list_test_files[num_thread * split_i:num_thread * (split_i + 1)]
            cur_case_ids = [test_file.split('/')[-1] for test_file in cur_test_files]

            print(f"==> Predicting {split_i}: {cur_case_ids}")

            # --------------------------------- Inference one split ------------------------------ #
            # Step 1, Read all images
            time_start = time.time()
            cur_ct_niis = self.read_DICOM_multi_thread(cur_test_files)
            print(f"    Finish Reading use : {time.time() - time_start} seconds")

            time_start = time.time()
            for case_i in range(len(cur_ct_niis)):
                case_id = cur_case_ids[case_i]
                ct_nii = cur_ct_niis[case_i]

                ori_nii_info = get_nii_info(ct_nii)  # Record original nii info before processing, eg. spacing, size

                # Step 2, predictor_stage1, segment C1-C7
                pred_1 = self.predict_stage1(ct_nii)

                #  Step 3, get c1-c7 bounding bbox
                c1_c7_bbox = self.get_c1_c7_bbox(pred_1, ori_nii_info['spacing'][::-1])

                # Step 4, predictor_stage2,  segment fracture
                if c1_c7_bbox is not None:
                    bz, ez, by, ey, bx, ex = c1_c7_bbox
                    roi_ct_nii = ct_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]
                    roi_pred_1 = pred_1[bz:ez + 1, by:ey + 1, bx:ex + 1]

                    # Stage 2 inference
                    roi_pred_2 = self.predict_stage2(roi_ct_nii)
                else:
                    roi_pred_1 = None
                    roi_pred_2 = None

                # Saving 5, get score
                if roi_pred_1 is None:
                    roi_pred_1 = np.zeros((2, 2, 2), np.uint8)
                    roi_pred_2 = np.zeros((2, 2, 2), np.float32)

                sitk.WriteImage(sitk.GetImageFromArray(np.uint8(roi_pred_1)), f"{output_dir}/{case_id}_C1_C7.nii.gz")
                sitk.WriteImage(sitk.GetImageFromArray(np.float32(roi_pred_2)), f"{output_dir}/{case_id}_fracture.nii.gz")

                score = self.get_score(roi_pred_1, roi_pred_2)
                self.results[case_id] = score
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     {score}")
            print(f"    Finish this split use : {time.time() - time_start} seconds")
            print(f"    Overall use : {time.time() - overall_time_start} seconds")

            gc.collect()


if __name__ == '__main__':
    for fold in [0, 1, 2, 3, 4]:
        # --------------------------- Need Modification --------------------------------- #
        input_dir = f"E:/RawData/rsna-2022-cervical-spine-fracture-detection/train_images"
        output_dir = f"E:/Output/BoneLabeling/CompetitionData/PredictedFractureV4_3_1TTA"

        list_model_C1_C7_segmentation = [
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_101_VertebralLocation_GeneratePseudoLabel/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model',
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_102_VertebralLocation_GeneratePseudoLabel_Run2/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model',
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_103_VertebralLocation_GeneratePseudoLabel_Run3/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model']
        plan_C1_C7_segmentation = f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_103_VertebralLocation_GeneratePseudoLabel_Run3/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl'

        list_model_fracture_detection = [
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_203_FractureDetection_Real5Fold/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold}/model_final_checkpoint.model',
        ]
        plan_fracture_detection = f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_203_FractureDetection_Real5Fold/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl'
        # --------------------------- Need Modification --------------------------------- #

        # Get list case id
        splits_final = pickle.load(open(
            'E:/Output/BoneLabeling/nnUnet/Models/nnUNet/3d_fullres/Task_203_FractureDetection_Real5Fold/nnUNetTrainerV2__nnUNetPlansv2.1/splits_final.pkl',
            'rb'))
        list_DICOM_dirs = list(splits_final[fold]['val'])
        for case_i in range(len(list_DICOM_dirs)):
            list_DICOM_dirs[case_i] = f"{input_dir}/{list_DICOM_dirs[case_i]}"
        # -------------------------------------------------------------------------------- #

        list_size_DICOM_dirs = []
        for case_i in range(len(list_DICOM_dirs)):
            size_ = 0
            for file in os.listdir(list_DICOM_dirs[case_i]):
                size_ += os.path.getsize(f"{list_DICOM_dirs[case_i]}/{file}")
            list_size_DICOM_dirs.append(size_)

        list_DICOM_dirs = list(np.array(list_DICOM_dirs)[np.argsort(list_size_DICOM_dirs)[::-1]])
        print(f"==> Sort DICOM dirs by size")
        print(list_DICOM_dirs[0])

        predictor_1 = NNUnetCTPredictor(
            list_model_pth=list_model_C1_C7_segmentation,
            plan_file=plan_C1_C7_segmentation,
            plan_stage=-1,
            device=torch.device('cuda:0'),
            use_gaussian_for_sliding_window=True,

            patch_size=None,
            stride=None,
            tta=False,
            tta_flip_axis=(4,),

            resampling_tolerance=0.01,
            resampling_mode=sitk.sitkNearestNeighbor,
            resampling_dtype=sitk.sitkInt16,
            resampling_constance_value=-1024,

            remove_air_CT=True
        )
        predictor_2 = PredictorStage2(
            list_model_pth=list_model_fracture_detection,
            plan_file=plan_fracture_detection,
            plan_stage=-1,
            device=torch.device('cuda:0'),
            use_gaussian_for_sliding_window=True,

            patch_size=(96, 224, 224),
            stride=(96, 224, 224),
            tta=True,
            tta_flip_axis=(4,),

            resampling_tolerance=0.01,
            resampling_mode=sitk.sitkNearestNeighbor,
            resampling_dtype=sitk.sitkInt16,
            resampling_constance_value=-1024,

            remove_air_CT=False,

            save_dtype=np.float32
        )

        c2f_predictor = FractureDetector(
            predictor_stage1=predictor_1,
            predictor_stage2=predictor_2
        )
        c2f_predictor.predict(
            output_dir=output_dir,
            list_test_files=list_DICOM_dirs
        )
