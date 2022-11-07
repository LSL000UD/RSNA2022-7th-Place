import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import torch
import path

from Utils.CommonTools.bbox import get_bbox, extend_bbox
from Utils.post_processing import keep_largest_cervical_cc
from Utils.CommonTools.dir import find_files_in_dir
from Utils.Inference.nnunet_inference import NNUnetCTPredictor
from Utils.CommonTools.sitk_base import copy_nii_info
from Utils.CommonTools.dir import try_recursive_mkdir


class FractureDetector:
    def __init__(self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi =extend_roi

        self.predictions_all_stages = []

    def get_roi_bbox(self, pred, image_spacing):
        pred = keep_largest_cervical_cc(pred, image_spacing)
        seg_bbox = get_bbox(np.logical_and(pred >= 1, pred <= 7))
        if seg_bbox is None:
            return None
        seg_bbox = extend_bbox(seg_bbox, max_shape=pred.shape, list_extend_length=self.extend_roi, spacing=image_spacing,
                               approximate_method=np.ceil)
        return seg_bbox

    def predict_from_nii_dir(self,
                             input_dir,
                             output_dir,
                             must_include_all=('.nii.gz',),
                             must_include_one_of=None,
                             must_exclude_all=None,
                             ):
        try_recursive_mkdir(output_dir)

        list_files = find_files_in_dir(input_dir,
                                       must_include_all=must_include_all,
                                       must_include_one_of=must_include_one_of,
                                       must_exclude_all=must_exclude_all
                                       )

        count = 0
        for file in list_files:
            case_id = file.split('/')[-1].split('.nii')[0]

            count += 1
            print(f"==> Predicting {count}: {case_id}")

            image_nii = sitk.ReadImage(file)

            # Stage 1
            pred_1 = self.predictor_stage1.predict_from_nii(image_nii, return_nii=False)

            # Crop
            bbox_pred_1 = self.get_roi_bbox(pred_1, image_spacing=image_nii.GetSpacing()[::-1])
            if bbox_pred_1 is None:
                pred_2 = np.zeros(pred_1.shape, np.uint8)
            else:
                bz, ez, by, ey, bx, ex = bbox_pred_1
                roi_image_nii = image_nii[bx:ex + 1, by:ey + 1, bz:ez + 1]

                # Stage 2
                roi_pred_2 = self.predictor_stage2.predict_from_nii(roi_image_nii, return_nii=False)
                pred_2 = np.zeros(pred_1.shape, np.float32)
                pred_2[bz:ez + 1, by:ey + 1, bx:ex + 1] = roi_pred_2

            print(f"        ----> Saving ... ")
            pred_2_nii = sitk.GetImageFromArray(pred_2)
            pred_2_nii = copy_nii_info(image_nii, pred_2_nii)
            sitk.WriteImage(pred_2_nii, f"{output_dir}/{case_id}.nii.gz")


class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def init_model(self):
        num_input_channels = 1  ########################
        base_num_features = self.plan['base_num_features']
        num_classes = self.plan['num_classes'] + 1
        net_numpool = len(self.plan['plans_per_stage'][self.plan_stage]['pool_op_kernel_sizes'])
        conv_per_stage = self.plan['conv_per_stage']
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        net_num_pool_op_kernel_sizes = self.plan['plans_per_stage'][self.plan_stage]['pool_op_kernel_sizes']
        net_conv_kernel_sizes = self.plan['plans_per_stage'][self.plan_stage]['conv_kernel_sizes']

        with torch.no_grad():
            self.list_model = []
            for i in range(len(self.list_model_pth)):
                model = self.model_class(num_input_channels, base_num_features, num_classes, net_numpool,
                                     conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                     dropout_op_kwargs,
                                     net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, None,
                                     net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

                ckpt = torch.load(self.list_model_pth[i], map_location='cpu')['state_dict']

                if self.list_model_pth[i].find('SWA') > -1:
                    new_ckpt = {}
                    for key in ckpt.keys():
                        if key.find('module') > -1:
                            new_ckpt[key[7:]] = ckpt[key]
                    ckpt = new_ckpt

                model.load_state_dict(ckpt)
                model.eval()
                model = model.to(self.device)
                self.list_model.append(model)

                print(f'==> Init model from {self.list_model_pth[i]} to device {self.device}')

    def post_processing(self, pred):
        return pred[1]


if __name__ == '__main__':
    predictor_1 = NNUnetCTPredictor(
        list_model_pth=[
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_101_VertebralLocation_GeneratePseudoLabel/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model',
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_102_VertebralLocation_GeneratePseudoLabel_Run2/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model',
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_103_VertebralLocation_GeneratePseudoLabel_Run3/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model'],
        plan_file=f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_103_VertebralLocation_GeneratePseudoLabel_Run3/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl',
        plan_stage=-1,
        device=torch.device('cuda:0'),
        use_gaussian_for_sliding_window=True,

        patch_size=None,
        stride=None,
        tta=True,
        tta_flip_axis=(4,),

        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkNearestNeighbor,
        resampling_dtype=sitk.sitkInt16,
        resampling_constance_value=-1024,

        remove_air_CT=True
    )
    predictor_2 = PredictorStage2(
        list_model_pth=[
            f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_201_FractureDetection_GeneratePseudoLabel/nnUNetTrainerV2__nnUNetPlansv2.1/all/model_final_checkpoint.model'
        ],
        plan_file=f'{path.path_root}/nnUnet/Models/nnUNet/3d_fullres/Task_201_FractureDetection_GeneratePseudoLabel/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl',
        plan_stage=-1,
        device=torch.device('cuda:0'),
        use_gaussian_for_sliding_window=True,

        patch_size=None,
        stride=None,
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
    c2f_predictor.predict_from_nii_dir(
        input_dir=f"{path.path_root}/CompetitionData/Image",
        output_dir=f"{path.path_root}/CompetitionData/PredictedFracture",

        must_include_all=['.nii.gz'],
        must_include_one_of=None,
        must_exclude_all=None,
    )
