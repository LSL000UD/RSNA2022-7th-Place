from Utils.Inference.nnunet_inference import NNUnetCTPredictor
import SimpleITK as sitk
import numpy as np
import torch
import path


class Predictor(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(Predictor, self).__init__(*args, **kwargs)

    def post_processing(self, pred):
        pred = np.argmax(pred, axis=0)
        return pred


if __name__ == '__main__':
    predictor_1 = Predictor(
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
        tta_flip_axis=(4, ),

        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkNearestNeighbor,
        resampling_dtype=sitk.sitkInt16,
        resampling_constance_value=-1024,

        remove_air_CT=True
    )

    predictor_1.predict_from_nii_dir(
        input_dir=f"{path.path_root}/CompetitionData/Image",
        output_dir=f"{path.path_root}/CompetitionData/PredictedSegmentation",

        must_include_all=['.nii.gz'],
        must_include_one_of=None,
        must_exclude_all=None,
    )