import pickle
from tqdm import tqdm

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn

from nnunet.network_architecture.generic_UNet import Generic_UNet

from Utils.DataAugmentation.crop_and_pad import center_pad_to_size
from Utils.CommonTools.patch import generate_patch_by_sliding_window
from Utils.CommonTools.CT_preprocess import remove_CT_air
from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info, set_nii_info
from Utils.CommonTools.dir import try_recursive_mkdir, find_files_in_dir
from Utils.CommonTools.bbox import get_bbox
from Utils.PreProcessing.resampling import sitk_dummy_3D_resample


from scipy.ndimage.filters import gaussian_filter


class NNUnetPredictor:
    def __init__(self, list_model_pth, plan_file, plan_stage, device,
                 use_gaussian_for_sliding_window=False,
                 patch_size=None,
                 stride=None,
                 tta=False,
                 tta_flip_axis=(2, 3, 4),

                 list_pre_process_channels=None,  # None means pre-process all channel

                 model_class=Generic_UNet
                 ):
        self.list_model_pth = list_model_pth
        self.plan_file = plan_file
        self.plan_stage = plan_stage
        self.device = device
        self.use_gaussian_for_sliding_window = use_gaussian_for_sliding_window
        self.gaussian_map = None

        self.patch_size = patch_size
        self.stride = stride
        self.tta = tta
        self.tta_flip_axis = tta_flip_axis

        self.list_pre_process_channels = list_pre_process_channels

        self.model_class = model_class
        print(f"        ----> Using TTA: {self.tta}, axis: {self.tta_flip_axis}")

        # Init params
        self.plan = None
        self.list_model = None

        self.init_plan()
        self.init_model()

    def init_plan(self):
        self.plan = pickle.load(open(self.plan_file, 'rb'))
        print(f'==> Init plan from {self.plan_file}')

        if self.plan['plans_per_stage'] is not None and self.plan_stage == -1:
            self.plan_stage = len(self.plan['plans_per_stage']) - 1

    def init_model(self):
        num_input_channels = self.plan['num_modalities']
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

    def init_gaussian(self, sigma_scale=1. / 8):
        output = np.zeros(self.patch_size, np.float32)

        # Get sigma and center
        sigmas = []
        center = []
        for dim_i in range(len(self.patch_size)):
            sigmas.append(self.patch_size[dim_i] * sigma_scale)
            center.append(self.patch_size[dim_i] // 2)

        # Get guassian
        output[tuple(center)] = 1
        output = gaussian_filter(output, sigmas, 0, mode='constant', cval=0)
        output = output / np.max(output) * 1
        output = output.astype(np.float32)
        self.gaussian_map = output

    def pre_processing(self, image):
        """
        :param image: C, Z, Y, X
        :return: image: C, Z, Y, X
        """
        image = np.float32(image)

        pre_precess_channels = self.list_pre_process_channels if self.list_pre_process_channels is not None else range(image.shape[0])
        for chan_i in pre_precess_channels:
            p_005 = self.plan['dataset_properties']['intensityproperties'][chan_i]['percentile_00_5']
            p_995 = self.plan['dataset_properties']['intensityproperties'][chan_i]['percentile_99_5']
            mean_ = self.plan['dataset_properties']['intensityproperties'][chan_i]['mean']
            std_ = self.plan['dataset_properties']['intensityproperties'][chan_i]['sd']

            image[chan_i] = np.clip(image[chan_i], a_min=p_005, a_max=p_995)
            image[chan_i] = (image[chan_i] - mean_) / (std_ + 1e-7)

        return image

    def sliding_window_inference(self, image):
        """
        :param image: (C, Z, Y, X)
        :param pad_value_for_each_channel: pad values for each channels
        :return:
        """

        if self.patch_size is None:
            self.patch_size = self.plan['plans_per_stage'][self.plan_stage]['patch_size']

        if self.stride is None:
            self.stride = self.patch_size // 2
            self.stride = [max(1, k) for k in self.stride]

        if self.use_gaussian_for_sliding_window and self.gaussian_map is None:
            self.init_gaussian()

        # Pad it if input_size < target_size
        [image], pad_bbox = center_pad_to_size(
            [image],
            target_size=self.patch_size,
            list_pad_value=[0.],
            return_list_bbox=True
        )

        # --------------------------------- Sliding window -------------------------------- #
        # Get sliding-window
        input_size = image.shape[1:]
        list_bboxes = generate_patch_by_sliding_window(
            input_size=input_size,
            patch_size=self.patch_size,
            stride=self.stride
        )

        # Init outputs
        num_class = self.plan['num_classes'] + 1
        output_size = (num_class,) + input_size
        output = np.zeros(output_size, np.float32)
        count = np.zeros(input_size, np.float32)

        # Model Predicting
        with torch.no_grad():
            for bbox in tqdm(list_bboxes):
                bz, ez, by, ey, bx, ex = bbox[:]
                patch_input_ori = image[:, bz:ez + 1, by:ey + 1, bx:ex + 1].copy()

                # Flip TTA
                if self.tta:
                    p_flip_z = (0, 1) if 2 in self.tta_flip_axis else (0,)
                    p_flip_y = (0, 1) if 3 in self.tta_flip_axis else (0,)
                    p_flip_x = (0, 1) if 4 in self.tta_flip_axis else (0,)
                else:
                    p_flip_z = (0,)
                    p_flip_y = (0,)
                    p_flip_x = (0,)

                for flip_z in p_flip_z:
                    for flip_y in p_flip_y:
                        for flip_x in p_flip_x:
                            patch_input = torch.from_numpy(patch_input_ori).to(self.device).unsqueeze(0)

                            # Get flip axis
                            flip_axis = []
                            if flip_z == 1:
                                flip_axis.append(2)
                            if flip_y == 1:
                                flip_axis.append(3)
                            if flip_x == 1:
                                flip_axis.append(4)

                            # Flip aug
                            do_flip = (flip_z == 1) or (flip_y == 1) or (flip_x == 1)
                            if do_flip:
                                patch_input = torch.flip(patch_input, dims=flip_axis)

                            for model in self.list_model:
                                pred = model(patch_input)
                                pred = torch.softmax(pred, dim=1)

                                # Flip back
                                if do_flip:
                                    pred = torch.flip(pred, dims=flip_axis)

                                pred = np.array(pred.cpu().data)[0]

                                if self.use_gaussian_for_sliding_window:
                                    output[:, bz:ez + 1, by:ey + 1, bx:ex + 1] += pred * self.gaussian_map[np.newaxis]
                                    count[bz:ez + 1, by:ey + 1, bx:ex + 1] += self.gaussian_map
                                else:
                                    output[:, bz:ez + 1, by:ey + 1, bx:ex + 1] += pred
                                    count[bz:ez + 1, by:ey + 1, bx:ex + 1] += 1.0

        # Return softmax
        output = output / (count + 1e-8)
        bz, ez, by, ey, bx, ex = pad_bbox
        output = output[:, bz:ez + 1, by:ey + 1, bx:ex + 1]

        return output

    def post_processing(self, pred):
        return np.argmax(pred, axis=0)


class NNUnetCTPredictor(NNUnetPredictor):
    def __init__(self, list_model_pth, plan_file, plan_stage, device,
                 use_gaussian_for_sliding_window=False,
                 patch_size=None,
                 stride=None,
                 tta=False,
                 tta_flip_axis=(2, 3, 4),

                 resampling_tolerance=0.01,
                 resampling_mode=sitk.sitkLinear,
                 resampling_dtype=sitk.sitkInt16,
                 resampling_constance_value=-1024,

                 remove_air_CT=False,
                 save_dtype = np.uint8
                 ):
        super(NNUnetCTPredictor, self).__init__(list_model_pth, plan_file, plan_stage, device,
                                                use_gaussian_for_sliding_window=use_gaussian_for_sliding_window,
                                                patch_size=patch_size,
                                                stride=stride,
                                                tta=tta,
                                                tta_flip_axis=tta_flip_axis,
                                                )

        self.resampling_tolerance = resampling_tolerance
        self.resampling_mode = resampling_mode
        self.resampling_dtype = resampling_dtype
        self.resampling_constance_value = resampling_constance_value
        self.remove_air_CT = remove_air_CT
        self.save_dtype = save_dtype

    def resampling(self, ct_nii):

        ori_spacing = ct_nii.GetSpacing()[::-1]  # to z,y,x
        new_spacing = self.plan['plans_per_stage'][self.plan_stage]['current_spacing']

        do_resampling = np.any(np.abs(np.array(ori_spacing) - np.array(new_spacing)) > self.resampling_tolerance)
        if do_resampling:
            ct_nii = sitk_dummy_3D_resample(
                ct_nii,
                new_spacing=new_spacing[::-1],
                interp_xy=self.resampling_mode,
                interp_z=sitk.sitkNearestNeighbor,
                out_dtype=self.resampling_dtype,
                constant_value=self.resampling_constance_value
            )
            # ct_nii = resample(
            #     ct_nii,
            #     new_spacing=new_spacing[::-1],
            #     new_origin=None,
            #     new_size=None,
            #     new_direction=None,
            #     center_origin=None,
            #     interp=resampling_mode,
            #     dtype=resampling_dtype,
            #     constant_value=resampling_constance_value
            # )
        else:
            print(f'==> No necessary to do resampling ori {ori_spacing}, new: {new_spacing}')

        return ct_nii

    def resampling_back(self, pred_prob, current_nii_info, ori_nii_info):
        num_cls = pred_prob.shape[0]

        list_pred_per_cls = []
        for i in range(num_cls):
            pred_nii = sitk.GetImageFromArray(pred_prob[i])
            set_nii_info(pred_nii, current_nii_info)

            pred_nii = resample(
                pred_nii,
                new_spacing=ori_nii_info['spacing'],
                new_origin=ori_nii_info['origin'],
                new_size=ori_nii_info['size'],
                new_direction=ori_nii_info['direction'],
                center_origin=None,
                interp=sitk.sitkLinear,
                dtype=sitk.sitkFloat32,
                constant_value=0
            )
            list_pred_per_cls.append(sitk.GetArrayFromImage(pred_nii)[np.newaxis])

        final_pred = np.concatenate(list_pred_per_cls, axis=0)
        return final_pred

    def predict_from_nii(self, ct_nii, return_nii=True):

        # Record original nii info before processing
        ori_nii_info = get_nii_info(ct_nii)

        # Remove air for CT
        if self.remove_air_CT:
            print(f"            ----> Removing CT air... ")
            ct_nii = remove_CT_air(ct_nii, return_bbox=False)[0]

        # Resampling
        print(f"            ----> Resampling ...")
        ct_nii = self.resampling(ct_nii)

        # Pre_processing
        print(f"            ----> Pre-processing ...")
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.pre_processing(image)

        print(f"            ----> Predicting  ... ")
        pred = self.sliding_window_inference(image)

        # Resampling back
        print(f"            ----> Re-sampling back ...")
        pred = self.resampling_back(pred, get_nii_info(ct_nii), ori_nii_info)

        # Post-Processing
        print(f"            ----> Post processing ... ")
        pred = self.post_processing(pred).astype(self.save_dtype)

        if return_nii:
            pred_nii = sitk.GetImageFromArray(pred)
            set_nii_info(pred_nii, ori_nii_info)

            return pred_nii
        else:
            return pred

    def predict_from_nii_dir(self,
                             input_dir,
                             output_dir,
                             must_include_all=('.nii.gz',),
                             must_include_one_of=None,
                             must_exclude_all=None
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
            pred_nii = self.predict_from_nii(image_nii)

            print(f"            ----> Saving ... ")
            sitk.WriteImage(pred_nii, f"{output_dir}/{case_id}.nii.gz")


class CoarseToFine_CTPredictor:
    def __init__(self, predictor_stage1, predictor_stage2):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2

        self.predictions_all_stages = []

    @staticmethod
    def get_roi_bbox(pred):
        bbox = get_bbox(pred > 0)
        return bbox

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
            bbox_pred_1 = self.get_roi_bbox(pred_1)
            bz, ez, by, ey, bx, ex = bbox_pred_1

            spacing = image_nii.GetSpacing()[::-1]
            extend_mm = 20.
            bz = max(0, bz - int(round(extend_mm / spacing[0])))
            by = max(0, by - int(round(extend_mm / spacing[1])))
            bx = max(0, bx - int(round(2 * extend_mm / spacing[2])))

            ez = min(pred_1.shape[0] - 1, ez + int(round(extend_mm / spacing[0])))
            ey = min(pred_1.shape[1] - 1, ey + int(round(extend_mm / spacing[1])))
            ex = min(pred_1.shape[2] - 1, ex + int(round(2 * extend_mm / spacing[2])))

            roi_image_nii = image_nii[bx:ex+1, by:ey+1, bz:ez+1]

            # Stage 2
            roi_pred_2 = self.predictor_stage2.predict_from_nii(roi_image_nii, return_nii=False)
            pred_2 = np.zeros(pred_1.shape, np.uint8)
            pred_2[bz:ez+1, by:ey+1, bx:ex+1] = roi_pred_2

            print(f"            ----> Saving ... ")
            pred_2_nii = sitk.GetImageFromArray(pred_2)
            pred_2_nii = copy_nii_info(image_nii, pred_2_nii)
            sitk.WriteImage(pred_2_nii, f"{output_dir}/{case_id}.nii.gz")

