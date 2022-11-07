# -*- encoding: utf-8 -*-
import random
import pickle
import os
import cv2
import numpy as np

import torch.utils.data as data

from Utils.DataAugmentation.color import *
from Utils.DataAugmentation.spatial import *
from Utils.DataAugmentation.common import *
from Utils.DataAugmentation.noise_and_blur import *
from Utils.DataAugmentation.crop_and_pad import *

import setting
import SimpleITK as sitk


class TrainDataset(data.Dataset):
    def __init__(self,
                 source_dir,
                 dataset_info,
                 num_samples_per_epoch,
                 ):
        self.source_dir = source_dir
        self.dataset_info = dataset_info
        self.num_samples_per_epoch = num_samples_per_epoch

        self.step1 = Step1Reader()
        self.step2 = Step2PreProcessor()
        self.step3 = Step3Augmentation()
        self.step4 = Step4PostProcessor()

        self.dataset_info = dataset_info
        self.list_all = dataset_info

        print(f"==> Total {len(self.list_all)} samples")
        print(f"==> Patch Size {setting.input_size} case")

    def __getitem__(self, index_):
        case_id = random.sample(self.list_all, 1)[0]
        if case_id.find('positive') > -1:
            fracture = 1
        else:
            fracture = 0

        list_files = [
            f"{self.source_dir}/imagesTr/{case_id}.nii.gz",
            f"{self.source_dir}/imagesTr/{case_id.replace('_pred', '_C1_C7')}.nii.gz",
            fracture
        ]
        try:
            list_images = self.step1(list_files)
            list_images = self.step2(list_images)
            list_images = self.step3(list_images)
            list_images = self.step4(list_images)
        except:  # Just for convenient debugging in multi-process dataloader
            raise Exception('Error in ' + case_id)

        return list_images

    def __len__(self):
        return self.num_samples_per_epoch


class Step1Reader:
    def __init__(self):
        pass

    def __call__(self, list_files):
        image_nii = sitk.ReadImage(list_files[0])
        image = sitk.GetArrayFromImage(image_nii)
        image = image[np.newaxis]

        seg_nii = sitk.ReadImage(list_files[1])
        seg = sitk.GetArrayFromImage(seg_nii)
        seg = seg[np.newaxis]

        fracture = list_files[2]
        return [image, seg, fracture]


class Step2PreProcessor:
    def __init__(self):
        pass

    def __call__(self, list_images):
        return list_images


class Step3Augmentation:
    def __init__(self):
        self.transforms = CompositeAug(
            CenterPadToSize(target_size=(128, 128, 128), list_pad_value=(0, 0), list_target_indexes=(0, 1)),
            RandomDummyRotateAndScale3d(
                list_interp=[cv2.INTER_LINEAR, cv2.INTER_NEAREST],
                list_constant_value=[0, 0],
                list_specific_angle=None,
                angle_range=(-30, 30),
                list_specific_scale=None,
                scale_range=(0.6, 1.4),
                p=0.3,
                list_target_indexes=(0, 1)
            ),
            RandomCrop(target_size=(96, 96, 96),  list_target_indexes=(0, 1)),
            RandomFlip(list_p_per_axis=[0., 0.5, 0.5, 0.5], p=0.5, list_target_indexes=(0, 1)),

        )

    def __call__(self, list_images):
        list_images = self.transforms(list_images)

        return list_images


class Step4PostProcessor:
    def __init__(self):
        pass

    def __call__(self, list_images):
        """
        All images in this function forced to be C*Z*Y*X
        """
        image = list_images[0]
        seg = list_images[1]
        fracture = np.array(list_images[2])

        # Final outputs
        list_images = [np.concatenate((image, seg), axis=0),
                       fracture
                       ]

        list_images = to_tensor(list_images)

        return list_images


def _visualize_augmented_image(list_images, save_path):
    spacing = (1.0, 1.0, 1.0)

    image = list_images[0].cpu().data[0, 0, :, :, :]
    image_nii = sitk.GetImageFromArray(np.float32(image))
    image_nii.SetSpacing(spacing)

    seg = list_images[0].cpu().data[0, 1, :, :, :]
    seg_nii = sitk.GetImageFromArray(np.uint8(seg))
    seg_nii.SetSpacing(spacing)

    print(list_images[1].cpu().numpy()[0])

    try_recursive_mkdir(save_path)

    sitk.WriteImage(image_nii, save_path + '/image.nii.gz')
    sitk.WriteImage(seg_nii, save_path + '/seg.nii.gz')


if __name__ == '__main__':
    import pickle
    import time
    from Utils.CommonTools.random_seed import set_random_seed
    from Utils.CommonTools.dir import try_recursive_mkdir

    set_random_seed(2022)

    dataset = TrainDataset(source_dir=setting.path_training_data,
                           dataset_info=pickle.load(open(setting.path_training_data_info, 'rb'))['no_val'],
                           num_samples_per_epoch=50
                           )
    train_loader = data.DataLoader(dataset=dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last=False,
                                   prefetch_factor=2,
                                   persistent_workers=True
                                   )

    time_start = time.time()
    for i in range(100):
        for batch_idx, list_images_ in enumerate(train_loader):
            # time.sleep(0.1)
            print(batch_idx)
            print(list_images_[0].shape)

            _visualize_augmented_image(list_images_,
                                       save_path='G:/Aug/' + str(batch_idx))

    print(time.time() - time_start)
