# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import warnings

import torch

from batchgenerators.transforms.abstract_transforms import Compose


from Utils.fake_fracture_generation import FakeFractureGenerator


class CustomTransform(Compose):
    def __init__(self, transforms):
        super(CustomTransform, self).__init__(transforms)
        print(f"==> Using custom transform from {__file__}")

        self.fake_fracture_generator = FakeFractureGenerator()

        # Use Nearest model for all spatial transform
        for t_i in range(len(self.transforms)):
            t = self.transforms[t_i]
            if t.__class__.__name__ == 'SpatialTransform':
                self.transforms[t_i].order_data = 0
                self.transforms[t_i].angle_z = (-0.0, 0.0)
                self.transforms[t_i].angle_y = (-0.0, 0.0)

    def __call__(self, **data_dict):
        for t in self.transforms:
            list_aug_for_all_channel = ['SpatialTransform', 'MirrorTransform', 'NumpyToTensor']
            if t.__class__.__name__ in list_aug_for_all_channel:
                warnings.warn(f'==> {t.__class__.__name__} will used for all channel !!!!!!!!!!!')
                data_dict = t(**data_dict)
            else:
                tmp = data_dict['data'].copy()
                data_dict = t(**data_dict)
                data_dict['data'][:, 1:] = tmp[:, 1:]

        # Fake fracture generation
        if random.random() <= 0.5:

            for batch_i in range(data_dict['data'].shape[0]):
                ct = data_dict['data'][batch_i, 0].numpy()
                seg = data_dict['data'][batch_i, 1].numpy()
                gt = data_dict['target'][0][batch_i, 0].numpy()

                ct, gt = self.fake_fracture_generator(ct, seg, gt)
                data_dict['data'][batch_i, 0] = torch.from_numpy(ct)
                data_dict['target'][0][batch_i, 0] = torch.from_numpy(gt)

        data_dict['data'] = data_dict['data'][:, 0:1, :, :, :]

        return data_dict

    def __repr__(self):
        return str(type(self).__name__) + " ( " + repr(self.transforms) + " )"
