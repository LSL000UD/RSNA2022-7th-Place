import pickle
import numpy as np
import os

import setting


path_plan = f"{setting.path_nnunet_preprocessing_output_dir}/{setting.task_name}/nnUNetPlansv2.1_plans_3D.pkl"

plan = pickle.load(open(path_plan, 'rb'))
plan['base_num_features'] = setting.base_num_features
plan['plans_per_stage'][setting.plan_stage]['batch_size'] = setting.batch_size
plan['plans_per_stage'][setting.plan_stage]['patch_size'] = setting.patch_size
plan['plans_per_stage'][setting.plan_stage]['num_pool_per_axis'] = setting.num_pool_per_axis

plan['plans_per_stage'][setting.plan_stage]['pool_op_kernel_sizes'] = setting.pool_op_kernel_sizes
plan['plans_per_stage'][setting.plan_stage]['conv_kernel_sizes'] = setting.conv_kernel_sizes
plan['plans_per_stage'][setting.plan_stage]['do_dummy_2D_data_aug'] = setting.do_dummy_2D_data_aug

pickle.dump(plan, open(path_plan, 'wb'))
print(f"==> Remake plan {path_plan}, stage: {setting.plan_stage}")
