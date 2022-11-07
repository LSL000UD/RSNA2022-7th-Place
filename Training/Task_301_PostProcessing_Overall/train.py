# -*- encoding: utf-8 -*-
import os
import sys
import pickle

src_path = os.path.abspath(os.path.join(__file__, '../../../'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from Utils.Trainer.seg_trainer import NetworkTrainer
from Utils.Trainer.Utils.setting import TrainerSetting
from Utils.Optimizers import Adam
from Utils.LrSchedulers.cosine_scheduler import CosineScheduler
from Utils.CommonTools.random_seed import set_random_seed

from model import Model
from online_evaluation import OnlineEvaluation
from loss import Loss
from dataloader import TrainDataset

import setting


if __name__ == '__main__':
    set_random_seed(111)

    for fold in setting.training_folds:
        # Setting
        trainer_setting = TrainerSetting(
            task_name=setting.task_name,
            output_dir=setting.path_output,
            list_GPU_ids=setting.list_GPU_ids,
            max_iter=setting.max_iter,
            iter_per_epoch=setting.iter_per_epoch,

            eps_train_loss=0.01,
            list_save_in_n_iter=setting.list_save_in_n_iter,

            continue_training=setting.continue_training,
            continue_with_model_weight_only=setting.continue_with_model_weight_only,

            batch_size=setting.train_bs,
            num_workers=setting.num_works,
            pin_memory=setting.pin_memory,
            prefetch_factor=setting.prefetch_factor,
            persistent_workers=setting.persistent_workers,

            list_val_epoch=setting.list_val_epoch,

            use_swa=False,
            swa_start=99999999,
            swa_freq=99999999,
            swa_update_bn=False,

            input_num=setting.input_num,

            save_prefix=fold,  # Save with fold name

            return_iter_to_model=True
        )

        # Model
        model = Model(in_ch=setting.in_ch, out_ch=setting.out_ch, list_ch=setting.list_ch, random_init=True)

        # Dataset
        dataset_info = pickle.load(open(setting.path_training_data_info, 'rb'))
        dataset = TrainDataset(
            source_dir=setting.path_training_data,
            dataset_info=dataset_info[fold],  # Pick a fold here
            num_samples_per_epoch=setting.train_bs * setting.iter_per_epoch
        )

        online_evaluation = OnlineEvaluation(
            source_dir=setting.path_training_data,
            dataset_info=dataset_info
        )

        # Optimizer
        optimizer = Adam(lrs=setting.lrs, weight_decay=3e-5, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

        # Learning rate scheduler
        lr_scheduler = CosineScheduler(T_max=setting.max_iter, eta_min=1e-8, last_epoch=-1)

        #  Start training
        trainer = Utils(
            network=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_function=Loss(),
            dataset=dataset,
            online_evaluation=online_evaluation,
            setting=trainer_setting
        )

        trainer.run()
