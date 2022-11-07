"""
1. nnunet/utilities/task_name_id_conversion.py,  line22
    change task name from Task001 to Task_1

2. nnunet/preprocessing/cropping.py, line 94
    maybe no cropping

3.nnunet/preprocessing/preprocessing.py, line 334
    maybe no re-sampling or normalization

4. nnunet/__init__.py
    no printing for paper info

5. nnunet/experiment_planning/experiment_planner_baseline_3DUNet.py, line 263
    always set transpose_forward to [0, 1, 2]

6. nnunet/configuration.py
    change default_num_threads

7.nnunet/training/network_training/nnUNetTrainerV2.py, line 49
    change max_num_epochs

8.nnunet/training/network_training/network_trainer.py， line 101
    set num_val_batches_per_epoch = 1， the validation is not promising, therefore use no validation

9.nnunet/training/network_training/network_trainer.py, line 128,
    change mechanism of saving checkpoint
    self.save_best_checkpoint = True -> False

10. nnunet/preprocessing/cropping.py, line 52, line59, line 127
    inorder to run in windows, replace('\\', '/')

11. nnunet/training/network_training/nnUNetTrainer.py
    self.oversample_foreground_percent = 0.33 ->

12. nnunet/training/dataloading/dataset_loading.py line 182
    print total training cases in the begin of training

13. nnunet/run/default_configuration.py
        manually set         plans_file, dataset_directory, batch_dice, stage

14. nnunet/training/network_training/nnUNetTrainerV2.py, line 52, line 179
    can choose Adam or SGD
"""