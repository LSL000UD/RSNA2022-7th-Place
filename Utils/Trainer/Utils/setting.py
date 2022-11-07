# -*- encoding: utf-8 -*-
class TrainerSetting:
    def __init__(self,
                 task_name= None,
                 output_dir=None,
                 list_GPU_ids=-1,
                 max_iter=99999999,
                 iter_per_epoch=500,
                 eps_train_loss=0.01,
                 list_save_in_n_iter=(-99999999, ),

                 continue_training=True,
                 continue_with_model_weight_only = False,

                 batch_size=1,
                 shuffle=True,
                 num_workers=2,
                 pin_memory=True,
                 drop_last=False,
                 prefetch_factor=2,
                 persistent_workers=True,

                 list_val_epoch=(0, ),

                 use_swa=False,
                 swa_start=99999999,
                 swa_freq=99999999,
                 swa_update_bn=True,

                 input_num=1,

                 save_prefix='fold_0',

                 return_iter_to_model=False
                 ):
        """
        :param task_name:
        :param output_dir: path for saving models and log
        :param list_GPU_ids: the gpu list for training, main gpu id will be list_GPU_ids[0], -1 means using cpu
        :param max_iter:
        :param iter_per_epoch:
        :param eps_train_loss:
        :param list_save_in_n_iter: save model after n iter if n in list_save_in_n_iter
        :param continue_training:
        :param continue_with_model_weight_only
        :param batch_size:
        :param shuffle:
        :param num_workers:
        :param pin_memory:
        :param drop_last:
        :param prefetch_factor:
        :param persistent_workers:
        :param list_val_epoch:
        :param use_swa:
        :param swa_start:
        :param swa_freq:
        :param swa_update_bn:
        :param input_num: sometimes the model has multiple inputs
        :param save_prefix:
        """
        self.task_name = task_name
        self.output_dir = output_dir
        self.list_save_in_n_iter = list_save_in_n_iter
        self.eps_train_loss = eps_train_loss

        self.device = None
        self.list_GPU_ids = list_GPU_ids
        self.max_iter = max_iter
        self.iter_per_epoch = iter_per_epoch

        self.continue_training = continue_training
        self.continue_with_model_weight_only = continue_with_model_weight_only

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        self.list_val_epoch = list_val_epoch

        self.use_swa = use_swa
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_update_bn = swa_update_bn

        self.input_num = input_num

        self.save_prefix = save_prefix

        self.return_iter_to_model = return_iter_to_model


