# -*- encoding: utf-8 -*-
import time
import os

import torch
import torch.nn as nn
import torch.utils.data as data

from Utils.Trainer.Utils.status import TrainerStatus
from Utils.Trainer.Utils.statistics import TrainerStatistics
from Utils.Trainer.Utils.swa import SWA
from Utils.Trainer.Utils.print_info import print_log_to_text_file

from Utils.CommonTools.timer import timer
from Utils.CommonTools.torch_base import batch_tensor_to_device
from Utils.CommonTools.dir import try_recursive_mkdir


class NetworkTrainer:
    def __init__(self,
                 network,
                 optimizer,
                 lr_scheduler,
                 loss_function,
                 dataset,
                 online_evaluation,
                 setting
                 ):
        """
        :param network: the torch model
        :param optimizer: see Utils/Optimizers
        :param lr_scheduler: see Utils/LrSchedulers
        :param loss_function:
        :param dataset:
        :param online_evaluation:
        :param setting:
        """
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function
        self.data_loader = None
        self.online_evaluation = online_evaluation
        self.setting = setting

        self.status = TrainerStatus()
        self.statistics = TrainerStatistics()

        self.init_output_dir()
        self.init_optimizer()
        self.init_lr_scheduler()
        self.init_dataloader(dataset)
        self.init_GPU_devices()
        self.try_continue_training()

        if not hasattr(network, "info"):
            self.network.info = ''

    def init_output_dir(self):
        try_recursive_mkdir(self.setting.output_dir + '/' + self.setting.save_prefix)

    def init_dataloader(self, dataset):
        self.data_loader = data.DataLoader(dataset=dataset,
                                           batch_size=self.setting.batch_size,
                                           shuffle=self.setting.shuffle,
                                           num_workers=self.setting.num_workers,
                                           pin_memory=self.setting.pin_memory,
                                           drop_last=self.setting.drop_last,
                                           prefetch_factor=self.setting.prefetch_factor,
                                           persistent_workers=self.setting.persistent_workers
                                           )

    def init_optimizer(self):
        if hasattr(self.network, 'decoder') and hasattr(self.network, 'encoder'):
            list_param_groups = [self.network.encoder.parameters(),
                                 self.network.decoder.parameters()]
        else:
            list_param_groups = [self.network.parameters()]

        self.optimizer = self.optimizer.get_optimizer(list_param_groups)
        self.print_optimizer()

        if self.setting.use_swa:
            self.optimizer = SWA(self.optimizer)
            self.print_swa_usage()

    def init_lr_scheduler(self):
        self.lr_scheduler = self.lr_scheduler.get_scheduler(self.optimizer)
        self.print_lr_scheduler()

    def init_GPU_devices(self):
        self.set_GPU_devices(self.setting.list_GPU_ids)

    def try_continue_training(self):
        if self.setting.continue_training and \
                os.path.exists(self.setting.output_dir + '/' + self.setting.save_prefix + '/latest_model.pkl'):
            self.load_trainer(
                self.setting.output_dir + '/' + self.setting.save_prefix + '/latest_model.pkl',
                self.setting.output_dir + '/' + self.setting.save_prefix + '/latest_info.pkl',
                model_only=self.setting.continue_with_model_weight_only
            )

    def run(self):
        # Start training
        self.print_start_training()

        while self.status.iter < self.setting.max_iter - 1:
            self.status.epoch += 1

            # Run a epoch
            self.print_training_status()
            _, time_epoch = self.run_one_epoch()

            # Start saving
            for status_i in range(len(self.status.save_status)):
                status = self.status.save_status[status_i]
                _, time_save_trainer = self.save_trainer(status=status)

                if status_i == 0:
                    self.statistics.time.save_trainer_epoch.append(
                        {'value': time_save_trainer, 'iter': self.status.iter})
                else:
                    self.statistics.time.save_trainer_epoch[-1]['value'] += time_save_trainer

            # Update time usage
            self.statistics.time.time_from_start += time_epoch
            self.statistics.time.time_from_start += self.statistics.time.save_trainer_epoch[-1]['value']

            self.print_average_train_loss()
            self.print_average_val_index()
            self.print_iter_best_train()
            self.print_iter_best_val()
            self.print_save_status()
            self.print_time_summary()

        if self.setting.use_swa:
            self.get_final_swa_model()

        self.plot_loss(trainer_statistics=self.statistics,
                       save_path=self.setting.output_dir + '/' + self.setting.save_prefix)

        self.print_ending_message()

    @timer
    def run_one_epoch(self):
        # Train
        _, train_use_time = self.train_epoch()
        # Val
        _, val_use_time = self.val_epoch()

        # Update time usage
        self.statistics.time.train_epoch.append({'value': train_use_time, 'iter': self.status.iter})
        self.statistics.time.val_epoch.append({'value': val_use_time, 'iter': self.status.iter})

        # Update saving status
        self.status.save_status.clear()

        if self.statistics.performance.best_train_ptr == len(self.statistics.performance.average_train) - 1:
            self.status.save_status.append('best_train')

        if self.statistics.performance.best_val_ptr == len(self.statistics.performance.average_val) - 1:
            self.status.save_status.append('best_val')

        if self.status.iter in self.setting.list_save_in_n_iter:
            self.status.save_status.append('iter_{:d}'.format(self.status.iter))

        self.status.save_status.append('latest')

    @timer
    def train_epoch(self):
        self.network.train()

        sum_loss = 0.
        count_iter = 0
        for batch_idx, list_loader_output in enumerate(self.data_loader):
            # Break if training is end
            if self.status.iter + 1 > self.setting.max_iter - 1:
                break
            self.status.iter += 1

            # Tensor to GPU
            input_ = list_loader_output[0:self.setting.input_num]
            target = list_loader_output[self.setting.input_num:]

            input_, time_input_to_device = batch_tensor_to_device(input_, device=self.setting.device, to_float=True)
            target, time_target_to_device = batch_tensor_to_device(target, device=self.setting.device, to_float=False)

            # Forward and Backward
            output, time_forward = self.forward(input_)
            loss, time_backward = self.backward(output, target)
            loss_item = loss.item()

            # Print moving train loss
            self.status.update_moving_train_loss(loss_item, self.setting.eps_train_loss)
            self.print_moving_train_loss(batch_idx)

            # Update forward, backward time usage
            if batch_idx == 0:
                self.statistics.time.forward_epoch.append({'value': time_forward, 'iter': self.status.iter})
                self.statistics.time.backward_epoch.append({'value': time_backward, 'iter': self.status.iter})
            else:
                self.statistics.time.forward_epoch[-1]['value'] += time_forward
                self.statistics.time.forward_epoch[-1]['iter'] = self.status.iter
                self.statistics.time.backward_epoch[-1]['value'] += time_backward
                self.statistics.time.backward_epoch[-1]['iter'] = self.status.iter

            # Update learning rate and swa
            self.update_lr()

            self.update_swa()

            sum_loss += loss_item
            count_iter += 1

        if count_iter > 0:
            self.statistics.performance.update_average_train(sum_loss / count_iter, self.status.iter)

    @timer
    def val_epoch(self):
        if self.status.epoch in self.setting.list_val_epoch or self.setting.list_val_epoch is None:
            with torch.no_grad():
                self.network.eval()
                val_index = self.online_evaluation(self.network, device=self.setting.device)
        else:
            val_index = -999.
        self.statistics.performance.update_average_val(val_index, self.status.iter)

    @timer
    def forward(self, input_):
        self.optimizer.zero_grad()
        if self.setting.return_iter_to_model:
            output = self.network(input_, self.status.iter)
        else:
            output = self.network(input_)
        return output

    @timer
    def backward(self, output, target):
        loss = self.loss_function(output, target)
        loss.backward()
        self.optimizer.step()
        return loss

    def update_lr(self):
        # Update learning rate, only 'ReduceLROnPlateau' need use the moving train loss
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.status.moving_train_loss)
        else:
            self.lr_scheduler.step()

    def update_swa(self):
        if self.setting.use_swa and (self.status.iter >= self.setting.swa_start) and (
                self.status.iter - self.setting.swa_start) % self.setting.swa_freq == 0:
            self.optimizer.update_swa()
            self.print_swa_update()

    def get_final_swa_model(self):
        self.optimizer.swap_swa_sgd()
        self.print_swa_final_weight_update()

        if self.setting.swa_update_bn:
            self.optimizer.bn_update(self.data_loader, self.network, device=self.setting.device)
        self.print_swa_final_bn_update()
        self.save_trainer(status='SWA')

    def set_GPU_devices(self, list_GPU_ids):
        self.setting.list_GPU_ids = list_GPU_ids
        # cpu only
        if list_GPU_ids == -1:
            self.setting.device = torch.device('cpu')
        else:
            self.setting.device = torch.device('cuda:' + str(list_GPU_ids[0]))
            # multi-GPU
            if len(list_GPU_ids) > 1:
                self.network = nn.DataParallel(self.network, device_ids=list_GPU_ids)

        self.network = self.network.to(self.setting.device)
        self.print_GPU_ids()

    @timer
    def save_trainer(self, status='latest'):
        # Model weights
        if isinstance(self.setting.list_GPU_ids, list) and len(self.setting.list_GPU_ids) > 1:
            network_state_dict = self.network.module.state_dict()
        else:
            network_state_dict = self.network.state_dict()
        torch.save(network_state_dict, self.setting.output_dir + '/' + self.setting.save_prefix + '/' + status + '_model.pkl')

        # Training status, only for latest model
        if status == 'latest':
            optimizer_state_dict = self.optimizer.state_dict()
            lr_scheduler_state_dict = self.lr_scheduler.state_dict()

            ckpt_training = {
                'lr_scheduler_state_dict': lr_scheduler_state_dict,
                'optimizer_state_dict': optimizer_state_dict,  # Can be 2~3 times size as model weights
                'status': self.status,
                'statistics': self.statistics
            }
            torch.save(ckpt_training, self.setting.output_dir + '/' + self.setting.save_prefix + '/' + status + '_info.pkl')

    def load_trainer(self, ckpt_model, ckpt_info, model_only=False):
        self.network = self.load_model_weights(self.network, ckpt_model)
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> Init trainer model weight from {:s} ! \n'.format(ckpt_model),
                               'a', True
                               )

        if not model_only:
            self.load_training_info(ckpt_info)
            print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                                   '==> Init trainer info from {:s} ! \n'.format(ckpt_info),
                                   'a', True
                                   )

    @staticmethod
    def load_model_weights(network, ckpt_file):
        network.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        return network

    def load_training_info(self, ckpt_file):
        ckpt = torch.load(ckpt_file, map_location='cpu')

        self.status = ckpt['status']
        self.statistics = ckpt['statistics']
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # If do not do so, the states of optimizer will always in cpu
        self.move_optimizer_state_to_GPU(self.optimizer, self.setting.device)

        if self.setting.use_swa:
            self.move_optimizer_state_to_GPU(self.optimizer.optimizer, self.setting.device)

    @staticmethod
    def move_optimizer_state_to_GPU(optimizer, device):
        for _, parameter in optimizer.state.items():
            for tensor_name in parameter:
                if isinstance(parameter[tensor_name], torch.Tensor):
                    parameter[tensor_name] = parameter[tensor_name].to(device)

    @staticmethod
    def plot_loss(trainer_statistics, save_path):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 10))

        # Train
        ax1 = fig.add_subplot(111)
        list_value = []
        list_iter = []
        for i in range(len(trainer_statistics.performance.average_train)):
            list_iter.append(trainer_statistics.performance.average_train[i]['iter'])
            list_value.append(trainer_statistics.performance.average_train[i]['value'])
        ax1.plot(list_iter, list_value, color='red')

        # Val
        list_value = []
        list_iter = []
        for i in range(len(trainer_statistics.performance.average_val)):
            if trainer_statistics.performance.average_val[i]['value'] > -999:
                list_iter.append(trainer_statistics.performance.average_val[i]['iter'])
                list_value.append(trainer_statistics.performance.average_val[i]['value'])
        if len(list_value) > 0:
            ax2 = ax1.twinx()
            ax2.plot(list_iter, list_value)

        plt.savefig(save_path + '/loss.png', dpi=600)
        plt.close(fig)

    ################################################
    def print_GPU_ids(self):
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> Current GPUs: {:<s} \n'.format(str(self.setting.list_GPU_ids)),
                               'a',
                               True)

    def print_optimizer(self):
        optimizer_type = type(self.optimizer)
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> Optimizer {:<s} \n'.format(str(optimizer_type)),
                               'a',
                               True)

    def print_swa_usage(self):
        if self.setting.use_swa:
            print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                                   '==> Using Stochastic Weight Averaging(SWA) ! \n',
                                   'a',
                                   True)

    def print_swa_update(self):
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '\n==> Update SWA after iter {:<d}! |\n'.format(self.status.iter),
                               mode='a',
                               terminal_only=True
                               )

    def print_swa_final_weight_update(self):
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> Employ SWA weight!\n',
                               mode='a',
                               terminal_only=True
                               )

    def print_swa_final_bn_update(self):
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> Update BN for SWA!\n',
                               mode='a',
                               terminal_only=True
                               )

    def print_lr_scheduler(self):
        lr_schedule_type = type(self.lr_scheduler)
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> Lr_scheduler {:<s} \n'.format(str(lr_schedule_type)),
                               'a',
                               True)

    def print_start_training(self):
        if self.status.iter == -1:
            print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                                   '==> Start training from 0 iter !\n', 'a', terminal_only=True)
        else:
            print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                                   '==> Continue training !\n', 'a',
                                   terminal_only=True)

        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               time.strftime('==> Current time: %H:%M:%S\n\n', time.localtime(time.time())),
                               'a', terminal_only=True)

    def print_moving_train_loss(self, batch_idx):
        if (self.setting.iter_per_epoch < 100) or (batch_idx + 1) % (
                int(self.setting.iter_per_epoch / 100)) == 0:

            print_log_to_text_file(
                self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                '\r| TRAINING: {:<3d}% {:<10s} Moving train loss: {:<11.6f}                     |'
                .format(int((batch_idx + 1) / self.setting.iter_per_epoch * 100),
                        '',
                        self.status.moving_train_loss
                        ),
                mode='a',
                terminal_only=True
            )

    # ----------------- Final output log --------------------#

    def print_training_status(self):
        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n\n\n'
            '+==============================================================================+\n'
            '|  {:<22s}   START TIME: {:<39s}|\n'
            '|  FOLD: {:<16s}   BATCH SIZE: {:<5d}            GPUs: {:<12s}    |\n'
            '|  EPOCH: {:<16s}  MODEL INFO: {:<35s}    |\n'
            '|  ITER: {:<16s}   LEARNING RATE: {:<14.10f}, {:<14.10f}      |\n'
            '|==============================================================================|\n'
            .format(
                # Project Name, Fold
                self.setting.task_name,
                time.strftime('%H:%M:%S', time.localtime(time.time())),


                # Fold, Batch size, GPUs
                self.setting.save_prefix,
                self.setting.batch_size,
                str(self.setting.list_GPU_ids),

                # Epoch, Start time
                str(self.status.epoch) + '/' + str(int(self.setting.max_iter / self.setting.iter_per_epoch - 1)),
                self.network.info,

                # Iter, Learning rate
                str(self.status.iter + 1) + '/' + str(self.setting.max_iter - 1),
                self.optimizer.param_groups[0]['lr'],
                self.optimizer.param_groups[-1]['lr']
            ),
            mode='a',
            terminal_only=False
        )

    def print_average_train_loss(self):
        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n|                           Avg Train(current/best): {:<11.6f}/{:>11.6f}   |'
            .format(self.statistics.performance.average_train[-1]['value'],
                    self.statistics.performance.average_train[self.statistics.performance.best_train_ptr]['value']
                    ),
            mode='a',
            terminal_only=False
        )

    def print_average_val_index(self):
        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n|                           Avg Val(current/best): {:<11.6f}/{:>11.6f}     |'
            .format(self.statistics.performance.average_val[-1]['value'],
                    self.statistics.performance.average_val[self.statistics.performance.best_val_ptr]['value']
                    ),
            mode='a',
            terminal_only=False
        )

    def print_iter_best_train(self):
        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n|                           Best Train Iter: {:<12d}                      |'
            .format(self.statistics.performance.average_train[self.statistics.performance.best_train_ptr]['iter']
                    ),
            mode='a',
            terminal_only=False
        )

    def print_iter_best_val(self):
        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n|                           Best Val Iter: {:<12d}                        |'
            .format(self.statistics.performance.average_val[self.statistics.performance.best_val_ptr]['iter']
                    ),
            mode='a',
            terminal_only=False
        )

    def print_save_status(self):
        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n|==============================================================================|\n'
            '|  CheckPointSize      BestTrain      BestVal      Latest      EveryNIter      |\n'
            '|   {:^8.2f} MB          {:<3s}            {:<3s}          {:<3s}           {:<3s}	       |'
            .format(os.path.getsize(
                self.setting.output_dir + '/' + self.setting.save_prefix + '/latest_model.pkl') / 1024. / 1024.,
                    'YES' if 'best_train' in self.status.save_status else 'NO',
                    'YES' if 'best_val' in self.status.save_status else 'NO',
                    'YES' if 'latest' in self.status.save_status else 'NO',
                    'YES' if 'iter_' + str(self.status.iter) in self.status.save_status else 'NO'
                    ),
            mode='a',
            terminal_only=False
        )

    def print_time_summary(self):
        all_time = self.statistics.time.train_epoch[-1]['value'] + self.statistics.time.val_epoch[-1]['value'] \
                   + self.statistics.time.save_trainer_epoch[-1]['value']

        print_log_to_text_file(
            self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
            '\n|==============================================================================|\n'
            '|  USE TIME(seconds)       EPOCH:                        {:<10.2f}(100%)      |\n'
            '|                          TRAIN:                        {:<10.2f}({:<4.1f}%)     |\n'
            '|                          VAL:                          {:<10.2f}({:<4.1f}%)     |\n'
            '|                          SAVE:                         {:<10.2f}({:<4.1f}%)     |\n'
            '|                                                                              |\n'
            '|                          FORWARD+BACKWARD/TRAIN:       {:<4.1f}%                 |\n'
            '|                                                                              |\n'
            '|  USE TIME(hours)         ALL TIME:                     {:<10.2f}            |\n'
            '|                          REMAIN TIME:                  {:<10.2f}            |\n'
            '+==============================================================================+\n'
            .format(
                all_time,

                self.statistics.time.train_epoch[-1]['value'],
                self.statistics.time.train_epoch[-1]['value'] / (all_time + 1e-6) * 100.,

                self.statistics.time.val_epoch[-1]['value'],
                self.statistics.time.val_epoch[-1]['value'] / (all_time + 1e-6) * 100.,

                self.statistics.time.save_trainer_epoch[-1]['value'],
                self.statistics.time.save_trainer_epoch[-1]['value'] / (all_time + 1e-6) * 100.,

                (self.statistics.time.forward_epoch[-1]['value'] + self.statistics.time.backward_epoch[-1]['value']) / (self.statistics.time.train_epoch[-1]['value'] + 1e-6) * 100.,

                self.statistics.time.time_from_start / 3600.,
                all_time / self.setting.iter_per_epoch * (self.setting.max_iter - 1 - self.status.iter) / 3600.
            ),
            mode='a',
            terminal_only=False
        )

    def print_ending_message(self):
        print_log_to_text_file(self.setting.output_dir + '/' + self.setting.save_prefix + '/log.txt',
                               '==> End successfully, totally use time         {:<20.2f} seconds\n\n\n\n\n'.format(self.statistics.time.time_from_start), 'a', False)
