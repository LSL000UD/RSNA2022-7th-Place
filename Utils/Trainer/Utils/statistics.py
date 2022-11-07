# -*- encoding: utf-8 -*-


class TrainerStatistics:
    def __init__(self):
        self.performance = Performance()
        self.time = Time()


class Performance:
    def __init__(self):
        self.average_train = []
        self.best_train_ptr = 0  # Index of best train in self.average_train

        self.average_val = []
        self.best_val_ptr = 0

    def update_average_train(self, statistic_value, cor_iter):
        self.average_train.append({'value': statistic_value, 'iter': cor_iter})

        if statistic_value < self.average_train[self.best_train_ptr]['value']:
            self.best_train_ptr = len(self.average_train) - 1

    def update_average_val(self, statistic_value, cor_iter):
        self.average_val.append({'value': statistic_value, 'iter': cor_iter})

        if statistic_value > self.average_val[self.best_val_ptr]['value']:
            self.best_val_ptr = len(self.average_val) - 1


class Time:
    def __init__(self):

        self.train_epoch = []
        self.val_epoch = []
        self.save_trainer_epoch = []
        self.forward_epoch = []
        self.backward_epoch = []

        self.time_from_start = 0

    def update_epoch(self, epoch, train, val, save_trainer, forward, backward, cor_iter):
        self.train_epoch.append({'value': train, 'iter': cor_iter})
        self.val_epoch.append({'value': val, 'iter': cor_iter})
        self.save_trainer_epoch.append({'value': save_trainer, 'iter': cor_iter})
        self.forward_epoch.append({'value': forward, 'iter': cor_iter})
        self.backward_epoch.append({'value': backward, 'iter': cor_iter})



