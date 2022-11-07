# -*- encoding: utf-8 -*-
class TrainerStatus:
    def __init__(self):
        self.iter = -1
        self.epoch = -1

        # Save status of the trainer, eg. "best_train_loss", "latest", "best_val_index"
        self.save_status = []

        self.moving_train_loss = None
        self.best_moving_train_loss = 99999999.

    def update_moving_train_loss(self, loss, eps_train_loss):
        if self.moving_train_loss is None:
            self.moving_train_loss = loss
            self.best_moving_train_loss = loss
        else:
            self.moving_train_loss = (1 - eps_train_loss) * self.moving_train_loss + eps_train_loss * loss
            if self.moving_train_loss <= self.best_moving_train_loss:
                self.best_moving_train_loss = self.moving_train_loss

