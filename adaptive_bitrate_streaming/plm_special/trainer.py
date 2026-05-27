import numpy as np
import torch
import time

from munch import Munch
from torch.utils.data import DataLoader

from plm_special.utils.utils import process_batch
from plm_special.utils.losses import compute_abr_train_loss


class Trainer:
    def __init__(self, args, model, optimizer, exp_dataset, loss_fn, device, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        self.loss_type = getattr(args, 'loss_type', 'ce')
        self.kd_alpha = getattr(args, 'kd_alpha', 0.5)
        self.kd_temperature = getattr(args, 'kd_temperature', 2.0)
        self.teacher_is_prob = getattr(args, 'teacher_is_prob', False)
        
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)

    def train_epoch(self, report_loss_per_steps=100):
        train_losses = []
        train_hard_losses = []
        train_soft_losses = []
        logs = dict()

        train_start = time.time()
        dataset_size = len(self.dataloader)

        self.model.train()
        for step, batch in enumerate(self.dataloader):
            train_loss, step_info = self.train_step(batch)
            train_losses.append(train_loss.item())
            if self.loss_type == 'ce_kl':
                train_hard_losses.append(step_info['loss_hard'])
                train_soft_losses.append(step_info['loss_soft'])

            # perform gradient accumulation update
            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == dataset_size):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if step % report_loss_per_steps == 0:
                mean_train_loss = np.mean(train_losses)
                msg = f'Step {step} - mean train loss {mean_train_loss:>9f}'
                if self.loss_type == 'ce_kl' and train_hard_losses:
                    msg += (
                        f'  (hard {np.mean(train_hard_losses):>9f}'
                        f'  soft {np.mean(train_soft_losses):>9f})'
                    )
                print(msg)

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        if self.loss_type == 'ce_kl' and train_hard_losses:
            logs['training/train_loss_hard_mean'] = np.mean(train_hard_losses)
            logs['training/train_loss_soft_mean'] = np.mean(train_soft_losses)

        return logs, train_losses

    def train_step(self, batch):
        expect_teacher = self.loss_type == 'ce_kl'
        states, actions, returns, timesteps, labels, teacher_logits = process_batch(
            batch, device=self.device, expect_teacher_logits=expect_teacher
        )
        actions_pred = self.model(states, actions, returns, timesteps)
        actions_pred = actions_pred.permute(0, 2, 1)
        loss, info = compute_abr_train_loss(
            actions_pred,
            labels,
            self.loss_fn,
            loss_type=self.loss_type,
            teacher_logits=teacher_logits,
            kd_alpha=self.kd_alpha,
            kd_temperature=self.kd_temperature,
            teacher_is_prob=self.teacher_is_prob,
        )
        return loss, info
