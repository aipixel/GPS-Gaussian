
import os
import json
import shutil
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def file_backup(exp_path, cfg, train_script):
    shutil.copy(train_script, exp_path)
    shutil.copytree('core', os.path.join(exp_path, 'core'), dirs_exist_ok=True)
    shutil.copytree('config', os.path.join(exp_path, 'config'), dirs_exist_ok=True)
    shutil.copytree('gaussian_renderer', os.path.join(exp_path, 'gaussian_renderer'), dirs_exist_ok=True)
    for sub_dir in ['lib']:
        files = os.listdir(sub_dir)
        for file in files:
            Path(os.path.join(exp_path, sub_dir)).mkdir(exist_ok=True, parents=True)
            if file[-3:] == '.py':
                shutil.copy(os.path.join(sub_dir, file), os.path.join(exp_path, sub_dir))

    json_file_name = exp_path + '/cfg.json'
    with open(json_file_name, 'w') as json_file:
        json.dump(cfg, json_file, indent=2)


class Logger:
    def __init__(self, scheduler, cfg):
        self.scheduler = scheduler
        self.sum_freq = cfg.loss_freq
        self.log_dir = cfg.logs_path
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / self.sum_freq for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / self.sum_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps and self.total_steps % self.sum_freq == 0:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def write_dict(self, results, write_step):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar(key, results[key], write_step)

    def close(self):
        self.writer.close()
