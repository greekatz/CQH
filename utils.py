import numpy as np
import logging
from datetime import datetime
import os 
from tensorboardX import SummaryWriter
import shutil
import torch

def read_and_parse_file(file_path):
    data_tbl = np.loadtxt(file_path, dtype=str)
    data, targets = data_tbl[:, 0], data_tbl[:, 1:].astype(np.int8)
    return data, targets


def set_logger(config):
    os.makedirs("./logs/", exist_ok=True)
    if config.notes:
        prefix = config.notes
    else:
        prefix = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_file = os.path.join('./logs/', prefix + '.log')

    log_directory = os.path.dirname(os.path.join('./logs/', prefix))
    os.makedirs(log_directory, exist_ok=True)
    # config.__dict__['checkpoint_root'] = os.path.join('./checkpoints/', prefix)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s.',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])
    if config.use_writer:
        writer_root = os.path.join('./logs/', prefix + '.writer')
        if os.path.exists(writer_root):
            shutil.rmtree(writer_root)
        writer = SummaryWriter(writer_root)        
    else:
        writer = None 
    return writer


class WarmUpAndCosineDecayScheduler:
    def __init__(self, optimizer, start_lr, base_lr, final_lr,
                 epoch_num, batch_num_per_epoch, warmup_epoch_num):
        self.optimizer = optimizer
        self.step_counter = 0
        warmup_step_num = batch_num_per_epoch * warmup_epoch_num
        decay_step_num = batch_num_per_epoch * (epoch_num - warmup_epoch_num)
        warmup_lr_schedule = np.linspace(start_lr, base_lr, warmup_step_num)
        cosine_lr_schedule = final_lr + 0.5 * \
            (base_lr - final_lr) * (1 + np.cos(np.pi *
                                               np.arange(decay_step_num) / decay_step_num))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    # step at each mini-batch
    def step(self):
        curr_lr = self.lr_schedule[self.step_counter]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = curr_lr
        self.step_counter += 1
        return curr_lr
    
    def state_dict(self):
        """Save scheduler state for checkpointing"""
        return {
            'step_counter': self.step_counter,
            'lr_schedule': self.lr_schedule
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.step_counter = state_dict['step_counter']
        self.lr_schedule = state_dict['lr_schedule']
    

class Monitor:
    def __init__(self, max_patience=5, delta=1e-6, save_dir='./checkpoints/'):
        self.counter_ = 0
        self.best_value = 0
        self.max_patience = max_patience
        self.patience = max_patience
        self.delta = delta
        self.save_dir = save_dir
        self.best_epoch = 0
        os.makedirs(save_dir, exist_ok=True)

    def update(self, cur_value, epoch=None, model=None, optimizer=None, scheduler=None):
        self.counter_ += 1
        is_break = False
        is_lose_patience = False
        
        if cur_value > self.best_value + self.delta:
            # New best value - save checkpoint
            self.best_value = cur_value
            self.best_epoch = epoch if epoch is not None else self.counter_
            self.patience = self.max_patience
            
            # Save checkpoint if model is provided
            if model is not None:
                self.save_checkpoint(model, optimizer, scheduler, epoch)
                
            logging.info(f"New best MAP: {self.best_value:.4f} at epoch {self.best_epoch}")
        else:
            # No improvement
            self.patience -= 1
            logging.info("the monitor loses its patience to %d!" % self.patience)
            is_lose_patience = True
            if self.patience == 0:
                self.patience = self.max_patience
                is_break = True
                
        return (is_break, is_lose_patience)

    def save_checkpoint(self, model, optimizer=None, scheduler=None, epoch=None):
        """Save checkpoint with best MAP"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_map': self.best_value,
            'best_epoch': self.best_epoch,
            'monitor_state': {
                'counter': self.counter_,
                'patience': self.patience,
                'best_value': self.best_value
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Save best checkpoint
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint to {best_path}")
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_model.pth')
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        """Load checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
            
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            # Restore monitor state
            if 'monitor_state' in checkpoint:
                monitor_state = checkpoint['monitor_state']
                self.counter_ = monitor_state['counter']
                self.patience = monitor_state['patience']
                self.best_value = monitor_state['best_value']
                
            logging.info(f"Loaded checkpoint from {checkpoint_path}")
            logging.info(f"Best MAP: {self.best_value:.4f} at epoch {checkpoint.get('epoch', 'unknown')}")
            return True
        else:
            logging.warning(f"No checkpoint found at {checkpoint_path}")
            return False

    @property
    def counter(self):
        return self.counter_