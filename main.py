import os
import random
import logging
from argparse import ArgumentParser

import numpy as np
import torch

from utils import set_logger, Monitor, WarmUpAndCosineDecayScheduler
from evaluation import Evaluator, get_db_codes_and_targets

from data import CIFAR10, Flickr25K, NUSWIDE, Cartoon18K
from network import HyperPQ
from loss import HyperbolicCQCLoss, ProtoLoss
from engine import train, test
from geoopt.optim import  RiemannianSGD
import math_util

def parse_args():
    parser = ArgumentParser(description="Run HyperPQ")
    # dataset configurations
    parser.add_argument('--dataset', 
                        type=str, default='CIFAR10',
                        help="Choose a dataset from 'CIFAR10', 'Flickr25K', 'NUSWIDE' or 'MSLS'.")
    parser.add_argument('--protocal', 
                        type=str, default='I',
                        help="Select evaluation protocal on CIFAR10. Options: 'I' or 'II'.")
    parser.add_argument('--download_cifar10', 
                        dest='download_cifar10', action='store_true',
                        help='Download CIFAR-10 via torchvision or not.')
    parser.set_defaults(download_cifar10=False)
    parser.add_argument('--num_workers', 
                        type=int, default=4,
                        help='Number of threads for data fetching.')

    # optimizer configurations
    parser.add_argument('--batch_size', 
                        type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epoch_num', 
                        type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--lr', 
                        type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--lr_scaling',
                        type=float, default=1e-3,
                        help='Learning rate scaling for CNN layers.')
    parser.add_argument('--momentum', 
                        type=float, default=0.9,
                        help='Learning rate.')
    parser.add_argument('--hp_beta', 
                        type=float, default=5e-6,
                        help='Weight decay factor.')
    parser.add_argument('--disable_scheduler', 
                        dest='use_scheduler', action='store_false',
                        help='Disabling the learning rate scheduler.')
    parser.set_defaults(use_scheduler=True)
    parser.add_argument('--warmup_epoch_num',
                        type=int, default=1,
                        help='Number of warmup epochs for lr scheduler.')
    parser.add_argument('--start_lr',
                        type=float, default=1e-5,
                        help='Learning rate at the start of warmup.')
    parser.add_argument('--final_lr',
                        type=float, default=1e-5,
                        help='Final learning rate of cosine decaying schedule.')

    # quantization configurations
    parser.add_argument('--feat_dim', 
                        type=int, default=64,
                        help='Dimension of image features.')
    parser.add_argument('--M', 
                        type=int, default=4,
                        help='Number of codebooks.')
    parser.add_argument('--K', 
                        type=int, default=256,
                        help='Number of sub-codewords per sub-codebook.')
    parser.add_argument('--softmax_temp', 
                        type=float, default=10,
                        help='Temperature parameter for soft codeword assignment.')
    parser.add_argument('--trainable_layer_num', 
                        type=int, default=0,
                        help='The number of trainable layers for VGG-16 backbone. Options: 0, 1 or 2.')
    parser.add_argument("--quant_method", 
                        type=str, default="softmax",
                        help="st or softmax")
    
    parser.add_argument("--init_neg_curvs",
                        type=str, help="initial value of negative curvatures")
    parser.add_argument("--clip_r", default=1.0,
                        type=float, help="clipped lr ")
    
    parser.add_argument('--T', 
                        type=float, default=0.1,
                        help='Temperature parameter for nce loss.')


    parser.add_argument("--tau_cqc", type=float, default=1.0)
   

    
    # evaluation configurations
    parser.add_argument('--symmetric_distance', 
                        dest='is_asym_dist', action='store_false',
                        help='Declare this option to use symmetric quantization distance, otherwise to use asymmetric quantization distance.')
    parser.set_defaults(is_asym_dist=True)
    parser.add_argument('--topK', 
                        type=int, default=None,
                        help='TopK for metric evaluation')
    parser.add_argument('--eval_interval', 
                        type=int, default=2,
                        help='Interval for evaluation (in epoch).')
    parser.add_argument('--monitor_counter', 
                        type=int, default=8,
                        help='The maximum patience for metric monitor.')

    # other configurations
    parser.add_argument('--device', 
                        type=str, default='cuda',
                        help="Device: 'cpu', 'cuda:X'")
    parser.add_argument('--seed', 
                        type=int, default=2021,
                        help='Random seed.')
    parser.add_argument('--notes', 
                        type=str, default="",
                        help="Notes and remarks for current experiment.")
    parser.add_argument('--disable_writer',
                        dest='use_writer', action='store_false',
                        help='Disabling tensorboard summary writer.')
    parser.set_defaults(use_writer=True)
    parser.add_argument("--only_train",
                        action="store_true",
                        help="if it's true, then no any evaluation.")
    parser.add_argument("--checkpoint_dir",
                        type=str, default="./checkpoints/",
                        help="Directory to save model checkpoints.")

    parser.add_argument("--add_supp_layer",
                        action="store_true",
                        help="add supplmental layers")
    parser.add_argument("--assymetric_loss",
                        action="store_true",
                        help="assymetric_loss")
    parser.add_argument("--loss_method",
                        type=str, default="cqc",
                        help="type of loss method")
    parser.add_argument("--full_hyperpq",
                        action="store_true")
 
    parser.add_argument("--prot_loss_weight",
                        type=float, default=1.0)
    parser.add_argument("--neighbor_loss_weight",
                        type=float, default=0.1)
    
    parser.add_argument("--concat_cwd", action="store_true")

    parser.add_argument("--clus_mode", type=str,
                        default="hier_residual", help="hier_residual, hier_normal, hcsc")
    parser.add_argument("--num_clus_list", type=str,
                       default = "100,50,20", help="hierarchy of cluster")
    parser.add_argument("--warmup_epoch", type=int,
                        default=3)
    parser.add_argument("--clus_interval", type=int,
                        default=1)

    parser.add_argument("--use_alpha", action="store_true")

    parser.add_argument("--save_quant_error", type=str, default="None")

    parser.add_argument('--resume', 
                        type=str, default=None,
                        help='Path to checkpoint file to resume training from.')


    
    return parser.parse_args()


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    writer = set_logger(config)

    config.num_clus_list = config.num_clus_list.split(",")

    logging.info("config: " + str(config))
    logging.info("prepare %s datatset" % config.dataset)
    if config.dataset == 'CIFAR10':
        datahub = CIFAR10(root='./datasets/CIFAR-10/',
                          protocal=config.protocal,
                          download=True,
                          batch_size=config.batch_size,
                          num_workers=config.num_workers)
    elif config.dataset == 'Flickr25K':
        datahub = Flickr25K(root="/kaggle/input/flickr25kdata/mirflickr25k/",
                            img_root="/kaggle/input/flickr25kdata/mirflickr25k/mirflickr/",
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

    elif config.dataset == "NUSWIDE":
        datahub = NUSWIDE(root="/kaggle/input/nuswidedata/NUS-WIDE/",
                          img_root="/kaggle/input/nuswidedata/NUS-WIDE/NUS-WIDE/Flickr/",
                          batch_size=config.batch_size,
                          num_workers=config.num_workers,
                          train_file="train_10500")
    elif config.dataset == "Cartoon18K":
        datahub = Cartoon18K(root="/kaggle/input/cartoon7k5/dataset/",
                       img_root="/kaggle/input/cartoon7k5/dataset/keyframes/",
                       batch_size=config.batch_size,
                       num_workers=config.num_workers)
    else:
        raise ValueError("Unknown dataset '%s'." % config.dataset)


    logging.info("setup model")
    model = HyperPQ(feat_dim=config.feat_dim,
                    M=config.M, K=config.K, softmax_temp=config.softmax_temp,
                    quant_method=config.quant_method,
                    trainable_layer_num=config.trainable_layer_num,
                    init_neg_curvs=eval(config.init_neg_curvs),
                    clip_r = config.clip_r,
                    add_supp_layer=config.add_supp_layer,
                    full_hyperpq=config.full_hyperpq,
                    use_alpha=config.use_alpha,
                    writer=writer)
    model = model.to(config.device)
    
 

    logging.info("define loss function")

    loss_fn1 = HyperbolicCQCLoss(tau_cqc=config.tau_cqc, writer=writer, assymetric_mode=config.assymetric_loss)
    loss_fn2 = ProtoLoss(temp=config.T)
   

    params = [
        {'params': model.vgg.parameters(), 'lr': config.lr * config.lr_scaling},
        {'params': model.projction_layer.parameters(), 'lr': config.lr},
        {'params': model.hyper_pq_head.parameters(), 'lr': config.lr}
    ]


    if config.add_supp_layer:
        params.append({'params': model.supp_layer.parameters(), 'lr': config.lr})
    

    optimizer = RiemannianSGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.hp_beta,stabilize=1)

    logging.info("prepare monitor and evaluator")
    monitor = Monitor(max_patience=config.monitor_counter, save_dir=config.checkpoint_dir)

    if config.only_train:
        evaluator = None 
    else:
        evaluator = Evaluator(feat_dim=config.feat_dim,
                            M=config.M, K=config.K,
                            is_asym_dist=config.is_asym_dist,
                            device=config.device)
        
    lr_scheduler = WarmUpAndCosineDecayScheduler(optimizer=optimizer, 
                                                 start_lr=config.start_lr, 
                                                 base_lr=config.lr, 
                                                 final_lr=config.final_lr,
                                                 epoch_num=config.epoch_num, 
                                                 batch_num_per_epoch=len(datahub.train_loader), 
                                                 warmup_epoch_num=config.warmup_epoch_num) if config.use_scheduler else None

    # Resume from checkpoint if specified
    start_epoch = 0
    if config.resume:
        logging.info(f"Resuming training from checkpoint: {config.resume}")
        if os.path.exists(config.resume):
            checkpoint = torch.load(config.resume, map_location='cpu', weights_only=False)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Model state loaded")
            
            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("Optimizer state loaded")
            
            # Load scheduler state
            if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info("Scheduler state loaded")
            
            # Get starting epoch
            start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"Resuming training from epoch {start_epoch}")
            
            # Update monitor if available
            if 'best_map' in checkpoint:
                monitor.best_value = checkpoint['best_map']
                logging.info(f"Best MAP so far: {monitor.best_value:.4f}")
        else:
            logging.error(f"Checkpoint file not found: {config.resume}")
            exit(1)

    logging.info("begin to train model")
    
    if config.save_quant_error == "None":
        save_quant_error = None
    else:
        save_quant_error = config.save_quant_error
    train(datahub=datahub,
          model=model,
          loss_fn1=loss_fn1,
          loss_fn2=loss_fn2,
          optimizer=optimizer,
          lr_scheduler=lr_scheduler,
          config=config,
          evaluator=evaluator,
          monitor=monitor,
          writer=writer,
          save_quant_error=save_quant_error,
          prot_loss_weight=config.prot_loss_weight,
          neighbor_loss_weight=config.neighbor_loss_weight,
          start_epoch=start_epoch)