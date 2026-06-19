
# region Import libraries

# TODO MODELFLOWS: check library versions
import argparse
import datetime
import json
import numpy as np
import os
import time
import wandb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import timm.optim.optim_factory as optim_factory

# endregion


# region Import local libraries

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util import utils

import model_mae_image_loss as models_mae
from engine_two_branch import train_one_epoch, evaluate

# endregion


# region Input parameters


param = {}


# region Dataset parameters


# List of full path directories
# Each with the classes, in turn the sequences, and, finally, the .npy files
param['training_database_path'] = ['/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials_AI/ecos_orig/ecos_orig_Training']


# List of full path directories
# Each with the classes, in turn the sequences, and, finally, the .npy files
param['validation_database_path'] = ['/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials_AI/ecos_orig/ecos_orig_Validation']


# Options: one full path, or None (for classification)
param['gt_outcome_directory'] = '/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials_AI/My_databases_ecos_Outcome_GT'


# Number: only used in regression
param['gt_outcome_scale_factor'] = 100


# One number for each class, or [] default (not very useful parameter).
param['weights_classes'] = []


# (NOT USED) original data path, for CIFAR datasets for instance...
# DEFAULT: '/datasets01/imagenet_full_size/061417/'
param['data_path'] = '' # 'c10'


# List with the classes to be excluded (Diabetic usually gives performance problems due to lack of samples)
param['excluded_classes'] = []


# Percentage (as decimal) of the dataset to use (always using 1: 100 %)
param['subset_size'] = 1


# Whether to perform random perspective transform on images (default: False, according to the original script)
param['perturb_perspective'] = False


# endregion


# region Other parameters


# Path where to save, empty for no saving
param['output_dir'] = '/home/w460/SCRATCH/Andres/Models/MAE/model_mae_prognosis__orig'


# Device to use for training / testing: 'cuda', 'cuda:0' 'cpu'
param['device'] = 'cuda'


# Seed
param['seed'] = 0


# (NOT USED) Resume from checkpoint (Default: '')
param['resume'] = ''


# Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus (Default: 64)
param['batch_size'] = 64


# Accumulate gradient iterations (for increasing the effective batch size under memory constraints) (Default: 1)
param['accum_iter'] = 1


# Number of epochs (Default: 100)
param['epochs'] = 100


# Start epoch (Default: 0)
param['start_epoch'] = 0


# Perform evaluation only (Default: False)
param['eval'] = False


# Enabling distributed evaluation (recommended during training for faster monitor) (Default: False)
param['dist_eval'] = False


# Number of workers (Default: 10)
param['num_workers'] = 10


# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU (Default: True, according to original script)
param['pin_mem'] = True


# NOT USED
param['no_pin_mem'] = False


# endregion


# region Model parameters


# Number in input channels (Keep to 3: I do not manage to make it work with 1 channel)
param['num_channels'] = 3


# Name of model to train (Default: mae_vit_tiny)
param['model'] = 'mae_vit_tiny'


# Use (per-patch) normalized pixels as targets for computing loss (Default: False)
param['norm_pix_loss'] = False


# Images input size (Default: 224)
param['input_size'] = 224


# Patches size (Default: 16)
param['patch_size'] = 16


# Masking ratio (percentage of removed patches) (Default: 0.75)
param['mask_ratio'] = 0.75


# Loss weightage, i.e., to weight the reconstruction loss and the downstream loss (Default: 0.1, according to the paper)
param['lambda_weight'] = 0.1


# Drop path rate for Transformer blocks (Default: 0.1)
param['drop_path'] = 0.1


# endregion


# region Optimizer parameters


# Clip gradient norm (default: None, no clipping)
param['clip_grad'] = None


# Weight decay (Default: 0.05)
param['weight_decay'] = 0.05


# Learning rate (absolute lr) (Default: None)
param['lr'] = None


# Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (Default: 1e-3)
param['blr'] = 1e-3


# Layer-wise lr decay from ELECTRA/BEiT (Default: 0.75)
param['layer_decay'] = 0.75


# Lower lr bound for cyclic schedulers that hit 0 (Default: 1e-6)
param['min_lr'] = 1e-6


# Epochs to warmup LR (Default: 5)
param['warmup_epochs'] = 5


# endregion


# region Augmentation parameters


# Mean used for normalization
param['mean'] = (82.95903)


# Std used for normalization
param['std'] = (53.71525)


# Enable Randaugment (Default: False, seems to not be effective)
param['apply_randaugment'] = False


# Data depth to properly build the transforms: 'float, 'uint8'. Default: 'float'
param['data_depth'] = 'float'


# Color jitter factor (enabled only when not using Auto/RandAug) (Default: None)
param['color_jitter'] = None


# Use AutoAugment policy ("v0" or "original") (Default: 'rand-m9-mstd0.5-inc1')
param['aa'] = 'rand-m9-mstd0.5-inc1'


# Label smoothing, only for classification (Default: 0.1)
param['smoothing'] = 0.1


# endregion


# region Random Erase (Augmentation) params


# Random erase probability (Default: 0.25)
param['reprob'] = 0.25


# Random erase mode (Default: 'pixel')
param['remode'] = 'pixel'


# Random erase count (Default: 1)
param['recount'] = 1


# Do not random erase first (clean) augmentation split (Default: False)
param['resplit'] = False


# endregion


# region Mixup parameters


# Mixup alpha, mixup enabled if > 0 (Default in MAE: 0) (Default in LIFE: 0.8)
param['mixup'] = 0


# Cutmix alpha, cutmix enabled if > 0 (Default in MAE: 0) (Default in LIFE: 1.0)
param['cutmix'] = 0


# Cutmix min/max ratio, overrides alpha and enables cutmix if set (Default in MAE: None)
param['cutmix_minmax'] = None


# Probability of performing mixup or cutmix when either/both is enabled (Default in MAE: 1.0)
param['mixup_prob'] = 1.0


# Probability of switching to cutmix when both mixup and cutmix enabled (Default in MAE: 0.5)
param['mixup_switch_prob'] = 0.5


# How to apply mixup/cutmix params (Per "batch", "pair", or "elem") (Default in MAE: 'batch')
param['mixup_mode'] = 'batch'


# endregion


# region Finetuning parameters


# Finetune from checkpoint (Default: '')
param['finetune'] = ''


# SET_DEFAULTS: True
param['global_pool'] = True


# Use class token instead of global pool for classification
param['cls_token'] = None


# endregion


# region Distributed training parameters


# Number of distributed processes (Default: 1)
param['world_size'] = 1


# (Default: -1)
param['local_rank'] = -1


# Default: False (according to original script)
param['dist_on_itp'] = False


# Url used to set up distributed training (Default: 'env://')
param['dist_url'] = 'env://'


# endregion


# endregion


# region Parse arguments


def get_args_parser(param):
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=param['batch_size'], type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=param['epochs'], type=int)
    parser.add_argument('--accum_iter', default=param['accum_iter'], type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--weights_classes', default=param['weights_classes'], type = list,
                        help='Class weights applied for the loss function')

    # Model parameters
    parser.add_argument('--model', default=param['model'], type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=param['norm_pix_loss'])
    
    parser.add_argument('--input_size', default=param['input_size'], type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=param['patch_size'], type=int,
                        help='patches size')
    parser.add_argument('--mask_ratio', default=param['mask_ratio'], type=float,
                        help='Masking ratio (percentage of removed patches)')

    parser.add_argument('--num_channels', default=param['num_channels'], type=int,
                        help='num channels')

    parser.add_argument('--lambda_weight', default=param['lambda_weight'], type=float,
                        help='Loss weightage')

    parser.add_argument('--drop_path', type=float, default=param['drop_path'], metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=param['clip_grad'], metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=param['weight_decay'],
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=param['lr'], metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=param['blr'], metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=param['layer_decay'],
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=param['min_lr'], metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=param['warmup_epochs'], metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--mean', type=float, default=param['mean'],
                        help='Mean value to use for normalization (custom datasets)')
    parser.add_argument('--std', type=float, default=param['std'],
                        help='Standard deviation value to use for normalization (custom datasets)')
    parser.add_argument('--apply_randaugment', type=bool, default=param['apply_randaugment'],
                        help='Apply randaugment or not (seems to not be working the default on float32)')
    parser.add_argument('--data_depth', type = str, default=param['data_depth'],
                        help='Depth of data (as data type) to properly build the transforms')
    parser.add_argument('--color_jitter', type=float, default=param['color_jitter'], metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default=param['aa'], metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=param['smoothing'],
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=param['reprob'], metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default=param['remode'],
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=param['recount'],
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=param['resplit'],
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=param['mixup'],
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=param['cutmix'],
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=param['cutmix_minmax'],
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=param['mixup_prob'],
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=param['mixup_switch_prob'],
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default=param['mixup_mode'],
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default=param['finetune'],
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=param['global_pool'])
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters

    # My parameters
    if 'training_database_path' in param:

        parser.add_argument('--training_database_path', default=param['training_database_path'], type=list,
                            help='training database path')

    if 'validation_database_path' in param:

        parser.add_argument('--validation_database_path', default=param['validation_database_path'], type=list,
                            help='validation database path')

    if 'gt_outcome_directory' in param:

        parser.add_argument('--gt_outcome_directory', default=param['gt_outcome_directory'], type=str,
                            help='ground-truth outcome directory')

    if 'gt_outcome_scale_factor' in param:

        parser.add_argument('--gt_outcome_scale_factor', default=param['gt_outcome_scale_factor'], type = float,
                            help = 'Scale factor to apply on the outcome values (for regression)')

    parser.add_argument('--data_path', default=param['data_path'], type=str,
                        help='dataset path')


    if 'excluded_classes' in param:

        parser.add_argument('--excluded_classes', default=param['excluded_classes'], type=list,
                            help='List of excluded classes')

    else:

        parser.add_argument('--excluded_classes', default=[], type=list,
                            help='List of excluded classes')


    parser.add_argument('--subset_size', default=param['subset_size'], type=float,
                        help='percentage (as decimal) of imagenet subset to use')
    parser.add_argument('--perturb_perspective', action='store_true',
                        help='whether to perform random perspective transform on images')

    parser.add_argument('--output_dir', default=param['output_dir'],
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default=param['device'],
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=param['seed'], type=int)
    parser.add_argument('--resume', default=param['resume'],
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=param['start_epoch'], type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=param['dist_eval'],
                        help='Enabling distributed evaluation (recommended during training for faster monitor)')
    parser.add_argument('--num_workers', default=param['num_workers'], type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.set_defaults(pin_mem=param['pin_mem'])

    # distributed training parameters
    parser.add_argument('--world_size', default=param['world_size'], type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=param['local_rank'], type=int)


    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default=param['dist_url'],
                        help='url used to set up distributed training')

    return parser


# endregion


def main(args):

    # region Initialization
#    wandb.init(project='MAE-Project', entity='tanmayj2020')
#    config = wandb.config
#    config.dataset = args.dataset
#    config.model = args.model
#    config.epoch = args.epochs
#    config.classification_loss_ratio= args.lambda_weight

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # endregion

    # region Database loading

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    # print(dataset_train[5])
    # print(dataset_val[5])

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.output_dir is not None and not args.eval:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Note: By default, mixup is not activated...
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")

        # Case regression
        if (args.gt_outcome_directory is not None):

            print('Mixup adapted to regression!')
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=0.0, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=0.0, num_classes=2)
            # NOTE: completely dummy Mixup function, the adapted one is employed inside

        # Case classification
        else:

            print('Mixup adapted to classification!')

            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=len(dataset_train.classes))

    # endregion

    # region Model

    model = models_mae.__dict__[args.model](
        patch_size= args.patch_size,
        img_size= args.input_size,
        num_classes=len(dataset_train.classes),
        regression_mode = (args.gt_outcome_directory is not None),
        norm_pix_loss = args.norm_pix_loss,
        in_chans = args.num_channels
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # endregion

    # region Param Optimizer and loss function

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # Loss function
    # Case regression: MSE or variants
    if args.gt_outcome_directory is not None:
        if not('weights_classes' in args):
            criterion = torch.nn.MSELoss()
        else:
            if len(args.weights_classes) == 0:
                criterion = torch.nn.MSELoss()

            else:
                criterion = models_mae.WeightedMSELoss()

    # Case classification: Cross Entropy Loss or variants
    else:

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # endregion

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)

        if (args.gt_outcome_directory is not None):
            print(f"MSE of the network on the {len(dataset_val)} test images: {test_stats['mse']:.3f}")
        else:
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    # region Model fitting

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    min_mse = 0.0
    save_model_best = False
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args, regression_mode = (args.gt_outcome_directory is not None)
        )

        test_stats = evaluate(data_loader_val, model_without_ddp, device, (args.gt_outcome_directory is not None))

        if (args.gt_outcome_directory is not None):

            print(f"MSE of the network on the {len(dataset_val)} test images: {test_stats['mse']:.3f}")

            if epoch == args.start_epoch:
                save_model_best = True
                min_mse = test_stats['mse']

            else:
                if test_stats['mse'] < min_mse:
                    save_model_best = True
                min_mse = min(min_mse, test_stats['mse'])

        else:

            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

            if test_stats['acc1'] > max_accuracy:
                save_model_best = True

            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        if args.output_dir:
            if save_model_best:
                print('Saving best model...')
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                save_model_best = False

        # print('-------------------------------------------')
        # utils.check_ram_usage()
        # print('-------------------------------------------')
        # utils.check_gpu_usage()
        # print('-------------------------------------------')

        if log_writer is not None:

            if (args.gt_outcome_directory is None):

                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)

            log_writer.add_scalar('perf/test_loss', test_stats['testing_loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
#        wandb.log({"epoch" : epoch , **train_stats , **test_stats})

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # endregion


if __name__ == '__main__':

    # Getting the arguments
    args = get_args_parser(param)

    # Parsing the arguments
    args = args.parse_args()

    # Creating the path to export the model training
    if args.output_dir:

        timestamp_model_path = time.strftime("%Y-%m-%d_%H-%M")
        model_complete_path = os.path.join(args.output_dir + '__' + timestamp_model_path)

        args.output_dir = model_complete_path

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        utils.export_input_arguments(param, args.output_dir)

    # Calling the main function
    main(args)
