
# region Import libraries

# TODO MODELFLOWS: check library versions
import argparse
import numpy as np
import os
import time

from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.backends.cudnn as cudnn

# endregion

# region Import local libraries

import model_mae_image_loss as models_mae
from util import utils
from util import results_functions as results_functions
from util.datasets import build_dataset

# endregion


# region Input parameters


param = {}


# region Dataset parameters


# List of full path directories
# Each with the classes, in turn the sequences, and, finally, the .npy files
# Note: Here in test, parameter NOT USED
param['training_database_path'] = ['/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials_AI/ecos_orig/ecos_orig_Training']


# List of full path directories
# Each with the classes, in turn the sequences, and, finally, the .npy files
param['validation_database_path'] = ['/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials_AI/ecos_orig/ecos_orig_Test'] # USED


# Options: one full path, or None (for classification)
param['gt_outcome_directory'] = None # USED


# Number: only used in regression
param['gt_outcome_scale_factor'] = 100 # USED


# (NOT USED) original data path, for CIFAR datasets for instance...
# DEFAULT: '/datasets01/imagenet_full_size/061417/'
param['data_path'] = '' # 'c10' # USED


# List with the classes to be excluded (Diabetic usually gives performance problems due to lack of samples)
param['excluded_classes'] = [] # USED


# Percentage (as decimal) of the dataset to use (always using 1: 100 %)
param['subset_size'] = 1 # USED


# Whether to perform random perspective transform on images (default: False, according to the original script)
param['perturb_perspective'] = False


# endregion


# region Other parameters


# Path where to save, empty for no saving
param['output_dir'] = '/home/w460/SCRATCH/Andres/Models/MAE/model_mae_diagnosis__orig'


# Device to use for training / testing: 'cuda', 'cuda:0' 'cpu'
param['device'] = 'cuda' # USED


# Seed
param['seed'] = 0 # USED


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
param['num_channels'] = 3 # USED


# Name of model to train (Default: mae_vit_tiny)
param['model'] = 'mae_vit_tiny' # USED


# Use (per-patch) normalized pixels as targets for computing loss (Default: False)
param['norm_pix_loss'] = False # USED


# Images input size (Default: 224)
param['input_size'] = 224 # USED


# Patches size (Default: 16)
param['patch_size'] = 16 # USED


# Masking ratio (percentage of removed patches) (Default: 0.75)
param['mask_ratio'] = 0.75 # USED


# Loss weightage, i.e., to weight the reconstruction loss and the downstream loss (Default: 0.1, according to the paper)
param['lambda_weight'] = 0.1


# Drop path rate for Transformer blocks (Default: 0.1)
param['drop_path'] = 0.1 # USED


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
param['mean'] = (82.95903) # USED


# Std used for normalization
param['std'] = (53.71525) # USED


# Enable Randaugment (Default: False, seems to not be effective)
param['apply_randaugment'] = False


# Data depth to properly build the transforms: 'float, 'uint8'. Default: 'float'
param['data_depth'] = 'float' # USED


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


# region Testing parameters


# Full path to the model
param['model_path_test'] = '/home/w460/SCRATCH/Andres/Models/MAE/model_mae_diagnosis__orig__2025-11-27_00-32/checkpoint-40.pth' # USED


# Full path to export results
param['test_results_path'] = '/home/w460/SCRATCH/Andres/Results/Diagnosis/results_mae__diagnosis__orig__orig-test' # USED


# Experimental: perform parallel prediction (Default: False)
param['parallel_prediction'] = False # USED

# Export figures about per-sequence prediction (Default: False)
param['export_figures_seq_analysis'] = False


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
                        help = 'Mean value to use for normalization (custom datasets)')
    parser.add_argument('--std', type=float, default=param['std'],
                        help='Standard deviation value to use for normalization (custom datasets)')
    parser.add_argument('--apply_randaugment', type=bool, default=param['apply_randaugment'],
                        help = 'Apply randaugment or not (seems to not be working on float32 bits)')
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

    # Testing parameters

    parser.add_argument('--model_path_test', default=param['model_path_test'], type=str,
                        help='Pre-trained model to load')

    parser.add_argument('--test_results_path', default=param['test_results_path'], type=str,
                        help='Path to save results')

    parser.add_argument('--parallel_prediction', type=bool, default=param['parallel_prediction'],
                        help = 'Perform the prediction on all samples at the same time or not')

    parser.add_argument('--export_figures_seq_analysis', type=bool, default=param['export_figures_seq_analysis'],
                        help = 'Export figures with results of predictions on sequences or not')


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

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # endregion

    # region Database loading

    dataset_test = build_dataset(is_train=False, args=args)

    # endregion

    # region Model loading

    model = models_mae.__dict__[args.model](
        patch_size= args.patch_size,
        img_size= args.input_size,
        num_classes=len(dataset_test.classes),
        regression_mode = (args.gt_outcome_directory is not None),
        norm_pix_loss = args.norm_pix_loss,
        in_chans = args.num_channels
    )

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_test = torch.load(args.model_path_test, map_location='cpu', weights_only = False)

    model.load_state_dict(model_test['model'])

    model.eval()

    # endregion

    # region Prediction

    loading_times = []
    prediction_times = []

    if (args.gt_outcome_directory is not None):
        num_outputs = 1

    else:
        num_outputs = len(dataset_test.classes)

    test_predictions = np.zeros((len(dataset_test), num_outputs))

    # Experimental: only if with one call of the prediction with all the database works
    if args.parallel_prediction:

        test_samples = torch.zeros((len(dataset_test), args.num_channels, args.input_size, args.input_size), device = device)

        for i in range(len(dataset_test)):
            start_loading_time = time.time()
            test_samples[i] = dataset_test[i][0].to(device)
            loading_time = time.time() - start_loading_time
            loading_times.append(loading_time)

        average_loading_per_sample = np.sum(loading_times) / len(dataset_test)

        with torch.cuda.amp.autocast():
            start_prediction_time = time.time()
            output = model.forward_test(test_samples)
            prediction_time = time.time() - start_prediction_time
            prediction_times.append(prediction_time)

            test_predictions = output.detach().cpu().numpy()

        average_time_per_sample = np.sum(prediction_times) / len(dataset_test)

    # Actual and safe case: just perform prediction one sample by one sample
    else:

        # compute output
        with torch.cuda.amp.autocast():
            for i in range(len(dataset_test)):
                start_prediction_time = time.time()
                output = model.forward_test(torch.unsqueeze(dataset_test[i][0], 0).to(device))
                prediction_time = time.time() - start_prediction_time
                prediction_times.append(prediction_time)

                test_predictions[i, :] = output.detach().cpu().numpy()

                if i % 200 == 0:
                    print('Predicted: ' + str(i) + ' test samples')


        average_time_per_sample = np.sum(prediction_times) / len(dataset_test)

    # endregion

    # region Determine classification or regression results

    # Get the prediction confidences and the predicted classes
    Pr_confidences = np.max(test_predictions, axis=1)

    if (args.gt_outcome_directory is None):

        # Get predicted classes
        Prid_class = np.argmax(test_predictions, axis=1)

        Pred_class_names = [dataset_test.classes[label] for label in Prid_class]


    else:

        Prid_class = dataset_test.database_class_labels
        test_predictions = test_predictions * args.gt_outcome_scale_factor
        dataset_test.database_outcome_values = dataset_test.database_outcome_values * args.gt_outcome_scale_factor

    # endregion

    # region Per-image prediction statistics

    if (args.gt_outcome_directory is not None):

        pred_stats_columns = ['Sample_name', 'GT_class_name', 'GT_label', 'Outcome_time_true', 'Outcome_time_estimated',
                              'Error_estimation', 'Quadratic_error']

    else:

        pred_stats_columns = ['Sample_name', 'GT_class_name', 'Predicted_class_name', 'GT_label', 'Predicted_label']

        for trained_class_name in dataset_test.classes:
            pred_stats_columns.append('Prediction_class_' + trained_class_name)

    pred_stats_df = pd.DataFrame(columns = pred_stats_columns)

    for idx_test_sample, prid_class_sample in enumerate(Prid_class):

        if (args.gt_outcome_directory is not None):

            sample_pred_stats = [dataset_test.database_filenames[idx_test_sample],
                                 dataset_test.database_class_names[idx_test_sample],
                                 dataset_test.database_class_labels[idx_test_sample],
                                 np.ndarray.item(dataset_test.database_outcome_values[idx_test_sample]),
                                 test_predictions[idx_test_sample, 0],
                                 (test_predictions[idx_test_sample, 0] - np.ndarray.item(
                                     dataset_test.database_outcome_values[idx_test_sample])),
                                 (test_predictions[idx_test_sample, 0] - np.ndarray.item(
                                     dataset_test.database_outcome_values[idx_test_sample])) ** 2]


        else:

            sample_pred_stats = [dataset_test.database_filenames[idx_test_sample],
                                 dataset_test.database_class_names[idx_test_sample],
                                 Pred_class_names[idx_test_sample],
                                 dataset_test.database_class_labels[idx_test_sample],
                                 Prid_class[idx_test_sample]]


            for idx_trained_class_name, trained_class_name in enumerate(dataset_test.classes):
                sample_pred_stats.append(test_predictions[idx_test_sample, idx_trained_class_name])

        pred_stats_df.loc[idx_test_sample] = sample_pred_stats

    pred_stats_df.index = [os.path.basename(x_test_filename) for x_test_filename in dataset_test.database_filenames]

    pred_stats_df.to_csv(os.path.join(args.test_results_path, 'test_prediction_stats.csv'))

    # endregion


    # region Save Test times

    # Save loading test data times
    if args.parallel_prediction:

        load_times_df = pd.DataFrame(data={'total_loading_time': np.sum(loading_times),
                                           'average_loading_per_sample': average_loading_per_sample,
                                           'num_test_samples': len(dataset_test)}, index = [0])
        load_times_df.to_csv(os.path.join(args.test_results_path, 'test_data_loading_times.csv'))


    # Save prediction times
    pred_times_df = pd.DataFrame(data={'total_prediction_time': np.sum(prediction_times),
                                       'average_prediction_per_sample': average_time_per_sample,
                                       'num_test_samples': len(dataset_test)}, index = [0])

    pred_times_df.to_csv(os.path.join(args.test_results_path, 'prediction_times.csv'))


    # endregion

    # Only classification
    if (args.gt_outcome_directory is None):

        # region Per-image classification report

        # Save classification report
        test_classification_report = classification_report(dataset_test.database_class_labels, Prid_class, target_names=dataset_test.classes, output_dict = True)
        print(classification_report(dataset_test.database_class_labels, Prid_class, target_names=dataset_test.classes))
        results_functions.export_test_classification_report_csv(test_classification_report, args.test_results_path, dataset_test.classes)


        # endregion


        # region Per-image confusion matrix

        # Save confusion matrix
        cm = confusion_matrix(dataset_test.database_class_labels, Prid_class)
        print(accuracy_score(dataset_test.database_class_labels, Prid_class))
        df_cm, cm_fig = results_functions.print_confusion_matrix(cm, dataset_test.classes)
        df_cm.to_csv(os.path.join(args.test_results_path, 'confusion_matrix.csv'))
        cm_fig.savefig(os.path.join(args.test_results_path, 'confusion_matrix.png'))

        # endregion


    # region Prediction statistics for sequences

    if (args.gt_outcome_directory is not None):

        pred_stats_seq_columns = ['Seq_name', 'GT_name', 'GT_label', 'Outcome_time_true', 'Outcome_time_estimated',
                                  'Error_estimation', 'Quadratic_error']

        # Initialization
        Outcome_true_per_class = [[] for _ in range(len(dataset_test.classes))]
        Outcome_estimated_per_class = [[] for _ in range(len(dataset_test.classes))]
        Error_per_class = [[] for _ in range(len(dataset_test.classes))]
        Quadratic_per_class = [[] for _ in range(len(dataset_test.classes))]

    else:

        pred_stats_seq_columns = ['Seq_name', 'GT_name', 'Pred_name_mean', 'Pred_name_max', 'GT_label', 'Pred_label_mean',
                                  'Pred_label_max', 'Frames_true', 'Frames_mean', 'Frames_max', 'Conf_mean_true', 'Conf_mean',
                                  'Conf_max_true', 'Conf_max']

        for trained_class_name in dataset_test.classes:
            pred_stats_seq_columns.append('Perc__' + trained_class_name)

        for trained_class_name in dataset_test.classes:
            pred_stats_seq_columns.append('Frames_predicted__' + trained_class_name)

        for trained_class_name in dataset_test.classes:
            pred_stats_seq_columns.append('Conf_mean__' + trained_class_name)

        for trained_class_name in dataset_test.classes:
            pred_stats_seq_columns.append('Conf_max__' + trained_class_name)

    pred_stats_seq_df = pd.DataFrame(columns=pred_stats_seq_columns)

    # endregion

    print('Performing per-sequence classification...')

    x_test_sequence_filenames = []

    for idx_test_filename, x_test_filename in enumerate(dataset_test.database_filenames):

      all_data_sequence = dataset_test.database_filenames[idx_test_filename].split('/')[-2].split('--')[0:-1] + \
                          [dataset_test.database_filenames[idx_test_filename].split('/')[-3]]

      final_test_sequence_filename = '--'.join(all_data_sequence)

      x_test_sequence_filenames = x_test_sequence_filenames + [final_test_sequence_filename]

    test_sequence_filenames = list(set(x_test_sequence_filenames))
    test_sequence_filenames.sort()

    # region Ground-truth of each sequence

    y_seq_test_class_names = [sequence_name.split('--')[-1] for sequence_name in test_sequence_filenames]
    y_seq_test = [dataset_test.classes.index(class_name_seq) for class_name_seq in y_seq_test_class_names]

    # endregion

    if (args.gt_outcome_directory is None):

        Prid_class_mean = -np.ones((Prid_class.shape), dtype=int)
        Prid_class_max = -np.ones((Prid_class.shape), dtype=int)

        Prid_sequence_class_mean = -np.ones(len(test_sequence_filenames))
        Prid_sequence_class_max = -np.ones(len(test_sequence_filenames))

    for idx_test_sequence_filename, test_sequence_filename in enumerate(test_sequence_filenames):

        all_data_filenames_sequence_idx = np.array([i for i, x in enumerate(x_test_sequence_filenames) if x == test_sequence_filename])

        if (args.gt_outcome_directory is not None):

            test_pred_reg_seq = np.mean(test_predictions[all_data_filenames_sequence_idx])

            y_test_o_seq = dataset_test.database_outcome_values[all_data_filenames_sequence_idx]

            assert (np.all(y_test_o_seq == y_test_o_seq[0]))

            seq_pred_stats = [test_sequence_filename,
                              y_seq_test_class_names[idx_test_sequence_filename],
                              y_seq_test[idx_test_sequence_filename],
                              np.ndarray.item(y_test_o_seq[0]),
                              test_pred_reg_seq,
                              (test_pred_reg_seq - np.ndarray.item(y_test_o_seq[0])),
                              (test_pred_reg_seq - np.ndarray.item(y_test_o_seq[0])) ** 2]

            Outcome_true_per_class[y_seq_test[idx_test_sequence_filename]].append(np.ndarray.item(y_test_o_seq[0]))
            Outcome_estimated_per_class[y_seq_test[idx_test_sequence_filename]].append(test_pred_reg_seq)
            Error_per_class[y_seq_test[idx_test_sequence_filename]].append(
                (test_pred_reg_seq - np.ndarray.item(y_test_o_seq[0])))
            Quadratic_per_class[y_seq_test[idx_test_sequence_filename]].append(
                (test_pred_reg_seq - np.ndarray.item(y_test_o_seq[0])) ** 2)

        else:

            confidences_seq_mean = np.mean(test_predictions[all_data_filenames_sequence_idx, :], axis=0)
            confidences_seq_max = np.max(test_predictions[all_data_filenames_sequence_idx, :], axis=0)

            label_predicted_sequence_mean = np.argmax(confidences_seq_mean)
            label_predicted_sequence_max = np.argmax(confidences_seq_max)

            confidence_predicted_seq_mean = confidences_seq_mean[label_predicted_sequence_mean]
            confidence_predicted_seq_max = confidences_seq_max[label_predicted_sequence_max]

            y_test_in_seq = dataset_test.database_class_labels[all_data_filenames_sequence_idx]
            Prid_class_in_seq = Prid_class[all_data_filenames_sequence_idx]

            Prid_class_in_seq_stats = np.unique(Prid_class_in_seq, return_counts=True)

            Prid_class_in_seq_all_frames = np.zeros(len(dataset_test.classes), dtype=int)
            Prid_class_in_seq_all_frames[Prid_class_in_seq_stats[0]] = Prid_class_in_seq_stats[1]

            Prid_class_in_seq_perc = Prid_class_in_seq_stats[1] / len(y_test_in_seq)

            Prid_class_in_seq_all_perc = np.zeros(len(dataset_test.classes))
            Prid_class_in_seq_all_perc[Prid_class_in_seq_stats[0]] = Prid_class_in_seq_perc

            Prid_class_mean[all_data_filenames_sequence_idx] = label_predicted_sequence_mean
            Prid_class_max[all_data_filenames_sequence_idx] = label_predicted_sequence_max

            Prid_sequence_class_mean[idx_test_sequence_filename] = label_predicted_sequence_mean
            Prid_sequence_class_max[idx_test_sequence_filename] = label_predicted_sequence_max

            seq_pred_stats = [test_sequence_filename,
                              y_seq_test_class_names[idx_test_sequence_filename],
                              dataset_test.classes[label_predicted_sequence_mean],
                              dataset_test.classes[label_predicted_sequence_max],
                              y_seq_test[idx_test_sequence_filename],
                              label_predicted_sequence_mean,
                              label_predicted_sequence_max,
                              Prid_class_in_seq_all_frames[y_seq_test[idx_test_sequence_filename]],
                              Prid_class_in_seq_all_frames[label_predicted_sequence_mean],
                              Prid_class_in_seq_all_frames[label_predicted_sequence_max],
                              confidences_seq_mean[y_seq_test[idx_test_sequence_filename]],
                              confidence_predicted_seq_mean,
                              confidences_seq_max[y_seq_test[idx_test_sequence_filename]],
                              confidence_predicted_seq_max
                              ]

            for idx_trained_class_name, trained_class_name in enumerate(dataset_test.classes):
                seq_pred_stats.append(Prid_class_in_seq_all_perc[idx_trained_class_name])

            for idx_trained_class_name, trained_class_name in enumerate(dataset_test.classes):
                seq_pred_stats.append(Prid_class_in_seq_all_frames[idx_trained_class_name])

            for idx_trained_class_name, trained_class_name in enumerate(dataset_test.classes):
                seq_pred_stats.append(confidences_seq_mean[idx_trained_class_name])

            for idx_trained_class_name, trained_class_name in enumerate(dataset_test.classes):
                seq_pred_stats.append(confidences_seq_max[idx_trained_class_name])

        pred_stats_seq_df.loc[idx_test_sequence_filename] = seq_pred_stats

    pred_stats_seq_df.to_csv(os.path.join(args.test_results_path, 'test_seq_prediction_stats.csv'))
    print('Done!')

    # region Classification reports

    if (args.gt_outcome_directory is None):

        # Save classification reports

        print('Results with mean: ')
        test_classification_report_mean = classification_report(dataset_test.database_class_labels, Prid_class_mean, target_names=dataset_test.classes, output_dict = True)
        print(classification_report(dataset_test.database_class_labels, Prid_class_mean, target_names=dataset_test.classes))
        results_functions.export_test_classification_report_csv(test_classification_report_mean, args.test_results_path, dataset_test.classes, 'mean')

        print('Results with mean (per-sequence): ')
        test_classification_report_mean_seq = classification_report(y_seq_test, Prid_sequence_class_mean, target_names=dataset_test.classes, output_dict = True)
        print(classification_report(y_seq_test, Prid_sequence_class_mean, target_names=dataset_test.classes))
        results_functions.export_test_classification_report_csv(test_classification_report_mean_seq, args.test_results_path, dataset_test.classes, 'mean_seq')

        print('Results with max: ')
        test_classification_report_max = classification_report(dataset_test.database_class_labels, Prid_class_max, target_names=dataset_test.classes, output_dict = True)
        print(classification_report(dataset_test.database_class_labels, Prid_class_max, target_names=dataset_test.classes))
        results_functions.export_test_classification_report_csv(test_classification_report_max, args.test_results_path, dataset_test.classes, 'max')

        print('Results with max (per-sequence): ')
        test_classification_report_max_seq = classification_report(y_seq_test, Prid_sequence_class_max, target_names=dataset_test.classes, output_dict = True)
        print(classification_report(y_seq_test, Prid_sequence_class_max, target_names=dataset_test.classes))
        results_functions.export_test_classification_report_csv(test_classification_report_max_seq, args.test_results_path, dataset_test.classes, 'max_seq')

    else:

        pred_stats_global_columns = ['Mean_GT', 'Std_GT', 'Mean_estimated', 'Std_estimated', 'RMSE_estimated', 'Max_estimated', 'Min_estimated']
        pred_stats_global_df = pd.DataFrame(columns=pred_stats_global_columns)

        for idx_class in range(len(dataset_test.classes)):

            Mean_GT_class = np.mean(Outcome_true_per_class[idx_class])
            Std_GT_class = np.std(Outcome_true_per_class[idx_class])
            Mean_estimated_class = np.mean(Outcome_estimated_per_class[idx_class])
            Std_estimated_class = np.std(Outcome_estimated_per_class[idx_class])
            RMSE_estimated_class = np.sqrt(np.mean(Quadratic_per_class[idx_class]))
            max_estimated_class = np.max(Error_per_class[idx_class])
            min_estimated_class = np.min(Error_per_class[idx_class])

            pred_stats_global_df.loc[idx_class] = [Mean_GT_class, Std_GT_class, Mean_estimated_class, Std_estimated_class,
                                                   RMSE_estimated_class, max_estimated_class, min_estimated_class]

        Mean_GT_global = np.mean([j for i in Outcome_true_per_class for j in i])
        Std_GT_global = np.std([j for i in Outcome_true_per_class for j in i])
        Mean_estimated_global = np.mean([j for i in Outcome_estimated_per_class for j in i])
        Std_estimated_global = np.std([j for i in Outcome_estimated_per_class for j in i])
        RMSE_estimated_global = np.sqrt(np.mean([j for i in Quadratic_per_class for j in i]))
        max_estimated_global = np.max([j for i in Error_per_class for j in i])
        min_estimated_global = np.min([j for i in Error_per_class for j in i])

        pred_stats_global_df.loc[idx_class + 1] = [Mean_GT_global, Std_GT_global, Mean_estimated_global, Std_estimated_global,
                                                   RMSE_estimated_global, max_estimated_global, min_estimated_global]

        list_indexes = [class_name for class_name in dataset_test.classes]
        list_indexes.append('Global')

        pred_stats_global_df.index = list_indexes
        pred_stats_global_df.to_csv(os.path.join(args.test_results_path, 'test_reg_global_stats.csv'))

        print('Printing summary of regression results')
        print(pred_stats_global_df.to_string())


    # endregion


    # region Confusion matrixes


    if (args.gt_outcome_directory is None):

        # Save confusion matrixes
        print('Results with mean: ')
        cm_mean = confusion_matrix(dataset_test.database_class_labels, Prid_class_mean)
        print(accuracy_score(dataset_test.database_class_labels, Prid_class_mean))
        df_cm_mean, cm_fig_mean = results_functions.print_confusion_matrix(cm_mean, dataset_test.classes)
        df_cm_mean.to_csv(os.path.join(args.test_results_path, 'confusion_matrix_mean.csv'))
        cm_fig_mean.savefig(os.path.join(args.test_results_path, 'confusion_matrix_mean.png'))

        print('Results with mean (per-sequence): ')
        cm_mean_seq = confusion_matrix(y_seq_test, Prid_sequence_class_mean)
        print(accuracy_score(y_seq_test, Prid_sequence_class_mean))
        df_cm_mean_seq, cm_fig_mean_seq = results_functions.print_confusion_matrix(cm_mean_seq, dataset_test.classes)
        df_cm_mean_seq.to_csv(os.path.join(args.test_results_path, 'confusion_matrix_mean_seq.csv'))
        cm_fig_mean_seq.savefig(os.path.join(args.test_results_path, 'confusion_matrix_mean_seq.png'))

        print('Results with max: ')
        cm_max = confusion_matrix(dataset_test.database_class_labels, Prid_class_max)
        print(accuracy_score(dataset_test.database_class_labels, Prid_class_max))
        df_cm_max, cm_fig_max = results_functions.print_confusion_matrix(cm_max, dataset_test.classes)
        df_cm_max.to_csv(os.path.join(args.test_results_path, 'confusion_matrix_max.csv'))
        cm_fig_max.savefig(os.path.join(args.test_results_path, 'confusion_matrix_max.png'))

        print('Results with max (per-sequence): ')
        cm_max_seq = confusion_matrix(y_seq_test, Prid_sequence_class_max)
        print(accuracy_score(y_seq_test, Prid_sequence_class_max))
        df_cm_max_seq, cm_fig_max_seq = results_functions.print_confusion_matrix(cm_max_seq, dataset_test.classes)
        df_cm_max_seq.to_csv(os.path.join(args.test_results_path, 'confusion_matrix_max_seq.csv'))
        cm_fig_max_seq.savefig(os.path.join(args.test_results_path, 'confusion_matrix_max_seq.png'))

    # endregion


    # region Plots about per-sequence prediction
    if args.export_figures_seq_analysis:
        if (args.gt_outcome_directory is None):

            test_seq_results_path = os.path.join(args.test_results_path, 'seq_analysis')

            if not os.path.exists(test_seq_results_path):
                os.makedirs(test_seq_results_path)


            for idx_test_sequence_filename, test_sequence_filename in enumerate(test_sequence_filenames):

                all_data_filenames_sequence_idx = np.array([i for i, x in enumerate(x_test_sequence_filenames) if x == test_sequence_filename])


                confidences_seq_mean = np.mean(test_predictions[all_data_filenames_sequence_idx, :], axis=0)
                confidences_seq_max = np.max(test_predictions[all_data_filenames_sequence_idx, :], axis=0)

                label_predicted_sequence_mean = np.argmax(confidences_seq_mean)
                label_predicted_sequence_max = np.argmax(confidences_seq_max)

                confidence_predicted_seq_mean = confidences_seq_mean[label_predicted_sequence_mean]
                confidence_predicted_seq_max = confidences_seq_max[label_predicted_sequence_max]


                plt.figure(figsize = (12.8, 9.6))
                plt.title("Per-image predicted label in sequence " + test_sequence_filename, fontsize = 16)
                plt.xlabel("N_Frame")
                plt.ylabel("Predicted label")

                # Per-image predicted label in the sequence (blue [0.2, 0.1, 0.9]) CORRECTED: black
                plt.plot(range(len(all_data_filenames_sequence_idx)), Prid_class[all_data_filenames_sequence_idx],
                         color=[0, 0, 0], label="Prid", linewidth=2)

                # Line representing the predicted label in the sequence (mean; red) CORRECTED: BLACK
                plt.plot(range(0, len(all_data_filenames_sequence_idx), 6), label_predicted_sequence_mean*np.ones(len(range(0, len(all_data_filenames_sequence_idx), 6)), dtype = int),
                         color=[0, 0, 0],
                         marker = "x",
                         label="Prid_seq_mean: " + dataset_test.classes[label_predicted_sequence_mean],
                         linewidth = 1,
                         linestyle = 'None')

                # Line representing the predicted label in the sequence (max; cyan [0.5, 0.8, 0.9]) CORRECTED: BLACK
                plt.plot(range(2, len(all_data_filenames_sequence_idx), 6), label_predicted_sequence_max*np.ones(len(range(2, len(all_data_filenames_sequence_idx), 6)), dtype = int),
                         color=[0, 0, 0],
                         marker= "^",
                         label="Prid_seq_max: " + dataset_test.classes[label_predicted_sequence_max],
                         linewidth = 1,
                         linestyle = 'None')

                # Line representing the GT of the sequence (green [0.1, 0.7, 0.2]) CORRECTED: red
                plt.plot(range(4, len(all_data_filenames_sequence_idx), 6), y_seq_test[idx_test_sequence_filename]*np.ones(len(range(4, len(all_data_filenames_sequence_idx), 6)), dtype = int),
                         color=[1, 0, 0],
                         marker = "o",
                         label="GT_seq: " + y_seq_test_class_names[idx_test_sequence_filename],
                         linewidth = 3,
                         linestyle = 'None')

                plt.legend()

                plt.savefig(os.path.join(test_seq_results_path, 'Prid_per-image_' + test_sequence_filename + '.png'))

                plt.close()

                for idx_trained_class_name, trained_class_name in enumerate(dataset_test.classes):


                    plt.figure(figsize = (12.8, 9.6))
                    plt.title("Per-image confidence in sequence " + test_sequence_filename + " in class " + trained_class_name, fontsize = 16)
                    plt.xlabel("N_Frame")
                    plt.ylabel("Confidence")

                    # Per-image confidence in class j (blue [0.2, 0.1, 0.9]) (CORRECTION: BLACK)
                    plt.plot(range(len(all_data_filenames_sequence_idx)), test_predictions[all_data_filenames_sequence_idx, idx_trained_class_name], color = [0, 0, 0],
                             label = "Conf_" + trained_class_name, linewidth = 2, linestyle = 'solid')

                    # Line representing the mean confidence of class j (black)
                    plt.plot(range(len(all_data_filenames_sequence_idx)), confidences_seq_mean[idx_trained_class_name]*np.ones(len(all_data_filenames_sequence_idx)), color = [0, 0, 0],
                             label = "Conf_mean_" + trained_class_name, linewidth = 1, linestyle = 'dashdot')

                    # Line representing the max confidence of class j (magenta [0.75, 0, 0.75]) (CORRECTED: BLACK)
                    plt.plot(range(len(all_data_filenames_sequence_idx)), confidences_seq_max[idx_trained_class_name]*np.ones(len(all_data_filenames_sequence_idx)), color = [0, 0, 0],
                             label = "Conf_max_" + trained_class_name, linewidth = 1, linestyle = 'dashed')

                    # Line representing the confidence of predicted class by mean (red)
                    plt.plot(range(len(all_data_filenames_sequence_idx)), confidence_predicted_seq_mean*np.ones(len(all_data_filenames_sequence_idx), dtype = int), color = [1, 0, 0],
                             label = "Conf_mean_predicted: " + dataset_test.classes[label_predicted_sequence_mean], linewidth = 2, linestyle = 'dashdot')

                    # Line representing the confidence of predicted class by max (cyan [0.5, 0.8, 0.9]) (CORRECTED: RED)
                    plt.plot(range(len(all_data_filenames_sequence_idx)), confidence_predicted_seq_max*np.ones(len(all_data_filenames_sequence_idx), dtype = int), color = [1, 0, 0],
                             label = "Conf_max_predicted: " + dataset_test.classes[label_predicted_sequence_max], linewidth = 2, linestyle = 'dashed')

                    plt.legend()

                    plt.savefig(os.path.join(test_seq_results_path, 'Conf_per-image_' + test_sequence_filename + '_' + trained_class_name + '.png'))

                    plt.close()


                    if (not(trained_class_name == y_seq_test_class_names[idx_test_sequence_filename])):

                        plt.figure(figsize=(12.8, 9.6))
                        plt.title("Per-image confidence in sequence " + test_sequence_filename + ": GT_" + y_seq_test_class_names[idx_test_sequence_filename] + " vs " + trained_class_name,
                                  fontsize=16)
                        plt.xlabel("N_Frame")
                        plt.ylabel("Confidence")

                        # Per-image confidence in class j (blue [0.2, 0.1, 0.9]) (CORRECTION: BLACK)
                        plt.plot(range(len(all_data_filenames_sequence_idx)),
                                 test_predictions[all_data_filenames_sequence_idx, idx_trained_class_name], color=[0, 0, 0],
                                 label="Conf_" + trained_class_name, linewidth=2, linestyle='solid')

                        # Per-image confidence in GT class (green [0.1, 0.7, 0.2]) (CORRECTION: RED)
                        plt.plot(range(len(all_data_filenames_sequence_idx)),
                                 test_predictions[all_data_filenames_sequence_idx, y_seq_test[idx_test_sequence_filename]],
                                 color=[1, 0, 0],
                                 label="Conf_" + y_seq_test_class_names[idx_test_sequence_filename], linewidth=3,
                                 linestyle='solid',
                                 marker="o")

                        # Line representing the mean confidence of class j (black)
                        plt.plot(range(len(all_data_filenames_sequence_idx)),
                                 confidences_seq_mean[idx_trained_class_name] * np.ones(len(all_data_filenames_sequence_idx)),
                                 color=[0.0, 0.0, 0.0], label="Conf_mean_" + trained_class_name, linewidth=1,
                                 linestyle='dashdot')

                        # Line representing the max confidence of class j (magenta [0.75, 0, 0.75]) (CORRECTED: BLACK)
                        plt.plot(range(len(all_data_filenames_sequence_idx)),
                                 confidences_seq_max[idx_trained_class_name] * np.ones(len(all_data_filenames_sequence_idx)),
                                 color=[0.0, 0.0, 0.0], label="Conf_max_" + trained_class_name, linewidth=1, linestyle='dashed')

                        # Line representing the confidence of predicted class by mean (red)
                        plt.plot(range(len(all_data_filenames_sequence_idx)),
                                 confidence_predicted_seq_mean * np.ones(len(all_data_filenames_sequence_idx), dtype=int),
                                 color=[1, 0, 0],
                                 label="Conf_mean_predicted: " + dataset_test.classes[label_predicted_sequence_mean],
                                 linewidth=2,
                                 linestyle='dashdot')

                        # Line representing the confidence of predicted class by max (cyan [0.5, 0.8, 0.9]) (CORRECTED: RED)
                        plt.plot(range(len(all_data_filenames_sequence_idx)),
                                 confidence_predicted_seq_max * np.ones(len(all_data_filenames_sequence_idx), dtype=int),
                                 color=[1, 0, 0],
                                 label="Conf_max_predicted: " + dataset_test.classes[label_predicted_sequence_max], linewidth=2,
                                 linestyle='dashed')

                        plt.legend()

                        plt.savefig(os.path.join(test_seq_results_path,
                                                 'Conf_per-image_GT_vs_' + trained_class_name + "_" + test_sequence_filename + '.png'))

                        plt.close()


if __name__ == '__main__':

    # Getting the arguments
    args = get_args_parser(param)

    # Parsing the arguments
    args = args.parse_args()

    # Creating the path to export the results
    if args.test_results_path:

        timestamp_results_path = time.strftime("%Y-%m-%d_%H-%M")
        results_complete_path = os.path.join(args.test_results_path + '-' + timestamp_results_path)

        args.test_results_path = results_complete_path

        Path(args.test_results_path).mkdir(parents=True, exist_ok=True)

        utils.export_input_arguments(param, args.test_results_path)

    # Calling the main function
    main(args)

