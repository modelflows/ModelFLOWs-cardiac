# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# region Import libraries

import os
import PIL

from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset

import timm.data
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import pandas as pd

from glob import glob
from sklearn import preprocessing

# endregion

def build_dataset(is_train, args):

    # First build sequence of Transforms
    transform = build_transform(is_train, args)

    # Set flags
    flag_single_path = True

    if len(args.training_database_path) > 0:
        flag_single_path = False

    flag_data_depth = 'float'

    if 'data_depth' in args:
        flag_data_depth = args.data_depth

    print(flag_data_depth)

    # Flag to use weights for regression
    flag_get_two_outputs = False
    if args.gt_outcome_directory is not None:
        if ('weights_classes' in args):
            if len(args.weights_classes) > 0:
                flag_get_two_outputs = True
                print('Activated class weights!')

    # Case of initial standard databases (CIFAR)
    if flag_single_path:

        if args.data_path == 'c10':
            dataset = datasets.CIFAR10(
                        root='./data', train=is_train, download=True, transform=transform
            )
        elif args.data_path == 'c100':
            dataset = datasets.CIFAR100(
                        root='./data', train=is_train, download=True, transform=transform
            )

        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)

    # Case of custom databases (e.g., Echocardiography Imagery)
    else:

        if 'gt_outcome_scale_factor' in args:
            gt_outcome_scale_factor = args.gt_outcome_scale_factor

        else:
            gt_outcome_scale_factor = 1

        excluded_classes_list = []

        if 'excluded_classes' in args:
            excluded_classes_list = args.excluded_classes

        print('List of excluded classes: ')
        print(excluded_classes_list)

        # Load databases of Echocardiography Images
        if is_train:
            dataset = EchocardiographyImageDataset(args.training_database_path, args.num_channels, args.gt_outcome_directory, gt_outcome_scale_factor, data_depth = flag_data_depth, transform = transform, excluded_classes= excluded_classes_list, get_two_outputs=flag_get_two_outputs)

        else:
            dataset = EchocardiographyImageDataset(args.validation_database_path, args.num_channels, args.gt_outcome_directory, gt_outcome_scale_factor, data_depth = flag_data_depth, transform=transform, excluded_classes= excluded_classes_list)

    # Select a subset of the database if indicated in the input parameters
    if is_train and args.subset_size != 1:
        np.random.seed(127)

        dsize = len(dataset)
        idxs = np.random.choice(dsize, int(dsize*args.subset_size), replace=False)

        dataset = torch.utils.data.Subset(dataset, idxs)
        print('Subset dataset size: ', len(dataset))

    print(f'{dataset} ({len(dataset)})')

    return dataset


def build_transform(is_train, args):

    flag_single_path = True

    if len(args.training_database_path) > 0:
        flag_single_path = False

    flag_data_depth = 'float'

    if 'data_depth' in args:

        flag_data_depth = args.data_depth

    print(flag_data_depth)

    if flag_single_path:

        if args.data_path == 'c10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        else:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD

    else:

        if args.num_channels == 3:
            mean = (args.mean, ) * 3
            std = (args.std, ) * 3


    # train transform
    if is_train:

        if args.num_channels == 3:

            if args.apply_randaugment:

                if flag_data_depth == 'uint8':
                    # this should always dispatch to transforms_imagenet_train
                    transform = create_transform(
                        input_size=args.input_size,
                        is_training=True,
                        color_jitter=args.color_jitter,
                        auto_augment=args.aa,
                        interpolation='bicubic',
                        re_prob=args.reprob,
                        re_mode=args.remode,
                        re_count=args.recount,
                        mean=mean,
                        std=std,
                    )
                    print('Transforms uint8 3 channel with randaugment')
                    return transform

            else:

                if flag_data_depth == 'uint8':
                    t1 = timm.data.transforms.RandomResizedCropAndInterpolation(size=(args.input_size, args.input_size), interpolation='bicubic')
                    t2 = transforms.RandomHorizontalFlip(p=0.5)

                    t4 = transforms.ToTensor()

                    t5 = transforms.Normalize(mean=mean, std=std)

                    # NOTE: assuming args.recount = 1
                    # t6 = timm.data.random_erasing.RandomErasing(probability=args.reprob, mode=args.remode) # , device=args.device)

                    t6 = transforms.RandomErasing(p=args.reprob)

                    transforms_list = [t1, t2, t4, t5, t6]

                    transform = transforms.Compose(transforms_list)

                    print('Transforms uint8 3 channel NO randaugment')

                    return transform

                if flag_data_depth == 'float':

                    transforms_list = [

                        transforms.RandomResizedCrop(size=args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std),
                        transforms.RandomErasing(p=args.reprob)

                    ]

                    print('Transforms float 3 channel NO randaugment')

                    transform = transforms.Compose(transforms_list)
                    return transform

    print('Transforms for validation!')

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    if 'perturb_perspective' in vars(args) and args.perturb_perspective:
        print('[Log] Perturbing perspective of images in dataset')
        t.append(transforms.RandomPerspective(distortion_scale=0.5, p=1.0))
    else:
        print('[Log] Not perturbing perspective of images in dataset')

    # NOTE: PAY ATTENTION TO THE ORIGINAL DATA TYPE AND APPLY THE MEAN AND STANDARD VALUES ACCORDINGLY:
    # IF SOURCE DATA ARE UINT8 -> ToTensor scales to [0, 1] -> mean and std adapted to be between 0 and 1
    # IF source data np.float32-> ToTensor does not change values -> mean and std in the original values
    if flag_data_depth == 'uint8':
        t.append(transforms.ToTensor())

    if args.num_channels == 3:

        t.append(transforms.Normalize(mean, std))

    return transforms.Compose(t)


# region Custom functions
class EchocardiographyImageDataset(Dataset):
    def __init__(self, img_dirs, num_channels = 1, GT_outcome_directory = None, gt_outcome_scale_factor = 1, data_depth = 'float', transform=None, target_transform=None, label_encoder=None, excluded_classes = [], get_two_outputs = False):


        all_database_images_filenames = []
        all_database_images_class_names = []
        class_names = []
        total_num_samples = 0

        all_image_outcome_values = []
        output_label_encoder = None

        if GT_outcome_directory is not None:

            GT_outcome_list = glob(os.path.join(GT_outcome_directory, '*'))

            print(GT_outcome_list)

            for GT_outcome_filename in GT_outcome_list:
                dataframe1 = pd.read_csv(GT_outcome_filename)

            print(dataframe1)
            print('Ground-truth data for regression loaded!')

        for img_dir_idx, img_dir in enumerate(img_dirs):

            database_as_directories = False
            database_images_directory = glob(os.path.join(img_dir, '*'))

            # Case if we have folders with the class names
            if os.path.isdir(database_images_directory[0]):
                database_as_directories = True

            if database_as_directories:

                print('Situation of samples with known classes')

                # Just in case
                database_images_directory.sort()

                for idx_class, database_class_name_fullpath in enumerate(database_images_directory):

                    if not (os.path.basename(database_class_name_fullpath) in excluded_classes):

                        all_sequence_filenames_class = glob(os.path.join(database_class_name_fullpath, '*'))
                        all_sequence_filenames_class.sort()

                        if img_dir_idx == 0:
                            class_names = class_names + [os.path.basename(database_class_name_fullpath)]

                        for idx_sequence_class, sequence_filename_class in enumerate(all_sequence_filenames_class):
                            samples_sequence_class_filenames = glob(os.path.join(sequence_filename_class, '*'))
                            samples_sequence_class_filenames.sort()

                            # NOTE: CASE WITHOUT LISTS OF SPLITTING INDEXES

                            all_database_images_filenames = all_database_images_filenames + samples_sequence_class_filenames

                            all_database_images_class_names = all_database_images_class_names + [
                                os.path.basename(database_class_name_fullpath)] * len(samples_sequence_class_filenames)

                            if GT_outcome_directory is not None:
                                sequence_id = int(os.path.basename(sequence_filename_class).split('_')[1])

                                # if sequence_id in dataframe1['ID'].values:
                                # sequence_outcome = dataframe1['Outcome'][dataframe1['ID'] == sequence_id]

                                # sequence_outcome = sequence_outcome.values[0]

                                # sequence_outcome_values = [sequence_outcome] * len(samples_sequence_class_filenames)

                                # all_image_outcome_values = all_image_outcome_values + sequence_outcome_values

                                sequence_outcome = dataframe1['Outcome'][dataframe1['ID'] == sequence_id]

                                sequence_outcome = sequence_outcome.values[0]

                                sequence_outcome_values = [sequence_outcome] * len(samples_sequence_class_filenames)

                                all_image_outcome_values = all_image_outcome_values + sequence_outcome_values

        total_num_samples = total_num_samples + len(all_database_images_filenames)

        print('Database with ' + str(total_num_samples) + ' samples and ' + str(len(database_images_directory)) + ' classes')

        if database_as_directories:

            if label_encoder is None:

                le = preprocessing.LabelEncoder()
                # Assume we have the same classes as in training, but if we do not have test samples in a certain class, the corresponding folder would be empty
                le.fit(all_database_images_class_names)

                all_database_images_class_labels = le.transform(all_database_images_class_names)

                output_label_encoder = le


            else:

                all_database_images_class_labels = label_encoder.transform(all_database_images_class_names)

                output_label_encoder = label_encoder




        self.database_filenames = all_database_images_filenames
        self.database_class_names = all_database_images_class_names
        self.database_class_labels = all_database_images_class_labels
        self.database_outcome_values = np.expand_dims(np.array(all_image_outcome_values, dtype = np.float32), axis = 1)/gt_outcome_scale_factor
        self.classes = class_names
        self.img_dir = img_dirs
        self.num_channels = num_channels
        self.data_depth = data_depth
        self.transform = transform
        self.target_transform = target_transform
        self.label_encoder = label_encoder
        self.get_two_outputs = get_two_outputs

        # return all_images, all_database_images_class_labels, np.array(all_image_outcome_values), all_database_images_filenames, all_database_images_class_names, class_names, output_label_encoder, min_max_scaler

        # self.img_labels = pd.read_csv(annotations_file)
        # self.img_dir = img_dir
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):

        return len(self.database_class_labels)

    def __getitem__(self, idx):

        sample_path = self.database_filenames[idx]
        label = self.database_class_labels[idx]

        if '.npy' in sample_path:
            sample = np.load(sample_path)

            if self.num_channels == 3:

                if self.data_depth == 'uint8':
                    # Assuming grayscale images...
                    sample = np.expand_dims(sample, axis=2)
                    sample = np.tile(sample, (1, 1, 3))

                    # sample = np.expand_dims(sample, axis = 0)
                    # sample = torch.from_numpy(sample)
                    sample = PIL.Image.fromarray(sample)

                if self.data_depth == 'float':
                    sample = np.expand_dims(sample, axis=0)
                    sample = np.tile(sample, (3, 1, 1))
                    sample = torch.from_numpy(sample.astype(np.float32))

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        if len(self.database_outcome_values) > 0:
            value_regression = self.database_outcome_values[idx]

            if self.get_two_outputs:

                return sample, [label, value_regression]
            else:

                return sample, value_regression

        else:

            return sample, label

# endregion