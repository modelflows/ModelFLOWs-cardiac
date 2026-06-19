#!/usr/bin/env Python 3.10.4
# -*- coding: utf-8 -*-

################### HODMD_IT ###################

__author__ = "Egoitz Maiora Otaegi"

"""
Use Multi-resolution HODMD extension of HODMD for the analysis of complex
signals, noisy data, complex flows...

Algorithm presented in:
Le Clainche, S., Vega, J.M. & Soria, J., Higher order dynamic mode
decomposition of noisy experimental data: The flow structure of a
zero-net-mass-flux jet,Exp. Therm. Fluid Sci., 2018, 88, pp. 336-353

HODMD by Le Clainche, S. & Vega, J.M., Higher order dynamic mode
decomposition, SIAM J. Appl. Dyn. Sys., 16(2), pp. 882-925

INPUT:
    -d: parameter of DMD-d (higher order Koopman assumption)
    -V: snapshot matrix
    -deltaT: time between snapshots
    -varepsilon1: first tolerance (SVD)
    -varepsilon: second tolerance (DMD-d modes)
OUTPUT:
    -Vreconst: reconstruction of the snapshot matrix V
    -deltas: growth rate of DMD modes
    -omegas: frequency of DMD modes(angular frequency)
    -amplitude: amplitude of DMD modes
    -modes: DMD modes

"""

# region Import libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pydicom as dicom
import h5py             # To read data from .mat files
import scipy.io         # To read data from .mat files
import os               # To create/open folders
import sys              # To export prints into a document
import time             # To calculate execution time
import cv2
from glob import glob
import pandas as pd

# endregion


# region Import local libraries

import DMDd
import hosvd
import utils

# endregion


print('Multi-resolution HODMD' + '\n')

param = {}


# region Input parameters

print('INPUT:' + '\n')

# region Param Path to data

param['database_path'] = input(
    'Write a path in which data are stored or press enter to introduce the tensor cylinder: ')

if not param['database_path']:
    # l: components (i.e.: streamwise, normal and spanwise velocity)
    # i: X (or Y)
    # j: Y (or X)
    # s: Z
    # k: Time

    # Detect current working directory:
    path0 = os.getcwd()
    param['database_path'] = f'{path0}/Tensor_cylinder_Re100.mat'

# endregion

# region Param Path to save results
param['results_path'] = input(
    'Write a path in which results will be saved or press enter to save in the project directory: ')

if not param['results_path']:
    param['results_path'] = ''

# endregion

# region Param Colormap to represent the images
param['colormap'] = input('Select colormap or press enter to continue with gray: ')

if not param['colormap']:
    param['colormap'] = 'gray'

# endregion

# region Param SVD library

# Valid values: numpy, torch, tensorflow, torch_32f
param['svd_library'] = input('Introduce library to use for SVD or press enter to continue with torch \nOptions: \nnumpy \ntorch \ntensorflow \nhybrid \nhybrid_lower_precision: ')

if not param['svd_library']:
    param['svd_library'] = 'torch'

print('Selected SVD library: ' + param['svd_library'])

# endregion

# region Param Driver for SVD (only used in torch CUDA)

# Valid values: None, gesvd, gesvdj, and gesvda.
param['svd_driver'] = input('Introduce driver to use for the case of SVD CUDA or press enter to continue with None: ')

if not param['svd_driver']:
    param['svd_driver'] = None

print('Selected driver for SVD CUDA: ' + str(param['svd_driver']))

# endregion

# region Param Number of snapshots nsnap
param['SNAP'] = input('Introduce number of snapshots or press enter to continue with a number of snapshots corresponding to the size of the temporal dimension of the input data: ')

if param['SNAP']:
    param['SNAP'] = int(param['SNAP'])

    introduced_SNAP = True

else:

    introduced_SNAP = False

# endregion

# region Param Tolerances
param['varepsilon1'] = input('Introduce first tolerance (SVD) or press enter to continue with 0.0005: ')
if not param['varepsilon1']:
    param['varepsilon1'] = 0.0005  # SVD
else:
    param['varepsilon1'] = float(param['varepsilon1'])

param['varepsilon2'] = input('Introduce second tolerance (DMD-d Modes) or press enter to continue with 0.0005: ')
if not param['varepsilon2']:
    param['varepsilon2'] = 0.0005  # DMD
else:
    param['varepsilon2'] = float(param['varepsilon2'])

# endregion
    
# region Param d
# TODO: TRY WITH D = SNAP / 3 OR SNAP / 5
param['d'] = input('Introduce parameter of DMD-d (d) or press enter to continue with 1 (options: a number; /3 or /5): ')
if not param['d']:
    param['d'] = 1

else:
    if not(param['d'] == '/3' or param['d'] == '/5'):
        param['d'] = int(param['d'])

# endregion

# region Param Time gradient
param['deltaT'] = input('Introduce time gradient (deltaT) or press enter to continue with 0.004: ')
if not param['deltaT']:
    param['deltaT'] = 0.004
else:
    param['deltaT'] = float(param['deltaT'])

# endregion

# region Param Complex data analysis
param['compyn'] = input('\n' + 'Are complex data being analyzed? Yes or No (y/n). Press enter to continue with no: ')
if not param['compyn']:
    param['compyn'] = 0
elif (param['compyn']=='n' or param['compyn']=='N' or param['compyn']=='no' or param['compyn']=='No' or param['compyn']=='NO'):
    param['compyn'] = 0
elif (param['compyn']=='y' or param['compyn']=='Y' or param['compyn']=='yes' or param['compyn']=='Yes' or param['compyn']=='YES'):
    param['compyn'] = 1

# endregion


# /media/andres/TOSHIBA EXT/Andres/Trabajo/Results/Project_v0.1-main_ModelFLOWs_app/good_ones/My_database_SVD_example_4/svd_reconst_images
# /media/andres/TOSHIBA EXT/Andres/Trabajo/Results/Project_HODMD_IT_Python/good ones/My_database_SVD_example_4




# DBC:
# DB LAX 3
# LAX16
# LAX17
# LAX19_01
# LAX19_02
# LAX21_01
# LAX22_01
# LAX23_02
# LAX28_02
# LAX31_01
# LAX 7
# LAX 8_125 - 32 W
# LAX 10
# LAX 11
# /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/results/LAX_torch_lower_precision_extensive_orig_DBC_problematic/svd_reconst_data
# /home/andres/andres/Results/Project_HODMD_IT_Python/results/LAX_extensive_DBC_problematic_1

# HL
# LAX6
# LAX7
# LAX8
# LAX10
# LAX11
# LAX12_01
# LAX16
# LAX18_01
# LAX21
# LAX25_01
# LAX25_02
# LAX26_01
# LAX27_01

# OB
# LAX4_01
# LAX4_02
# LAX5_01
# LAX7_01
# LAX9_02
# LAX12_02
# LAX14_02
# LAX27_01
# LAX28_02


# region Used parameters for LAX DICOM video
# /home/andres/andres/Databases/Data_DICOM/LAX DICOM_fixed.dcm
# /home/andres/andres/Results/Project_HODMD_IT_Python/LAX DICOM_ILAS 2023
# SNAP = 130
# varepsilon1 = 0.0005
# varepsilon2 = 0.0005
# d = 50
# deltaT = 0.004
# compyn = 0

# endregion

# region Used parameters for SAX DICOM video
# /home/andres/andres/Databases/Data_DICOM/SAX DICOM.dcm
# /home/andres/andres/Results/Project_HODMD_IT_Python/SAX DICOM_ILAS 2023
# SNAP = 130
# varepsilon1 = 0.0005
# varepsilon2 = 0.0005
# d = 50
# deltaT = 0.004
# compyn = 0

# endregion



# endregion


# region Output
temp = sys.stdout

print('\n' + 'OUTPUT:' + '\n')
print('Wait until the run is completed.')
print('This could take some time...' + '\n')

# region Save results in folder: param['results_path']/DMD_solution

print('Results will be saved in the following folder: ' + os.path.join(param['results_path'], 'DMD_solution'))

# Create new folders
if not os.path.exists(os.path.join(param['results_path'], 'original_images')):
    os.makedirs(os.path.join(param['results_path'], 'original_images'))

if not os.path.exists(os.path.join(param['results_path'], 'original_images_data')):
    os.makedirs(os.path.join(param['results_path'], 'original_images_data'))

if not os.path.exists(os.path.join(param['results_path'], 'DMD_solution')):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution'))

if not os.path.exists(os.path.join(param['results_path'], 'List_input_filenames')):
    os.makedirs(os.path.join(param['results_path'], 'List_input_filenames'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data'))

if not(os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_images'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_images'))

if not(os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_data'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_data'))

if not(os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_images'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_images'))

if not(os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_data'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_data'))

if not(os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_images'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_images'))

if not(os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_data'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_data'))




if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_tols'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_tols'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'Amplitudes'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'Amplitudes'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'GrowthRates'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'GrowthRates'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'RRMSE'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'RRMSE'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_times'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_times'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_times'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_times'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_SVD_times'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_SVD_times'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_times'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_times'))

if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_values'))):
    os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_values'))






# Create file to print data:
file = open(os.path.join(param['results_path'], 'DMD_solution', 'DMD_history.txt'), 'w')
file.truncate(0)
file.close()

# From here, everything will be written in the code DMD_history.txt, and the terminal will remain empty.
# sys.stdout = open(f'{path0}/DMD_solution/DMD_history.txt', 'w')

# endregion





# region Load data

print('Loading data...')

database_extension = os.path.splitext(param['database_path'])[1]
list_all_filenames = []

# region Case of a .mat file (e.g. Cylinder)
if database_extension == '.mat':

    # Tensor_ = h5py.File(f'{path0}\Tensor_cylinder_Re100.mat')
    # Tensor = Tensor_.get('Tensor')

    Tensor_ = scipy.io.loadmat(param['database_path'])  # Imports the data as dictionary
    Tensors = Tensor_['Tensor']

    ## Position of temporal variable: The analysis is always carried out over the temporal variable
    # print('\n' + 'Set the position of temporal variable (TimePos).\nThe code is prepared to set the temporal evolution in the last component of the tensor,\nso this variable also determines the dimension of the tensor. Press enter to continue with 5.' + '\n')
    # TimePos = input('TimePos = ')
    # if not TimePos:
    #    TimePos = 5
    # else:
    #    TimePos = int(TimePos)
    TimePos = Tensors.ndim

    if not param['SNAP']:
        param['SNAP'] = Tensors.shape[TimePos - 1]

    # Python imports the tensor with the indexes switched:
    # if TimePos == 5:
    #    Tensor = np.transpose(Tensor, (4,3,2,1,0))
    # elif TimePos == 4:
    #    Tensor = np.transpose(Tensor, (3,2,1,0))

    # Correction of Tensor dimension with number of snapshots:
    Tensor0 = Tensors
    shapeTens = list(np.shape(Tensors))
    shapeTens[-1] = param['SNAP']
    Tensors = np.zeros(shapeTens)

    Tensors[..., :] = Tensor0[..., 0:param['SNAP']]

# endregion

# region Case of a DICOM file

if database_extension == '.dcm':

    Tensor0 = dicom.dcmread(param['database_path']).pixel_array

    TimePos = Tensor0.ndim
    if not param['SNAP']:
        param['SNAP'] = Tensor0.shape[0] # TODO: previously, it was Tensor0.shape[-1], but it should be Tensor0.shape[0], because the temporal dimension is here

    ymin = 390 - 1  # left
    ymax = 780  # right
    xmin = 140 - 1  # up
    xmax = 650  # down


    for n_frame in range(Tensor0.shape[0]):

        # Visualize original frames of the DICOM image (normally not activated)
        if 0:
            plt.figure()
            ax = plt.gca()
            im = ax.imshow(Tensor0[n_frame, :, :, :])
            plt.title('Frame: ' + str(n_frame))
            plt.show()
            time.sleep(1e-2)
            plt.close()

        # Visualize cropped frames of the DICOM image (normally not activated)
        if 0:
            plt.figure()
            ax = plt.gca()
            im = ax.imshow(Tensor0[n_frame, xmin:xmax, ymin:ymax, :])
            plt.title('Frame: ' + str(n_frame))
            plt.show()
            time.sleep(1e-2)
            plt.close()

        gray_image = cv2.cvtColor(Tensor0[n_frame, xmin:xmax, ymin:ymax, :], cv2.COLOR_BGR2GRAY)

        # Note: we directly do this step to make the algorithm a little faster...
        if n_frame == 0:
            Tensors = np.zeros((1, gray_image.shape[0], gray_image.shape[1], param['SNAP']))
            print('Original image dimensions')
            print(Tensors.shape)

        Tensors[0,:,:,n_frame] = gray_image


        np.save(os.path.join(param['results_path'], 'original_images_data',
                             'original_image_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.npy'), gray_image)

        # Export the original images from the sequence
        plt.figure()
        ax = plt.gca()
        im = ax.imshow(gray_image, extent=[0, gray_image.shape[1], gray_image.shape[0], 0], cmap=plt.get_cmap(param['colormap']),
                       aspect="auto")

        plt.title('Original image frame ' + str(n_frame))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.savefig(os.path.join(param['results_path'], 'original_images',
                                 'original_image_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png'))

        # plt.show()
        plt.close()


        if n_frame == param['SNAP'] - 1:
            break



# None of previous formats -> Folder
else:

    list_files = glob(os.path.join(param['database_path'], '*'))
    list_files.sort()


    # V2
    Tensors = [[]]*len(list_files)

    list_data_info = [[]]*3

    list_data_info[0] = list_files

    list_data_info[1] = [[]] * len(list_files)

    list_data_info[2] = [[]] * len(list_files)

    total_number_samples = 0


    for idx_file, file_name in enumerate(list_files):

        # Count total number of files
        if os.path.isdir(file_name):
            list_files_2 = glob(os.path.join(file_name, '*'))
            list_files_2.sort()



            #### NEW
            if os.path.isdir(list_files_2[0]):
                print('Sequences!')

                list_files_2.sort()

                list_data_info[1][idx_file] = list_files_2

                list_data_info[2][idx_file] = [[]] * len(list_files_2)



                for idx_file_2, file_name_2 in enumerate(list_files_2):

                    # FILES IN SEQUENCE
                    list_files_3 = glob(os.path.join(file_name_2, '*'))

                    list_files_3.sort()

                    total_number_samples = total_number_samples + len(list_files_3)

                    list_data_info[2][idx_file][idx_file_2] = list_files_3




    for idx_file, file_name in enumerate(list_files):

        if os.path.isdir(file_name):

            list_files_2 = glob(os.path.join(file_name, '*'))

            # SEQUENCES
            list_files_2.sort()

            Tensors[idx_file] = [[]] * len(list_files_2)

            for idx_file_2, file_name_2 in enumerate(list_files_2):

                list_files_3 = glob(os.path.join(file_name_2, '*'))

                # SAMPLES
                list_files_3.sort()

                for idx_file_3, file_name_3 in enumerate(list_files_3):

                    file_extension_3 = os.path.splitext(file_name_3)[1]

                    if 'npy' in file_extension_3:

                        Tensor_individual = np.load(file_name_3)


                    if (idx_file_3 == 0):
                        Tensor_sequence = np.zeros((1, Tensor_individual.shape[0],
                                          Tensor_individual.shape[1],
                                          len(list_files_3)))

                        TimePos = Tensor_sequence.ndim

                    Tensor_sequence[0, :, :, idx_file_3] = Tensor_individual


                Tensors[idx_file][idx_file_2] = Tensor_sequence

    # TODO: for case with folders, export original images as well

# endregion

print('...data loaded!')

# endregion

# region File heading
print('MULTI-RESOLUTION HODMD EXTENSION OF HODMD FOR THE ANALYSIS OF COMPLEX SIGNALS, NOISY DATA, COMPLEX FLOWS...')
print('\n' + 'HODMD by Le Clainche, S. & Vega, J.M., Higher order dynamic mode decomposition, SIAM J. Appl. Dyn. Sys., 16(2), pp. 882-925')

print('\n' + 'Database path: ' + param['database_path'])
print('Path to save results: ' + param['results_path'])
print('Selected SVD library: ' + param['svd_library'])
print('Selected Driver for SVD CUDA: ' + str(param['svd_driver']))
# print(f"Number of snapshots set at {param['SNAP']}")
print(f"d Parameter set at {param['d']}")
print(f"Tolerances set at {param['varepsilon1']} for SVD and {param['varepsilon2']} for DMD")
print(f"Time gradient set at deltaT = {param['deltaT']}")
print(f"Position of temporal variable set at TimePos = {TimePos}")
if param['compyn']==0:
    print('Analyzed data is not complex' + '\n')
elif param['compyn']==1:
    print('Complex data is being analyzed' + '\n')

utils.export_input_arguments(param, os.path.join(param['results_path'], 'DMD_solution'))

# endregion

# region Algorithm

if isinstance(Tensors, list):

    list_class_names = list_data_info[0]
    list_sequences_names = list_data_info[1]
    list_input_filenames = list_data_info[2]




for idx_class, class_name in enumerate(list_class_names):

    class_name_extracted = os.path.basename(class_name)
    list_sequences_class = list_sequences_names[idx_class]



    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_data', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_data', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_data', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_data', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted))

    if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_data', class_name_extracted))):
        os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_data', class_name_extracted))



    for idx_sequence, sequence_name in enumerate(list_sequences_class):


        total_hosvd_time = 0
        total_svd_time = 0
        total_hodmd_it_time = 0

        total_time = time.time()

        Tensor = Tensors[idx_class][idx_sequence]

        sequence_name_extracted = os.path.basename(sequence_name).split('--')[0]

        print(sequence_name_extracted)


        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images', class_name_extracted, sequence_name_extracted + '--hodmd_reconst--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images', class_name_extracted, sequence_name_extracted + '--hodmd_reconst--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data', class_name_extracted, sequence_name_extracted + '--hodmd_reconst--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data', class_name_extracted, sequence_name_extracted + '--hodmd_reconst--data'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted, sequence_name_extracted + '--hodmd_real--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted, sequence_name_extracted + '--hodmd_real--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_data', class_name_extracted, sequence_name_extracted + '--hodmd_real--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_data', class_name_extracted, sequence_name_extracted + '--hodmd_real--data'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', class_name_extracted, sequence_name_extracted + '--hodmd_imag--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', class_name_extracted, sequence_name_extracted + '--hodmd_imag--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_data', class_name_extracted, sequence_name_extracted + '--hodmd_imag--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_data', class_name_extracted, sequence_name_extracted + '--hodmd_imag--data'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted, sequence_name_extracted + '--hodmd_abs--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted, sequence_name_extracted + '--hodmd_abs--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_data', class_name_extracted, sequence_name_extracted + '--hodmd_abs--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_data', class_name_extracted, sequence_name_extracted + '--hodmd_abs--data'))



        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted, sequence_name_extracted + '--hodmd_real_negative--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted, sequence_name_extracted + '--hodmd_real_negative--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'real_data', class_name_extracted, sequence_name_extracted + '--hodmd_real_negative--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'real_data', class_name_extracted, sequence_name_extracted + '--hodmd_real_negative--data'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', class_name_extracted, sequence_name_extracted + '--hodmd_imag_negative--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', class_name_extracted, sequence_name_extracted + '--hodmd_imag_negative--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'imag_data', class_name_extracted, sequence_name_extracted + '--hodmd_imag_negative--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'imag_data', class_name_extracted, sequence_name_extracted + '--hodmd_imag_negative--data'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted, sequence_name_extracted + '--hodmd_abs_negative--images'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted, sequence_name_extracted + '--hodmd_abs_negative--images'))

        if not (os.path.exists(os.path.join(param['results_path'], 'DMD_solution', 'abs_data', class_name_extracted, sequence_name_extracted + '--hodmd_abs_negative--data'))):
            os.makedirs(os.path.join(param['results_path'], 'DMD_solution', 'abs_data', class_name_extracted, sequence_name_extracted + '--hodmd_abs_negative--data'))


        # NEW VERSION: REQUIRED TO SET THE SNAPSHOT PARAM
        # if not param['SNAP']:
        if not introduced_SNAP:
            param['SNAP'] = len(list_input_filenames[idx_class][idx_sequence])


        # NEW VERSION: REQUIRED TO CHANGE THE 'd' param
        # if len(list_input_filenames[idx_class][idx_sequence]) - 8 <= param['d']:
            # print("Warning: d value should be changed!")
            # d_value_used = round(4*len(list_input_filenames[idx_class][idx_sequence])/5) # round(len(list_input_filenames[idx_class][idx_sequence])/2)
        # else:
            # d_value_used = param['d']
        if isinstance(param['d'], str):

            if (param['d'] == '/3'):

                d_value_used = int(np.floor(param['SNAP']/3))

            if (param['d'] == '/5'):

                d_value_used = int(np.floor(param['SNAP']/5))

        else:

            d_value_used = param['d']

        print(d_value_used)


        Time = np.linspace(1,param['SNAP'],num=param['SNAP'])*param['deltaT']

        # region ITERATIVE

        nn0 = np.shape(Tensor)
        nn = np.array(nn0)
        nn[1:np.size(nn)] = 0

        print('Iterations:' + '\n')

        for zz in range(0,1000):

            print(f'Iteration number: {zz+1}')

            if zz != 0:
                del S,U,Frequency,GrowthRate,hatT,hatMode,sv,Tensor,nnin
                Tensor = TensorReconst
            # Reconstruction
            ## Perform HOSVD decomposition to calculate the reduced temporal matrix: hatT
            nnin = nn
            nnin = tuple(nnin)
            # IMPORTANT NOTE: there is no implementation of HOSVD inside the library torch

            [hatT,U,S,sv,nn1, total_hosvd_time_per_it] = hosvd.HOSVD(Tensor, param['varepsilon1'],nn,nn0,TimePos, param['svd_library'], param['svd_driver'])
            # hatT: Reduced temporal matrix
            # U: Temporal SVD modes
            # S: Tensor core
            # sv: singular values
            # nn: number of singular values retained
            # n: initial number of singular values

            ## Perform HODMD to the reduced temporal matrix hatT:

            [hatMode,Amplitude,Eigval,GrowthRate,Frequency, total_svd_time_per_it, total_hodmd_it_time_per_it] = DMDd.hodmd_IT(hatT, d_value_used, Time, param['varepsilon1'], param['varepsilon2'], param['svd_library'], param['svd_driver'])
            # GrowthRate: Growth rate of the DMD modes
            # Frequency: Frequency of the DMD modes
            # hatMode: reduced DMD mode
            # Amplitude: Amplitudes wighting the DMD modes
            # modes retained are > varepsilon2

            ## Reconstruct the original Tensor using the DMD expansion:
            TensorReconst = DMDd.reconst_IT(hatMode,Time,U,S,sv,nn1,TimePos,GrowthRate,Frequency)

            # Check if data is complex:
            if param['compyn']==0:
                TensorReconst = np.real(TensorReconst)

            ## Print outcome:

            RRMSE_it = np.linalg.norm(np.reshape(Tensor-TensorReconst,newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))
            print(f'Relative mean square error made in the calculations: {RRMSE_it}')

            # region Export results per iteration
            np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_times', 'HOSVD_time_it_' + str(zz+1) + '_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                       np.array([total_hosvd_time_per_it]))

            np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_SVD_times', 'HODMD_IT_ONLY_SVD_time_it_' + str(zz+1) + '_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                       np.array([total_svd_time_per_it]))

            np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_times', 'HODMD_IT_time_it_' + str(zz+1) + '_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                       np.array([total_hodmd_it_time_per_it]))

            np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'RRMSE', 'RRMSE_it_' + str(zz+1) + '_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                       np.array([RRMSE_it]))
            # endregion


            total_hosvd_time = total_hosvd_time + total_hosvd_time_per_it
            total_svd_time = total_svd_time + total_svd_time_per_it
            total_hodmd_it_time = total_hodmd_it_time + total_hodmd_it_time_per_it


            GRFreqAmp = np.zeros((np.size(GrowthRate),3))
            for ind in range (0,np.size(GrowthRate)):
                GRFreqAmp[ind,0] = GrowthRate[ind]
                GRFreqAmp[ind,1] = Frequency[ind]
                GRFreqAmp[ind,2] = Amplitude[ind]

            print('GrowthRate, Frequency and Amplitude:')
            print(GRFreqAmp)

            ## Break the loop when the number of singular values is the same in two consecutive iterations:

            num = 0
            for ii in range(1,np.size(nn1)):
                if nnin[ii]==nn1[ii]:
                    num = num+1

            if num==np.size(nn1)-1:
                break
            nn = nn1
            print('\n')

        print('\n' + f'Final number of iterations: {zz+1}')

        # endregion

        ## Stop saving the output on the file:
        sys.stdout = temp

        # region Calculate and save DMD modes

        print('Calculating DMD modes...' + '\n')
        N = np.shape(hatT)[0]
        DMDmode = DMDd.modes_IT(N,hatMode,Amplitude,U,S,nn1,TimePos)

        # endregion

        total_time = time.time() - total_time

        print('Total elapsed time: ' + str(total_time))

        # region Save the reconstruction of the tensor and measures: times, the Growth rates, frequencies and amplitudes

        if len(list_input_filenames[idx_class][idx_sequence]) > 0:

            df = pd.DataFrame(list_input_filenames[idx_class][idx_sequence], columns=['Input_filename'])
            df.to_csv(os.path.join(param['results_path'], 'List_input_filenames', 'List_input_filenames_' + os.path.basename(list_sequences_names[idx_class][idx_sequence]) + '.csv'))

            np.save(os.path.join(param['results_path'], 'List_input_filenames', "list_input_filenames_" + os.path.basename(list_sequences_names[idx_class][idx_sequence]) + ".npy"), list_input_filenames[idx_class][idx_sequence])

        # Reconstruction:
        # file1 = open(f'{path0}\DMD_solution\TensorReconst.txt', 'w')
        # file1.truncate(0)
        # file1.close()

        # np.save(os.path.join(param['results_path'], 'DMD_solution', 'TensorReconst_' + param['svd_library'] + '.npy'), TensorReconst)



        for n_frame in range(TensorReconst.shape[-1]):

            # Export the reconstructed images from the sequence



            reconst_data_filename =  os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data', class_name_extracted,
                                                  sequence_name_extracted + '--hodmd_reconst--data',
                                                  sequence_name_extracted + '--hodmd_reconst--data_' +
                                                  str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.npy')


            np.save(reconst_data_filename, TensorReconst[0,:,:,n_frame])


            # PREVIOUS
            # np.save(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images_data',
                                 # 'reconstructed_image_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.npy'),
                                 # TensorReconst[0,:,:,n_frame])

            plt.figure()
            ax = plt.gca()
            im = ax.imshow(TensorReconst[0,:,:,n_frame], extent=[0, TensorReconst.shape[2], TensorReconst.shape[1], 0], cmap=plt.get_cmap(param['colormap']),
                           aspect="auto")

            if len(list_all_filenames) > 0:

                title_figure = 'Reconstructed image frame ' + str(n_frame) + '\n' + os.path.join(os.path.basename(os.path.split(list_all_filenames[n_frame])[0]),
                                                                                    os.path.basename(list_all_filenames[n_frame]))

            else:

                title_figure = 'Reconstructed image frame ' + str(n_frame)

            plt.title(title_figure)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)


            # sequence_name_extracted + '--hodmd_reconst--data',

            reconst_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images',
                                                  class_name_extracted, sequence_name_extracted + '--hodmd_reconst--images',
                                                  sequence_name_extracted + '--hodmd_reconst--images_' +
                                                  str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png')


            plt.savefig(reconst_image_filename)

            # plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'reconstructed_images',
                                     # 'reconstructed_image_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png'))

            # plt.show()
            plt.close()





        # GrowthRate, Frequency and Amplitude:
        # file2 = open(f'{path0}\DMD_solution\GrowthRateFrequencyAmplitude.txt', 'w')
        # file2.truncate(0)
        # file2.close()
        np.save(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_values', 'GrowthRateFrequencyAmplitude_' + param['svd_library'] + "_" + sequence_name_extracted + '.npy'), GRFreqAmp)

        # endregion



        # region Saving results of values
        np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_times', 'HOSVD_time_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                   np.array([total_hosvd_time]))

        np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_SVD_times', 'HODMD_IT_ONLY_SVD_time_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                   np.array([total_svd_time]))


        np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'HODMD_times', 'HODMD_IT_time_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                   np.array([total_hodmd_it_time]))

        # Just the last value of RRMSE obtained
        np.savetxt(os.path.join(param['results_path'], 'DMD_solution', 'RRMSE', 'finalRRMSE_' + param['svd_library'] + "_" + sequence_name_extracted + '.txt'),
                   np.array([RRMSE_it]))

        # endregion



        # region Save DMD modes
        # file3 = open(f'{path0}\DMD_solution\DMDmode.txt', 'w')
        # file3.truncate(0)
        # file3.close()



        # np.save(os.path.join(param['results_path'], 'DMD_solution', 'DMDmode_' + param['svd_library'] + '.npy'), DMDmode)

        n_modes = DMDmode.shape[-1]

        DMD_n1 = DMDmode.shape[1]
        DMD_n2 = DMDmode.shape[2]

        n_first_dim = DMDmode.shape[0]

        mode_count_pos = 0

        mode_count_neg = 0

        for n_first in range(n_first_dim):
            for n_mode in range(n_modes):


                if round(Frequency[n_mode], 4) >= 0:

                    mode_data_filename = os.path.join(param['results_path'], 'DMD_solution', 'real_data',
                                                      class_name_extracted,
                                                      sequence_name_extracted + '--hodmd_real--data',
                                                      sequence_name_extracted + '--hodmd_real--' + class_name_extracted + '_' + str(
                                                          n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(
                                                          mode_count_pos).zfill(len(str(n_modes)) + 1)
                                                      + '.npy')



                else:


                    mode_data_filename = os.path.join(param['results_path'], 'DMD_solution', 'real_data',
                                                      class_name_extracted,
                                                      sequence_name_extracted + '--hodmd_real_negative--data',
                                                      sequence_name_extracted + '--hodmd_real--' + class_name_extracted + '_' + str(
                                                          n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(
                                                          mode_count_neg).zfill(len(str(n_modes)) + 1)
                                                      + '.npy')



                np.save(mode_data_filename, np.real(DMDmode[n_first, :, :, n_mode]))


                # PREVIOUS
                # np.save(os.path.join(param['results_path'], 'DMD_solution', 'real_data',
                #         'DMDmode_real_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(n_mode+1).zfill(len(str(n_modes)) + 1)
                #         + '.npy'), np.real(DMDmode[n_first, :, :, n_mode]))


                plt.figure()
                ax = plt.gca()
                im = ax.imshow(np.real(DMDmode[n_first, :, :, n_mode]), extent=[0, DMD_n2, DMD_n1, 0], cmap=plt.get_cmap(param['colormap']), aspect="auto")

                plt.title('Mode: ' + str(n_mode+1).zfill(len(str(n_modes)) + 1) + ' Ampl: ' + str(
                    round(Amplitude[n_mode], 4)) + ' Freq: ' + str(round(Frequency[n_mode], 4)))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                if round(Frequency[n_mode], 4) >= 0:


                    mode_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted, sequence_name_extracted + '--hodmd_real--images',
                                                        sequence_name_extracted + '--hodmd_real--' + class_name_extracted + '_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(mode_count_pos).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                                                        round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png')

                else:

                    mode_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'real_images', class_name_extracted, sequence_name_extracted + '--hodmd_real_negative--images',
                                                        sequence_name_extracted + '--hodmd_real--' + class_name_extracted + '_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(mode_count_neg).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                                                        round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png')

                plt.savefig(mode_image_filename)


                # plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'real_images', 'DMDmode_real_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(n_mode+1).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                    # round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png'))

                # plt.show()
                plt.close()


                if round(Frequency[n_mode], 4) >= 0:

                    mode_data_filename = os.path.join(param['results_path'], 'DMD_solution', 'imag_data',
                                                      class_name_extracted, sequence_name_extracted + '--hodmd_imag--data',
                                                      sequence_name_extracted + '--hodmd_imag--' + class_name_extracted + '_' + str(
                                                      n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(
                                                      mode_count_pos).zfill(len(str(n_modes)) + 1)
                                                      + '.npy')

                else:


                    mode_data_filename = os.path.join(param['results_path'], 'DMD_solution', 'imag_data',
                                                      class_name_extracted, sequence_name_extracted + '--hodmd_imag_negative--data',
                                                      sequence_name_extracted + '--hodmd_imag--' + class_name_extracted + '_' + str(
                                                      n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(
                                                      mode_count_neg).zfill(len(str(n_modes)) + 1)
                                                      + '.npy')


                np.save(mode_data_filename, np.imag(DMDmode[n_first, :, :, n_mode]))

                #                 np.save(os.path.join(param['results_path'], 'DMD_solution', 'imag_data',
                #                              'DMDmode_imag_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(n_mode+1).zfill(len(str(n_modes)) + 1)
                #                                      + '.npy'), np.imag(DMDmode[n_first, :, :, n_mode]))



                plt.figure()
                ax = plt.gca()
                im = ax.imshow(np.imag(DMDmode[n_first, :, :, n_mode]), extent=[0, DMD_n2, DMD_n1, 0], cmap=plt.get_cmap(param['colormap']),
                           aspect="auto")

                plt.title('Mode: ' + str(n_mode+1).zfill(len(str(n_modes)) + 1) + ' Ampl: ' + str(
                    round(Amplitude[n_mode], 4)) + ' Freq: ' + str(round(Frequency[n_mode], 4)))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                if round(Frequency[n_mode], 4) >= 0:


                    mode_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'imag_images',
                                                       class_name_extracted,
                                                       sequence_name_extracted + '--hodmd_imag--images',
                                                       sequence_name_extracted + '--hodmd_imag--' + class_name_extracted + '_' + str(
                                                           n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(
                                                           mode_count_pos).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                                                           round(Amplitude[n_mode], 4)) + '_F_' + str(
                                                           round(Frequency[n_mode], 4)) + '.png')

                else:

                    mode_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'imag_images',
                                                       class_name_extracted,
                                                       sequence_name_extracted + '--hodmd_imag_negative--images',
                                                       sequence_name_extracted + '--hodmd_imag--' + class_name_extracted + '_' + str(
                                                           n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(
                                                           mode_count_neg).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                                                           round(Amplitude[n_mode], 4)) + '_F_' + str(
                                                           round(Frequency[n_mode], 4)) + '.png')



                plt.savefig(mode_image_filename)


                # plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'imag_images', 'DMDmode_imag_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(n_mode+1).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                    # round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png'))

                # plt.show()
                plt.close()

                if round(Frequency[n_mode], 4) >= 0:

                    mode_data_filename = os.path.join(param['results_path'], 'DMD_solution', 'abs_data',
                                                      class_name_extracted, sequence_name_extracted + '--hodmd_abs--data',
                                                      sequence_name_extracted + '--hodmd_abs--' + class_name_extracted + '_' + str(
                                                      n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(
                                                      mode_count_pos).zfill(len(str(n_modes)) + 1)
                                                      + '.npy')

                else:

                    mode_data_filename = os.path.join(param['results_path'], 'DMD_solution', 'abs_data',
                                                      class_name_extracted, sequence_name_extracted + '--hodmd_abs_negative--data',
                                                      sequence_name_extracted + '--hodmd_abs--' + class_name_extracted + '_' + str(
                                                      n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(
                                                      mode_count_neg).zfill(len(str(n_modes)) + 1)
                                                      + '.npy')



                np.save(mode_data_filename, np.absolute(DMDmode[n_first, :, :, n_mode]))

                #                 np.save(os.path.join(param['results_path'], 'DMD_solution', 'abs_data',
                #                              'DMDmode_abs_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_mode_' + str(n_mode+1).zfill(len(str(n_modes)) + 1)
                #                                      + '.npy'), np.absolute(DMDmode[n_first, :, :, n_mode]))

                plt.figure()
                ax = plt.gca()
                im = ax.imshow(np.absolute(DMDmode[n_first, :, :, n_mode]), extent=[0, DMD_n2, DMD_n1, 0], cmap=plt.get_cmap(param['colormap']),
                           aspect="auto")

                plt.title('Mode: ' + str(n_mode+1).zfill(len(str(n_modes)) + 1) + ' Ampl: ' + str(
                    round(Amplitude[n_mode], 4)) + ' Freq: ' + str(round(Frequency[n_mode], 4)))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                if round(Frequency[n_mode], 4) >= 0:


                    mode_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted, sequence_name_extracted + '--hodmd_abs--images',
                                                        sequence_name_extracted + '--hodmd_abs--' + class_name_extracted + '_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(mode_count_pos).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                                                        round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png')

                else:

                    mode_image_filename = os.path.join(param['results_path'], 'DMD_solution', 'abs_images', class_name_extracted, sequence_name_extracted + '--hodmd_abs_negative--images',
                                                        sequence_name_extracted + '--hodmd_abs--' + class_name_extracted + '_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(mode_count_neg).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                                                        round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png')


                plt.savefig(mode_image_filename)


                # plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'abs_images', 'DMDmode_abs_' + str(n_first).zfill(len(str(n_first_dim)) + 1) + '_m_' + str(n_mode+1).zfill(len(str(n_modes)) + 1) + '_A_' + str(
                    # round(Amplitude[n_mode], 4)) + '_F_' + str(round(Frequency[n_mode], 4)) + '.png'))

                # plt.show()
                plt.close()

                if round(Frequency[n_mode], 4) >= 0:

                    mode_count_pos = mode_count_pos + 1


                else:

                    mode_count_neg = mode_count_neg + 1


        # endregion

        # region Result plots
        print('Results have been saved in the following folder: ' + os.path.join(param['results_path'], 'DMD_solution') + '\n')
        print('ATTENTION! The run will not end until all figure windows are closed.')
        print('Recommendation: Close figures and check folder.')

        # Frequency vs absolute value of GrowthRate:
        plt.figure(num='CLOSE TO END RUN - Frequency/GrowthRate')
        plt.plot(Frequency,np.abs(GrowthRate), 'k+')
        plt.yscale('log')           # Logarithmic scale in y axis
        plt.xlabel('Frequency ($\omega_{m}$)')
        plt.ylabel('Absolute value of GrowthRate (|$\delta_{m}$|)')
        plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'GrowthRates', 'FrequencyGrowthRate_ ' + param['svd_library'] + '_' + sequence_name_extracted + '.png'))


        # Frequency vs amplitudes:
        # TODO: DUDA SOBRE LO DE LOS BPM (VER CÓDIGO EN MATLAB PARA RECORDAR): el factor 10 que se aplicaba en matlab y aquí no:
        #  error??
        plt.figure(num='CLOSE TO END RUN - Frequency/Amplitude')
        plt.plot(Frequency, Amplitude,'r+')
        plt.yscale('log')           # Logarithmic scale in y axis
        plt.xlabel('Frequency ($\omega_{m}$)')
        plt.ylabel('Amplitude ($a_{m}$)')
        plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'Amplitudes', 'FrequencyAmplitude_ ' + param['svd_library'] + '_' + sequence_name_extracted + '.png'))

        plt.close()
        # plt.show()

        # También te paso el código de la gráfica que representa los modos retenidos en main_HOSVD.py
        # TODO: entiendo que es algo así, ¿no?

        fig, ax = plt.subplots()
        markers = ['yo', 'gx', 'r*', 'bv', 'y+']
        labels = ["Variable Singular Values",
                  "X Space Singular Values",
                  "Y Space Singular Values",
                  "Z Space Singular Values",
                  "Time Singular Values"]

        sub_axes = plt.axes([.6, .6, .25, .2])

        # nn1 instead of n?
        if np.array(nn1).size == 4: # Si tiene 4 dims, significa que no hay componente "Z"
            labels.remove("Z Space Singular Values")

        # nn1 instead of n?
        for i in range(np.array(nn1).size):
            ax.plot(sv[0, i] / sv[0, i][0], markers[i])
            sub_axes.plot(sv[0, i][:nn1[i]] / sv[0, i][0], markers[i])

        # nn1 instead of n?
        ax.hlines(y=param['varepsilon1'], xmin = 0, xmax=np.array(nn1).max(), linewidth=2, color='black', label = f"SVD tolerance: {param['varepsilon1']}")   # Para trazar la linea de la tolerancia

        ax.set_yscale('log')           # Logarithmic scale in y axis
        sub_axes.set_yscale('log')
        ax.set_xlabel('SVD modes')

        ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_title('SVD modes vs. Singular Values')
        sub_axes.set_title('Retained Singular Values', fontsize = 8)
        plt.savefig(os.path.join(param['results_path'], 'DMD_solution', 'HOSVD_tols', 'HOSVD_solution_TOL_' + str(param['varepsilon1']) + '_' + param['svd_library'] + '_' + sequence_name_extracted + '_' + 'SingularValues.png'))

        plt.close()
        # plt.show()



# endregion

# endregion

# endregion
