import hdf5storage
import numpy as np
import pickle
import pandas as pd
import h5py
import pydicom as dicom
import cv2
import os
from glob import glob

from sys import platform


def load_data_sequences(wantedfile):


    list_files = glob(os.path.join(wantedfile, '*'))

    # CLASS_NAMES
    list_files.sort()

    Tensor = [[]]*len(list_files)


    list_data_info = [[]]*3

    list_data_info[0] = list_files

    list_data_info[1] = [[]] * len(list_files)

    list_data_info[2] = [[]] * len(list_files)

    total_number_samples = 0

    for idx_file, file_name in enumerate(list_files):

        # Count total number of files
        if os.path.isdir(file_name):
            list_files_2 = glob(os.path.join(file_name, '*'))

            # SEQUENCES
            list_files_2.sort()

            list_data_info[1][idx_file] = list_files_2

            list_data_info[2][idx_file] = [[]] * len(list_files_2)

            for idx_file_2, file_name_2 in enumerate(list_files_2):

                # FILES IN SEQUENCE
                list_files_3 = glob(os.path.join(file_name_2, '*'))

                list_files_3.sort()

                total_number_samples = total_number_samples + len(list_files_3)



                list_data_info[2][idx_file][idx_file_2] = list_files_3

    actual_idx = 0


    for idx_file, file_name in enumerate(list_files):


        if os.path.isdir(file_name):

            list_files_2 = glob(os.path.join(file_name, '*'))

            # SEQUENCES
            list_files_2.sort()

            Tensor[idx_file] = [[]] * len(list_files_2)

            for idx_file_2, file_name_2 in enumerate(list_files_2):

                list_files_3 = glob(os.path.join(file_name_2, '*'))

                # SAMPLES
                list_files_3.sort()

                for idx_file_3, file_name_3 in enumerate(list_files_3):


                    file_extension_3 = os.path.splitext(file_name_3)[1]

                    if 'npy' in file_extension_3:
                        # Check the operating system
                        if platform in ["linux", "linux2"]:
                            Tensor_individual = np.load(file_name_3)
                        elif platform == "darwin":
                            Tensor_individual = np.load(file_name_3)
                        elif platform in ["win32", "win64"]:
                            Tensor_individual = np.load(file_name_3)

                    if (idx_file_3 == 0):
                        Tensor_sequence = np.zeros((1, Tensor_individual.shape[0],
                                          Tensor_individual.shape[1],
                                          len(list_files_3)))

                    Tensor_sequence[0, :, :, idx_file_3] = Tensor_individual


                    actual_idx = actual_idx + 1

                Tensor[idx_file][idx_file_2] = Tensor_sequence


    return Tensor, list_data_info

def load_data_folder(wantedfile):

    list_files = glob(os.path.join(wantedfile, '*'))
    list_files.sort()

    list_all_filenames = []

    total_number_samples = 0

    for idx_file, file_name in enumerate(list_files):

        # Count total number of files (case with subfolders)
        if os.path.isdir(file_name):
            all_files_subfolder = glob(os.path.join(file_name, '*'))
            all_files_subfolder.sort()

            total_number_samples = total_number_samples + len(all_files_subfolder)

        # Directly import into Tensor when having directly the samples
        if not(os.path.isdir(file_name)):

            file_extension = os.path.splitext(file_name)[1]

            if 'npy' in file_extension:
                # Check the operating system
                if platform in ["linux", "linux2"]:
                    Tensor_individual = np.load(file_name)
                elif platform == "darwin":
                    Tensor_individual = np.load(file_name)
                elif platform in ["win32", "win64"]:
                    Tensor_individual = np.load(file_name)

            if idx_file == 0:
                Tensor = np.zeros((1, Tensor_individual.shape[0], Tensor_individual.shape[1], len(list_files)))

            Tensor[0, :, :, idx_file] = Tensor_individual
            list_all_filenames.append(file_name)


    # Case with several sufolders -> deep into each one and add each tensor
    actual_idx = 0
    for idx_file, file_name in enumerate(list_files):
        if os.path.isdir(file_name):


            all_files_subfolder = glob(os.path.join(file_name, '*'))
            all_files_subfolder.sort()

            for idx_file_2, file_name_2 in enumerate(all_files_subfolder):

                file_extension_2 = os.path.splitext(file_name_2)[1]

                if 'npy' in file_extension_2:
                    # Check the operating system
                    if platform in ["linux", "linux2"]:
                        Tensor_individual = np.load(file_name_2)
                    elif platform == "darwin":
                        Tensor_individual = np.load(file_name_2)
                    elif platform in ["win32", "win64"]:
                        Tensor_individual = np.load(file_name_2)

                if (idx_file == 0) and (idx_file_2 == 0):
                    Tensor = np.zeros((1, Tensor_individual.shape[0],
                                       Tensor_individual.shape[1],
                                       total_number_samples))

                Tensor[0, :, :, actual_idx] = Tensor_individual
                list_all_filenames.append(file_name_2)
                actual_idx = actual_idx + 1


    return Tensor, list_all_filenames

def load_dcm_data(wantedfile, snap = None):

    Tensor_ = dicom.dcmread(f'{wantedfile}').pixel_array

    if snap is None:
        param_snap = Tensor_.shape[0]
    else:
        param_snap = snap

    ymin = 390 - 1  # left
    ymax = 780  # right
    xmin = 140 - 1  # up
    xmax = 650  # down

    for n_frame in range(Tensor_.shape[0]):

        gray_image = cv2.cvtColor(Tensor_[n_frame, xmin:xmax, ymin:ymax, :], cv2.COLOR_BGR2GRAY)

        # Note: we directly do this step to make the algorithm a little faster...
        if n_frame == 0:
            Tensor = np.zeros((1, gray_image.shape[0], gray_image.shape[1], param_snap))

        Tensor[0, :, :, n_frame] = gray_image

        if n_frame == param_snap - 1:
            break

    return Tensor

def load_data(format, wantedfile, snapshots = None):
    # LOAD MATLAB .mat FILES
    if format == 'mat':

        list_wanted_filenames = [wantedfile]

        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor_ = hdf5storage.loadmat(f'{wantedfile}')
            Tensor = list(Tensor_.values())[-1]
        elif platform == "darwin":
            # OS X
            Tensor_ = hdf5storage.loadmat(f'{wantedfile}')
            Tensor = list(Tensor_.values())[-1]
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor_ = hdf5storage.loadmat(f'{wantedfile}')
            Tensor = list(Tensor_.values())[-1]
        return Tensor, list_wanted_filenames

#-----------------------------------------------------------------------

    # LOAD NUMPY .npy FILES
    elif format == 'npy':

        list_wanted_filenames = [wantedfile]

        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor = np.load(f'{wantedfile}')
        elif platform == "darwin":
            # OS X
            Tensor = np.load(f'{wantedfile}')
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor = np.load(f'{wantedfile}')
        return Tensor, list_wanted_filenames

#-----------------------------------------------------------------------

    # LOAD .csv FILES
    elif format == 'csv':

        list_wanted_filenames = [wantedfile]

        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor = pd.read_csv(f'{wantedfile}')
            Tensor = np.array(Tensor)
        elif platform == "darwin":
            # OS X
            Tensor = pd.read_csv(f'{wantedfile}')
            Tensor = np.array(Tensor)
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor = pd.read_csv(f'{wantedfile}')
            Tensor = np.array(Tensor)
        return Tensor, list_wanted_filenames

#-----------------------------------------------------------------------
  
    # LOAD .h5 FILES
    elif format == 'h5':

        list_wanted_filenames = [wantedfile]

        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            with h5py.File(f'{wantedfile}', 'r+') as file:
                for dataset_name in file:
                    Tensor = file[dataset_name][()]
        elif platform == "darwin":
            # OS X
            with h5py.File(f'{wantedfile}', 'r+') as file:
                for dataset_name in file:
                    Tensor = file[dataset_name][()]
        elif platform in ["win32", "win64"]:
            # Windows
            with h5py.File(f'{wantedfile}', 'r+') as file:
                for dataset_name in file:
                    Tensor = file[dataset_name][()]
        return Tensor, list_wanted_filenames

#-----------------------------------------------------------------------
   
    # LOAD .pkl FILES
    elif format == 'pkl':

        list_wanted_filenames = [wantedfile]

        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            with open(f'{wantedfile}', 'rb') as file:
                Tensor = pickle.load(file)
        elif platform == "darwin":
            # OS X
            with open(f'{wantedfile}', 'rb') as file:
                Tensor = pickle.load(file)
        elif platform in ["win32", "win64"]:
            # Windows
            with open(f'{wantedfile}', 'rb') as file:
                Tensor = pickle.load(file)
        return Tensor, list_wanted_filenames

#-----------------------------------------------------------------------

    # LOAD .dcm FILES
    elif format == 'dcm':

        list_wanted_filenames = [wantedfile]

        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor = load_dcm_data(wantedfile, snap=snapshots)
        elif platform == "darwin":
            # OS X
            Tensor = load_dcm_data(wantedfile, snap=snapshots)
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor = load_dcm_data(wantedfile, snap=snapshots)

        return Tensor, list_wanted_filenames

# -----------------------------------------------------------------------

    # LOAD DATA INSIDE A FOLDER
    elif format == 'folder':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor, list_wanted_filenames = load_data_folder(wantedfile)
        elif platform == "darwin":
            # OS X
            Tensor, list_wanted_filenames = load_data_folder(wantedfile)
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor, list_wanted_filenames = load_data_folder(wantedfile)

        return Tensor, list_wanted_filenames

# -----------------------------------------------------------------------

    # LOAD DATA INSIDE A FOLDER BASED ON SEQUENCES TO PROCESS SEPARATELY
    elif format == 'sequences':
        # Check the operating system
        if platform in ["linux", "linux2"]:
            # linux
            Tensor, list_wanted_info= load_data_sequences(wantedfile)
        elif platform == "darwin":
            # OS X
            Tensor, list_wanted_info = load_data_sequences(wantedfile)
        elif platform in ["win32", "win64"]:
            # Windows
            Tensor, list_wanted_info = load_data_sequences(wantedfile)

        return Tensor, list_wanted_info

# def main(filetype):
def main(param):
    while True:
        if param['filetype'].strip().lower() in ['mat', '.mat']:
            param['wantedfile'] = input('Introduce name of input data file (as in: Desktop/Folder/data.mat): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('mat', param['wantedfile'])
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')
        
        elif param['filetype'].strip().lower() in ['.npy', 'npy']:
            param['wantedfile'] = input('Introduce name of input data file (as in: Desktop/Folder/data.npy): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('npy', param['wantedfile'])
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')      

        elif param['filetype'].strip().lower() in ['csv', '.csv']:
            param['wantedfile'] = input('Introduce name of input data file (as in: Desktop/Folder/data.csv): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('csv', param['wantedfile'])
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')
        
        elif param['filetype'].strip().lower() in ['pkl', '.pkl']:
            param['wantedfile'] = input('Introduce name of input data file (as in: Desktop/Folder/data.pkl): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('pkl', param['wantedfile'])
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')

        elif param['filetype'].strip().lower() in ['h5', '.h5']:
            param['wantedfile'] = input('Introduce name of input data file (as in: Desktop/Folder/data.h5): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('h5', param['wantedfile'])
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')


        elif param['filetype'].strip().lower() in ['dcm', '.dcm']:
            param['wantedfile'] = input('Introduce name of input data file (as in: Desktop/Folder/data.dcm): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('dcm', param['wantedfile'], param['SNAP'])
                break
            else:
                print(f'\tError: File does not exist in the selected path\n')

        elif param['filetype'].strip().lower() in ['folder', '.folder']:
            param['wantedfile'] = input('Introduce name of input data folder (as in: Desktop/Folder/Data_Folder): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('folder', param['wantedfile'])
                break
            else:
                print(f'\tError: Folder does not exist in the selected path\n')

        elif param['filetype'].strip().lower() in ['sequences', '.sequences']:
            param['wantedfile'] = '/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials/LAX/Results_step1/original_data'
            # input('Introduce name of input data folder with sequences to separately process (as in: Desktop/Folder/Data_Folder): ')
            if os.path.exists(f"{param['wantedfile']}"):
                Tensor, list_wanted_info = load_data('sequences', param['wantedfile'])
                break
            else:
                print(f'\tError: Folder does not exist in the selected path\n')

    database = param['wantedfile'].strip('.mat').strip('.npy').strip('.csv').strip('.pkl').strip('.h5').strip('.dcm')

    return Tensor, list_wanted_info, database

