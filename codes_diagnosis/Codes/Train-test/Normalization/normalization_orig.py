# region Load libraries

import numpy as np
from glob import glob
import os
import cv2
import psutil

# endregion

# region Input parameters
param = {}


# Param input directory
param['input_data_directories'] = ['/home/w460/SCRATCH/Andres/Databases/Mice_ecos/Ecos_tutorials_AI/ecos_orig/ecos_orig_Training']

param['exported_filename'] = 'normalization_orig_224.txt'

param['excluded_classes'] = []

param['depth_final_data'] = np.float32

param['img_size'] = (224, 224)

param['Preadjust_scale'] = False


# endregion



# region Functions

def check_ram_usage():
    # Get system memory usage
    memory_info = psutil.virtual_memory()

    print("Total Memory:", convert_bytes(memory_info.total))
    print("Available Memory:", convert_bytes(memory_info.available))
    print("Used Memory:", convert_bytes(memory_info.used))
    print("Memory Usage Percentage:", memory_info.percent, "%")

def convert_bytes(num):
    """
    Convert bytes to a human-readable format.
    """
    for x in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


# endregion





# region Program


all_database_images_filenames = []
all_database_images_class_names = []
class_names = []
total_num_samples = 0

mean_final = 0
std_final = 0


for img_dir_idx, img_dir in enumerate(param['input_data_directories']):

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

            if not (os.path.basename(database_class_name_fullpath) in param['excluded_classes'] ):

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


total_num_samples = total_num_samples + len(all_database_images_filenames)

print('Database with ' + str(total_num_samples) + ' samples and ' + str(len(database_images_directory)) + ' classes')

check_ram_usage()

for img_idx, img_path in enumerate(all_database_images_filenames):

    if '.npy' in img_path:

        sample = np.load(img_path)

        sample = np.expand_dims(cv2.resize(sample.astype(param['depth_final_data']), param['img_size']), axis=2)

        if img_idx == 0:
            print('Numpy format')
            all_samples = np.zeros((total_num_samples, param['img_size'][0], param['img_size'][1], 1),
                                  dtype=sample.dtype)

            check_ram_usage()

        all_samples[img_idx] = sample

if param['Preadjust_scale']:
    all_samples = all_samples/255

mean_final = np.mean(all_samples)
std_final = np.std(all_samples)

print('Mean: ')
print(mean_final)

print('Std: ')
print(std_final)

f = open(param['exported_filename'], "a")
f.write("Mean: " + str(mean_final) + " Std: " + str(std_final))
f.close()


# endregion
