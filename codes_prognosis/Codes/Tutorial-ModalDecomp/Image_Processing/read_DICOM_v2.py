
# Import libraries

import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
import os
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob

import pandas as pd


# endregion

param = {}


# region Input parameters


# region Database path
### Type a database path:
### - For example: /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others

###     /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others/CTL
###     /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others/CTL/0056W_87_LA.dcm
###     /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others/CTL/0056W_88_LA.dcm
###     /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others/DB
###     /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others/DB/0020W_123_LA.dcm
###     /home/andres/andres/Databases/New_ecos/USEFUL/LAX_Others/DB/0032W_128_LA.dcm

param['database_path'] = input('Write a path in which data are stored: ')

# endregion

# region Param Number of snapshots nsnap

### Just enter (that is, export all snapshots of each sequence)
param['SNAP'] = input('Introduce number of snapshots or press enter to continue with a number of snapshots corresponding to the size of the temporal dimension of the input data: ')

if param['SNAP']:
    param['Introduced_SNAP'] = True
    param['SNAP'] = int(param['SNAP'])
else:
    param['Introduced_SNAP'] = False
# endregion

# region Param Results path

param['results_path'] = input('Write a path to save results: ')

# endregion


# region Param ColorMap

### Just enter (gray image)
param['colormap'] = input('Introduce colormap or press enter to introduce gray: ')

if not param['colormap']:

    param['colormap'] = 'gray'

# endregion

# region Param Experimental: Activate algorithm to find graphics and perform inpainting
# NOTE: the graphics in the mice databases seem to have color.

### Set to 1
param['perform_inpainting'] = input('Activate or not the inpainting algorithm (EXPERIMENTAL; default: deactivated): ')

if not param['perform_inpainting']:

    param['perform_inpainting'] = 0

else:
    param['perform_inpainting'] = 1
# endregion


# region Param Experimental: Activate detection of letters (as outliers)

### Just enter (that is, set to 0), so ignore the params 'lower_th' and 'upper_th'
param['letter_recognition'] = input('Activate or not the letter recognition (EXPERIMENTAL; default: deactivated): ')

if not(param['letter_recognition']):

    param['letter_recognition'] = 0

else:

    param['letter_recognition'] = 1

if param['letter_recognition']:

    param['lower_th'] = input('Set lower threshold for the letter recognition process or press enter to introduce 210 (EXPERIMENTAL): ')

    if not(param['lower_th']):

        param['lower_th'] = 210

    param['upper_th'] = input('Set upper threshold for the letter recognition process or press enter to introduce 220 (EXPERIMENTAL): ')

    if not (param['upper_th']):

        param['upper_th'] = 220

# endregion


### Set to 'yes'
param['export_data'] = input('\n' + 'Are data exported? Yes or No (y/n). Press enter to continue with no: ')
if not param['export_data']:
    param['export_data'] = 0
elif (param['export_data']=='n' or param['export_data']=='N' or param['export_data']=='no' or param['export_data']=='No' or param['export_data']=='NO'):
    param['export_data'] = 0
elif (param['export_data']=='y' or param['export_data']=='Y' or param['export_data']=='yes' or param['export_data']=='Yes' or param['export_data']=='YES'):
    param['export_data'] = 1




# region Param Read and print video information

### Set to 'yes'
param['read_info'] = input('Read video info? Yes or No (y/n). Press enter to continue with no: ')
if not param['read_info']:
    param['read_info'] = 0
elif (param['read_info']=='n' or param['read_info']=='N' or param['read_info']=='no' or param['read_info']=='No' or param['read_info']=='NO'):
    param['read_info'] = 0
elif (param['read_info']=='y' or param['read_info']=='Y' or param['read_info']=='yes' or param['read_info']=='Yes' or param['read_info']=='YES'):
    param['read_info'] = 1

# endregion


# endregion

# region Program


def get_area_limits(img):

    # ymin # left (INCLUDED)
    # ymax # right (EXCLUDED)
    # xmin # up (INCLUDED)
    # xmax # down (EXCLUDED)


    non_zero_axis_0 = np.count_nonzero(img, axis = 0)
    non_zero_axis_1 = np.count_nonzero(img, axis = 1)

    non_zero_axis_0_diff = np.diff(non_zero_axis_0, prepend = 0)

    ymin = np.argsort(non_zero_axis_0_diff[0:round(img.shape[1] / 3)])[::-1][0] + 3 # Just in case to delete some of the black degradation
    ymax = np.argsort(non_zero_axis_0_diff[2*round(img.shape[1] / 3):])[0] + 2*round(img.shape[1] / 3)

    non_zero_axis_1_diff = np.diff(non_zero_axis_1, prepend = 0)

    argsorted_axis1_diff_1stpart = np.argsort(non_zero_axis_1_diff[0:round(img.shape[0]/3)])[::-1]
    xmin = np.max(argsorted_axis1_diff_1stpart[:6])

    argsorted_axis1_diff_2ndpart = np.argsort(non_zero_axis_1_diff[2 * round(img.shape[0] / 3):]) + 2 * round(img.shape[0] / 3)
    xmax = np.min(argsorted_axis1_diff_2ndpart[:6])


    return (ymin, ymax, xmin, xmax)


classes_subfolders = glob(os.path.join(param['database_path'], '*'))
classes_subfolders.sort()

if not (param['export_data']):

    if not os.path.exists(param['results_path']):
        os.makedirs(param['results_path'])

    list_heights = [[]] * len(classes_subfolders)
    list_weights = [[]] * len(classes_subfolders)

if param['export_data']:

    if not os.path.exists(os.path.join(param['results_path'], 'original_images')):
        os.makedirs(os.path.join(param['results_path'], 'original_images'))

    if not os.path.exists(os.path.join(param['results_path'], 'no_cropped_images')):
        os.makedirs(os.path.join(param['results_path'], 'no_cropped_images'))

    if not os.path.exists(os.path.join(param['results_path'], 'original_data')):
        os.makedirs(os.path.join(param['results_path'], 'original_data'))


if param['letter_recognition']:

    if param['export_data']:

        if not os.path.exists(os.path.join(param['results_path'], 'original_data_excluded')):
            os.makedirs(os.path.join(param['results_path'], 'original_data_excluded'))

        if not os.path.exists(os.path.join(param['results_path'], 'original_images_excluded')):
            os.makedirs(os.path.join(param['results_path'], 'original_images_excluded'))


all_data_filenames = [[]]*len(classes_subfolders)
all_data_type = [[]]*len(classes_subfolders)

labels_excel = ['Full path', 'Basename', 'Class name', 'Final Shape', 'Frames', 'Height', 'Width']

for idx_class, class_subfolder in enumerate(classes_subfolders):

    metadata_info_list = []

    class_name = os.path.basename(class_subfolder)

    print(class_name)

    if param['export_data']:

        if not os.path.exists(os.path.join(param['results_path'], 'original_images', class_name)):
            os.makedirs(os.path.join(param['results_path'], 'original_images', class_name))

        if not os.path.exists(os.path.join(param['results_path'], 'no_cropped_images', class_name)):
            os.makedirs(os.path.join(param['results_path'], 'no_cropped_images', class_name))

        if not os.path.exists(os.path.join(param['results_path'], 'original_data', class_name)):
            os.makedirs(os.path.join(param['results_path'], 'original_data', class_name))

    if param['letter_recognition']:

        if param['export_data']:

            if not os.path.exists(os.path.join(param['results_path'], 'original_data_excluded', class_name)):
                os.makedirs(os.path.join(param['results_path'], 'original_data_excluded', class_name))

            if not os.path.exists(os.path.join(param['results_path'], 'original_images_excluded', class_name)):
                os.makedirs(os.path.join(param['results_path'], 'original_images_excluded', class_name))

    dicom_sequences_subfolder = glob(os.path.join(class_subfolder, '*'))
    dicom_sequences_subfolder.sort()

    for dicom_sequence_filename in dicom_sequences_subfolder:

        prefix_filename = os.path.splitext(os.path.basename(dicom_sequence_filename))[0]

        print(dicom_sequence_filename)
        print(prefix_filename)

        if not param['read_info']:

            Tensor0 = dicom.dcmread(dicom_sequence_filename).pixel_array

        else:

            Tensor0 = dicom.dcmread(dicom_sequence_filename)

            try:
                print(Tensor0.SeriesDescription)
            except:
                print('No Series Description')

            try:
                print(Tensor0.StudyDescription)

            except:
                print('No Study Description')

            Tensor0 = Tensor0.pixel_array

        if param['export_data']:

            if not os.path.exists(os.path.join(param['results_path'], 'original_images', class_name, prefix_filename + '--original--images')):
                os.makedirs(os.path.join(param['results_path'], 'original_images', class_name, prefix_filename + '--original--images'))

            if not os.path.exists(os.path.join(param['results_path'], 'no_cropped_images', class_name, prefix_filename + '--no_cropped_images--')):
                os.makedirs(os.path.join(param['results_path'], 'no_cropped_images', class_name, prefix_filename + '--no_cropped_images--'))

            if not os.path.exists(os.path.join(param['results_path'], 'original_data', class_name, prefix_filename + '--original--data')):
                os.makedirs(os.path.join(param['results_path'], 'original_data', class_name, prefix_filename + '--original--data'))


        if param['letter_recognition']:

            if param['export_data']:

                if not os.path.exists(os.path.join(param['results_path'], 'original_data_excluded', class_name, prefix_filename + '--original--data')):
                    os.makedirs(os.path.join(param['results_path'], 'original_data_excluded', class_name, prefix_filename + '--original--data'))

                if not os.path.exists(os.path.join(param['results_path'], 'original_images_excluded', class_name, prefix_filename + '--original--images')):
                    os.makedirs(os.path.join(param['results_path'], 'original_images_excluded', class_name, prefix_filename + '--original--images'))

        if Tensor0.ndim < 4:
            print("Not a video: " + str(Tensor0.shape))

            if param['export_data']:

                plt.figure()
                ax = plt.gca()
                im = ax.imshow(Tensor0)

                plt.title('No cropped single frame')

                plt.savefig(os.path.join(param['results_path'], 'no_cropped_images', class_name, prefix_filename + '--no_cropped_images--',
                                         prefix_filename + '_' + class_name + '--no_cropped_images--' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png'))

                # plt.show()
                plt.close()

            continue

        print('Total number of frames: ' + str(Tensor0.shape[0]))

        if not(param['Introduced_SNAP']):
            param['SNAP'] = Tensor0.shape[0]

        for n_frame in range(Tensor0.shape[0]):

            if np.count_nonzero(Tensor0[n_frame, :, :, :]) == 0:
                print('Found empty frame in ' + prefix_filename)
                print('Frame ' + str(n_frame))
                continue

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
                # ymin = 390 - 1  # left
                # ymax = 780  # right
                # xmin = 140 - 1  # up
                # xmax = 650  # down

                plt.figure()
                ax = plt.gca()
                im = ax.imshow(Tensor0[n_frame, xmin:xmax, ymin:ymax, :])
                plt.title('Frame: ' + str(n_frame))
                plt.show()
                time.sleep(1e-2)
                plt.close()

            # Export the no cropped images from the sequence

            if param['export_data']:

                plt.figure()
                ax = plt.gca()
                im = ax.imshow(Tensor0[n_frame, :, :, :])

                plt.title('No cropped frame ' + str(n_frame))

                plt.savefig(os.path.join(param['results_path'], 'no_cropped_images', class_name, prefix_filename + '--no_cropped_images--',
                                         prefix_filename + '_' + class_name + '--no_cropped_images--' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png'))

                # plt.show()
                plt.close()

            if n_frame == 0:
                (ymin, ymax, xmin, xmax) = get_area_limits(cv2.cvtColor(Tensor0[n_frame, :, :, :], cv2.COLOR_BGR2GRAY))

                # Identify rows where all elements are zero
                zero_rows = np.all(Tensor0[n_frame, xmin:xmax, ymin:ymax, 0] == 0, axis=1)

                # Get the indices of the rows that contain only zeros
                zero_row_indices = np.where(zero_rows)[0]

                if len(zero_row_indices) > 0:
                    print('Warning: encountered redundant rows. Modifying the limits...')

                    xmax = np.min(zero_row_indices) + xmin

                print('ymin = ' + str(ymin))
                print('ymax = ' + str(ymax))

                print('xmin = ' + str(xmin))
                print('xmax = ' + str(xmax))

            gray_image = cv2.cvtColor(Tensor0[n_frame, xmin:xmax, ymin:ymax, :], cv2.COLOR_BGR2GRAY)

            if n_frame == 0:
                print('Image dimensions (after cropping)')
                print(gray_image.shape)

                # labels_excel = ['Full path', 'Basename', 'Class name', 'Final Shape', 'Frames', 'Height', 'Width']
                metadata_info_sample = [dicom_sequence_filename, prefix_filename, class_name, (Tensor0.shape[0], gray_image.shape[0], gray_image.shape[1]),
                                        Tensor0.shape[0], gray_image.shape[0], gray_image.shape[1]]

                metadata_info_list.append(metadata_info_sample)

            if param['perform_inpainting']:

                # Split the image into its RGB channels
                b, g, r = cv2.split(Tensor0[n_frame, xmin:xmax, ymin:ymax, :])

                # Create a mask for pixels where the RGB values are not equal
                non_gray_mask = (r != g) | (r != b) | (g != b)

                # Convert the mask to a uint8 format for visualization or further processing
                non_gray_mask = non_gray_mask.astype(np.uint8) * 255

                if np.any(non_gray_mask):
                    print('Graphics in frame: ' + str(n_frame))

                    # Perform inpainting on the original image using the mask of the white lines
                    # TODO: SET THRESHOLD, OR DILATE THE MASK A LITTLE BIT BEFORE FILTERING
                    kernel = np.ones((3, 3), np.uint8)
                    non_gray_mask_dilated = cv2.dilate(non_gray_mask, kernel, iterations=1)
                    gray_image = cv2.inpaint(gray_image, non_gray_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            if not param['export_data'] and (n_frame == 0):

                list_heights[idx_class].append(gray_image.shape[0])

                list_weights[idx_class].append(gray_image.shape[1])

                break


            # Export the original images from the sequence

            if param['export_data']:
                plt.figure()
                ax = plt.gca()
                im = ax.imshow(gray_image, extent=[0, gray_image.shape[1], gray_image.shape[0], 0],
                               cmap=plt.get_cmap(param['colormap']),
                               aspect="auto")

                plt.title('Original image frame ' + str(n_frame))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            if param['letter_recognition']:


                my_image_preprocessed_with_text = gray_image.copy()
                norm_img = np.zeros(gray_image.shape)

                my_image_preprocessed_with_text = cv2.normalize(my_image_preprocessed_with_text, norm_img, 0, 255,
                                                                cv2.NORM_MINMAX)

                my_image_preprocessed_with_text = cv2.threshold(my_image_preprocessed_with_text, param['lower_th'], param['upper_th'], cv2.THRESH_BINARY)[1]

                my_image_preprocessed_with_text = cv2.GaussianBlur(my_image_preprocessed_with_text, (1, 1), 0)

                text_extracted = pytesseract.image_to_string(my_image_preprocessed_with_text)

                new_text_extracted=''.join(c for c in text_extracted if c.isalnum())


                if len(new_text_extracted) > 0:
                    print('Text found in frame: ' + str(n_frame))

                    if param['export_data']:
                        plt.savefig(os.path.join(param['results_path'], 'original_images_excluded', class_name,
                                                 prefix_filename + '--original--images',
                                                 prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(
                                                     len(str(param['SNAP'])) + 1) + '.png'))

                        np.save(os.path.join(param['results_path'], 'original_data_excluded', class_name,
                                             prefix_filename + '--original--data',
                                             prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(
                                                 len(str(param['SNAP'])) + 1) + '.npy'),
                                             gray_image)

                else:

                    if param['export_data']:

                        plt.savefig(os.path.join(param['results_path'], 'original_images', class_name, prefix_filename + '--original--images',
                                                 prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png'))


                        np.save(os.path.join(param['results_path'], 'original_data', class_name, prefix_filename + '--original--data',
                                             prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.npy'),
                                             gray_image)

            else:

                if param['export_data']:

                    plt.savefig(os.path.join(param['results_path'], 'original_images', class_name, prefix_filename + '--original--images',
                                             prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.png'))


                    np.save(os.path.join(param['results_path'], 'original_data', class_name, prefix_filename + '--original--data',
                                         prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.npy'),
                                         gray_image)

            if param['export_data']:
                # plt.show()
                plt.close()

            # all_data_filenames[idx_class].append(os.path.join(param['results_path'], 'original_data', class_name,
                                                 # prefix_filename + '--original--' + class_name + '_' + str(n_frame).zfill(len(str(param['SNAP'])) + 1) + '.npy'))

            # all_data_type[idx_class].append('original')

            if n_frame == param['SNAP'] - 1:
                break

    if not param['export_data']:

        np.save(os.path.join(param['results_path'], 'list_heights_' + str(idx_class)), np.array(list_heights[idx_class]))
        np.save(os.path.join(param['results_path'], 'list_weights_' + str(idx_class)), np.array(list_weights[idx_class]))


    df_metadata = pd.DataFrame(metadata_info_list, columns=labels_excel)

    # Save MAC matrix as an Excel file
    excel_path = os.path.join(param['results_path'], "Metadata_databases_" + class_name + ".xlsx")
    df_metadata.to_excel(excel_path)
    print(f"Metadata of the databases saved to {excel_path}")


print('Finished!')

# endregion










