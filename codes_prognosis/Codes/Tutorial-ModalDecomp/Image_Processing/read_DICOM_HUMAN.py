# region Import libraries

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import cv2
import pandas as pd
import scipy

# endregion

# region Input parameters
param = {}

# Define the root directory containing folders and DICOM files
param['root_dir'] = '/home/ander/Escritorio/Workshop2025-2026/Databases/Human_ecos/Ecos_tutorials/DICOM_videos' # '/home/andres/andres/Databases/Humanfiles/Humanfiles/HumanDICOM'

# Perform image processing (Default: True)
param['image_processing'] = True

# Output directory to save processed images and npy files
param['output_dir'] = '/home/ander/Escritorio/Workshop2025-2026/Results_step1' # '/home/andres/andres/Databases/Humanfiles/Humanfiles/Results_step1'

# Apply ot not the removal of the while lines inside the region of interest
param['remove_white_lines'] = True

# EXPERIMENTAL (at the moment, Keep False)
param['second_inpainting'] = False #Yo creo que no hace falta porque la línea es verde y se quita cuando se quitan los brillos
param['remove_highlight'] = False

# Set to True at the end (after having decided which preprocessing methods will be applied)
param['get_square'] = True


os.makedirs(param['output_dir'], exist_ok=True)

# endregion

# region Functions

def line_intersection(p1, p2, p3, p4):
    """Calculate intersection of two lines defined by points."""
    A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]
    A2, B2 = p4[1] - p3[1], p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]
    determinant = A1 * B2 - A2 * B1
    if determinant == 0:
        return None  # Lines are parallel
    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant
    return int(x), int(y)


def func(x, a, b):
    y = a * x + b
    return y

def get_mask(img):
    
    # Step 1: Threshold the image to create a binary mask for the ROI
    _, thresholded = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    
    # Step 2: Find the largest contour, which is assumed to be the ROI boundary
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    roi_contour = contours[0]  # Largest contour as the ROI
    
    alto, ancho = img.shape
    mask2 = np.zeros_like(img)
    contour2 = cv2.drawContours(mask2, [cv2.convexHull(contours[1])], -1, (255), thickness=cv2.FILLED)
    if np.any(contour2[100:500, ancho//2]==255):
        roi_contour = np.vstack((contours[0], contours[1]))
    
    # Step 3: Post-process the contour by creating a convex hull to reconstruct the cone
    hull = cv2.convexHull(roi_contour)
    
    # Create a mask with the convex hull of the ROI filled in white
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [hull], -1, (255), thickness=cv2.FILLED)

    
        # Step 4: Detect the two main lines of the cone in the mask
    
    roi_contour_2 = roi_contour[:,0,:]
    roi_contour_3 = roi_contour_2[(roi_contour_2[:, 1] >= 100) & (roi_contour_2[:, 1] <= 380)]

    left_points = roi_contour_3[(roi_contour_3[:,0] <= 400)]
    right_points = roi_contour_3[(roi_contour_3[:,0] > 400)]

    line_left = scipy.optimize.curve_fit(func, xdata=left_points[:,0], ydata=left_points[:,1])[0]

    line_right = scipy.optimize.curve_fit(func, xdata=right_points[:,0], ydata=right_points[:,1])[0]

    y_1_left = func(200, line_left[0], line_left[1])
    y_2_left = func(300, line_left[0], line_left[1])

    y_1_right = func(500, line_right[0], line_right[1])
    y_2_right = func(600, line_right[0], line_right[1])

    intersection_point = line_intersection([200, y_1_left], [300, y_2_left], [500, y_1_right], [600, y_2_right])

    # Step 5: Fill the area defined by the convex hull and the peak intersection
    if intersection_point:
        cone_points = np.array([[200, y_1_left], [300, y_2_left], intersection_point, [500, y_1_right], [600, y_2_right]])
        cone_points = cone_points.astype(np.int32)

        cv2.fillConvexPoly(mask, cone_points, (255))
    
    
    # Step 6: Apply the mask to the original image to keep only the reconstructed ROI, setting everything outside to black
    result = cv2.bitwise_and(img, mask)
    
    return (result, mask)


def remove_highlight(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract the saturation channel
    saturation = hsv[:, :, 1]

    # Create a mask to detect pixels with high saturation
    sat_threshold = 50  # Saturation threshold (adjustable)
    mask = saturation > sat_threshold

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the mask to remove all colors
    gray[mask] = 0
    
    return gray
    
    

# def process_pixel_array(ds, save_path_base, param):
def process_pixel_array(ds, class_name, sequence_name, param):
    """
    Process the pixel array from a DICOM dataset and save frames as .png and .npy files.

    Parameters:
        ds: pydicom.dataset.FileDataset
            The DICOM dataset containing the pixel array.
        save_path_base: str
            Base path to save the output files.
        param: dict
            List of input parameters.
    """

    pixel_array = ds.pixel_array

    # save_path_data = save_path_base + '--original--data'
    save_path_data = os.path.join(param['output_dir'], 'original_data', class_name)

    # save_path_data_no_processed = save_path_base + '--no_processed--data'
    save_path_data_no_processed = os.path.join(param['output_dir'], 'no_processed_data', class_name)
    
    # save_path_img = save_path_base + '--original--images'
    save_path_img = os.path.join(param['output_dir'], 'original_images', class_name)
    
    # save_path_img_no_processed = save_path_base + '--no_processed--images'
    save_path_img_no_processed = os.path.join(param['output_dir'], 'no_processed_images', class_name)

    print(pixel_array.shape)

    # Check pixel array dimensions

    # The potentially useful data
    if pixel_array.ndim == 4:  # [num_frames, height, width, num_channels]
        num_frames, height, width, num_channels = pixel_array.shape

        save_path_data_seq = os.path.join(save_path_data, sequence_name + '--original--data')
        save_path_data_no_processed_seq = os.path.join(save_path_data_no_processed, sequence_name + '--no_processed--data')
        save_path_img_seq = os.path.join(save_path_img, sequence_name + '--original--images')
        save_path_img_no_processed_seq = os.path.join(save_path_img_no_processed, sequence_name + '--no_processed--images')

        os.makedirs(save_path_data_seq, exist_ok=True)
        os.makedirs(save_path_data_no_processed_seq, exist_ok=True)
        os.makedirs(save_path_img_seq, exist_ok=True)
        os.makedirs(save_path_img_no_processed_seq, exist_ok=True)


        
        #if os.path.exists(param['mask_path']):
            #mask = np.load(param['mask_path'])

        for frame_idx in range(num_frames):

            frame_idx_notated = str(frame_idx).zfill(len(str(num_frames)) + 1)
            
            frame_array = pixel_array[frame_idx, :, :, :]

            # region TODO: REMOVE WHEN READY
            #if frame_idx == num_frames - 1:
            #    save_image(frame_channel_array, os.path.join(save_path_img, f"{os.path.basename(save_path_base)}--no_processed--{param['name_class']}_channel_{channel_idx}_{frame_idx_notated}"))
            # endregion

            # region TODO: UNCOMMENT WHEN READY
            # save_image(frame_array, os.path.join(save_path_img_no_processed, f"{os.path.basename(save_path_base)}--no_processed--{param['name_class']}_{frame_idx_notated}"))
            # save_npy(frame_array, os.path.join(save_path_data_no_processed, f"{os.path.basename(save_path_base)}--no_processed--{param['name_class']}_{frame_idx_notated}"))
            save_image(frame_array, os.path.join(save_path_img_no_processed_seq, f"{sequence_name}--no_processed--{class_name}_{frame_idx_notated}"))
            save_npy(frame_array, os.path.join(save_path_data_no_processed_seq, f"{sequence_name}--no_processed--{class_name}_{frame_idx_notated}"))


            # endregion
            
            #Remove highlights
            if param['remove_highlight']:
                frame_result = remove_highlight(pixel_array[frame_idx,:,:,:])  
            else:
                frame_result = frame_array[:,:,0]
            # region TODO: REMOVE the if WHEN READY and shift all inside the if to the left
            #if frame_idx == num_frames - 1:

            #Crop outside part
            frame_result, mask = get_mask(frame_result)

            if param['remove_white_lines']:
                # Thresholding to isolate white lines
                _, mask_lines = cv2.threshold(frame_result, 230, 255, cv2.THRESH_BINARY)

                # Perform inpainting on the original image using the mask of the white lines
                frame_result = cv2.inpaint(frame_result, mask_lines, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            if param['second_inpainting']:

                frame_section = frame_result[499:, :]

                frame_section_ch_2 = pixel_array[frame_idx, :, :, 2]
                frame_section_ch_2 = frame_section_ch_2[499:, :]

                most_common_value = np.unique(frame_section_ch_2, return_counts=True)[0][np.argmax(np.unique(frame_section_ch_2, return_counts=True)[1])]

                mask_section = (frame_section_ch_2 < 0.9*most_common_value).astype(np.uint8)

                mask_intersection = mask[499:, :] * mask_section

                if np.max(mask_intersection) > 0:

                    print('Found electrocardiogram inside ROI, suppressing...')
                    # Perform inpainting on the section using the mask of the white lines
                    frame_section_processed = cv2.inpaint(frame_section, mask_intersection, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    frame_result[499:, :] = frame_section_processed

            # Final step before exporting: by adding and removing zeros appropriately, we get square images
            # so as to not deform the images when resizing to the typical neural network resolution
            # Note: only applied here to the .npy images, not to the .png files (at least for the moment).
            if param['get_square']:
                
                height, width = frame_result.shape
                new_size=max(height, width)
                
                array_resized = np.zeros((new_size, new_size), dtype=frame_result.dtype)
                
                x_offset = (new_size - width) // 2
                y_offset = (new_size- height) // 2
                
                array_resized[y_offset:y_offset+height, x_offset:x_offset+width] = frame_result

                # save_image(array_resized, os.path.join(save_path_img, f"{os.path.basename(save_path_base)}--original--{param['name_class']}_{frame_idx_notated}"))
                # save_npy(array_resized, os.path.join(save_path_data, f"{os.path.basename(save_path_base)}--original--{param['name_class']}_{frame_idx_notated}"))

                save_image(array_resized, os.path.join(save_path_img_seq, f"{sequence_name}--original--{class_name}_{frame_idx_notated}"))
                save_npy(array_resized, os.path.join(save_path_data_seq, f"{sequence_name}--original--{class_name}_{frame_idx_notated}"))

            else:

                # save_image(frame_result, os.path.join(save_path_img, f"{os.path.basename(save_path_base)}--original--{param['name_class']}_{frame_idx_notated}"))
                # save_npy(frame_result, os.path.join(save_path_data, f"{os.path.basename(save_path_base)}--original--{param['name_class']}_{frame_idx_notated}"))

                save_image(frame_result, os.path.join(save_path_img_seq, f"{sequence_name}--original--{class_name}_{frame_idx_notated}"))
                save_npy(frame_result, os.path.join(save_path_data_seq, f"{sequence_name}--original--{class_name}_{frame_idx_notated}"))

                    

                    # endregion

    elif pixel_array.ndim == 3:  # [num_frames, height, width] or [height, width, num_channels]

        # TODO: ONLY EXPERIMENTAL (REMOVE WHEN READY)
        if 0:
            # Assume [height, width, num_channels] and num_channels == 3
            height, width, num_channels = pixel_array.shape

            # os.makedirs(save_path_data, exist_ok=True)

            for channel_idx in range(num_channels):

                save_path_img = save_path_base + '--no_processed--images_' + str(channel_idx)
                os.makedirs(save_path_img, exist_ok=True)

                channel_array = pixel_array[:, :, channel_idx]
                save_image(channel_array, os.path.join(save_path_img, f"{os.path.basename(save_path_base)}--no_processed--{param['name_class']}_frame_{channel_idx}"))

                if channel_idx == 0:
                    if 0:
                        save_npy(channel_array, os.path.join(save_path_data,
                                                             f"{os.path.basename(save_path_base)}--original--{param['name_class']}_frame_{channel_idx}"))

    elif pixel_array.ndim == 2:  # [height, width]

        # TODO: ONLY EXPERIMENTAL (REMOVE WHEN READY)
        if 0:

            save_path_img = save_path_base + '--no_processed--images'

            os.makedirs(save_path_img, exist_ok=True)

            # os.makedirs(save_path_data, exist_ok=True)

            save_image(pixel_array, os.path.join(save_path_img, f"{os.path.basename(save_path_base)}--no_processed--{param['name_class']}_frame_0"))

            if 0:
                save_npy(pixel_array, os.path.join(save_path_data, f"{os.path.basename(save_path_base)}--original--{param['name_class']}_frame_0"))


def save_npy(array, filename_base):
    """
        Save a 2D array as a .npy file.

        Parameters:
            array: np.ndarray
                The 2D array to save.
            filename_base: str
                Base filename to save the output files.
        """

    # Save as .npy
    npy_filename = f"{filename_base}.npy"
    np.save(npy_filename, array)



def save_image(array, filename_base):
    """
    Save a 2D array as a .png file with a colorbar.

    Parameters:
        array: np.ndarray
            The 2D array to save.
        filename_base: str
            Base filename to save the output files.
    """

    # Save as .png with colorbar
    plt.figure()
    ax = plt.gca()

    # plt.imshow(array, cmap='gray')
    im = ax.imshow(array, cmap='gray')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # plt.colorbar()
    # plt.axis('off')
    png_filename = f"{filename_base}.png"
    plt.savefig(png_filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def traverse_and_process(param):
    """
    Traverse directories, process DICOM files, and save results.

    Parameters:
        root_dir: str
            The root directory containing folders with DICOM files.
    """

    metadata_info_list = []

    labels_excel = ['Full path', 'Directory', 'Basename', 'Class name', 'Original Shape', 'Dims', 'Frames', 'Height', 'Width', 'Channels', 'FrameTime']

    for dirpath, _, filenames in os.walk(param['root_dir']):
        for filename in filenames:

            # print(dirpath)
            # print(filename)
            print('Processing ' + os.path.join(dirpath, filename))

            frame_time_value = -1

            if not(os.path.isdir(filename)):
                dicom_path = os.path.join(dirpath, filename)

                try:
                    # Load the DICOM file
                    ds = pydicom.dcmread(dicom_path)

                    if hasattr(ds, 'FrameTime'):

                        frame_time_value = float(ds.FrameTime)/1000

                    # Check for a pixel array in the DICOM file
                    if hasattr(ds, 'pixel_array'):
                        # Generate a base path to save the processed files
                        relative_path = os.path.relpath(dirpath, param['root_dir'])

                        class_name = relative_path

                        # save_dir = os.path.join(param['output_dir'], relative_path)
                        # os.makedirs(save_dir, exist_ok=True)

                        file_id = os.path.splitext(filename)[0]
                        # save_path_base = os.path.join(save_dir, file_id)

                        sequence_name = file_id

                        sample_dims = ds.pixel_array.ndim

                        if ds.pixel_array.ndim == 4:

                            sample_frames = ds.pixel_array.shape[0]
                            sample_height = ds.pixel_array.shape[1]
                            sample_width = ds.pixel_array.shape[2]
                            sample_channels = ds.pixel_array.shape[3]

                        if ds.pixel_array.ndim == 3:

                            sample_frames = 'None'
                            sample_height = ds.pixel_array.shape[0]
                            sample_width = ds.pixel_array.shape[1]
                            sample_channels = ds.pixel_array.shape[2]

                        if ds.pixel_array.ndim == 2:

                            sample_frames = 'None'
                            sample_height = ds.pixel_array.shape[0]
                            sample_width = ds.pixel_array.shape[1]
                            sample_channels = 'None'

                        metadata_info_sample = [dicom_path, relative_path, filename, class_name, ds.pixel_array.shape,
                                                sample_dims, sample_frames, sample_height, sample_width, sample_channels,
                                                frame_time_value]


                        if param['image_processing']:
                            print('Perform image processing...')
                            # Process and save pixel array
                            # process_pixel_array(ds, save_path_base, param)
                            process_pixel_array(ds, class_name, sequence_name, param)

                    else:
                        print('No pixel array available!')
                        metadata_info_sample = [dicom_path, relative_path, filename, class_name, 'None',
                                                'None', 'None', 'None', 'None', 'None', frame_time_value]

                except Exception as e:
                    print(f"Error processing file {dicom_path}: {e}")
                    metadata_info_sample = [dicom_path, 'Error', filename, class_name, 'Error',
                                            'Error', 'Error', 'Error', 'Error', 'Error', frame_time_value]

            else:
                print('Folder inside folder!')
                metadata_info_sample = [filename, 'Error', filename, class_name, 'Error',
                                        'Error', 'Error', 'Error', 'Error', 'Error', frame_time_value]

            metadata_info_list.append(metadata_info_sample)

    df_metadata = pd.DataFrame(metadata_info_list, columns=labels_excel)

    # Save MAC matrix as an Excel file
    excel_path = os.path.join(param['output_dir'], "Metadata_databases.xlsx")
    df_metadata.to_excel(excel_path)
    print(f"Metadata of the databases saved to {excel_path}")

# endregion

# Main
traverse_and_process(param)
