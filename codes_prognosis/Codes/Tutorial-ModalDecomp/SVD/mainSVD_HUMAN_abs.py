
# region Import libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import hdf5storage
import torch
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
# endregion

# region Import local libraries
import data_load_HUMAN_abs
import utils
# endregion

# region Setting environment

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('\nDetected device: ')
print(device)

# endregion

def svdtrunc(A, svd_library = 'numpy'):

    if svd_library == 'numpy':

        U, S, V = np.linalg.svd(A, full_matrices = False)


    if svd_library == 'torch':

        U, S, V = torch.linalg.svd(torch.tensor(A).to(device), full_matrices=False)


    if svd_library == 'torch_lower_precision':

        U, S, V = torch.linalg.svd(torch.tensor(A.astype(np.float32)).to(device), full_matrices=False)


    return U, S, V

print('\nSVD Algorithm')
print('\n-----------------------------')

# region Input parameters

param = {}

print('Inputs:' + '\n')

while True:
    param['filetype'] = '.sequences' # input('Select the input file format (.mat, .npy, .csv, .pkl, .h5, .dcm, .folder, .sequences): ')
    if param['filetype'].strip().lower() in ['mat', '.mat', 'npy', '.npy', 'csv', '.csv', 'pkl', '.pkl', 'h5', '.h5', 'dcm', '.dcm', 'folder', '.folder', 'sequences', '.sequences']:
        break
    else: 
        print('\tError: The selected input file format is not supported\n')

while True:
    param['SNAP'] = None # input('Select number of snapshots. Continue with the temporal dimension of the input: ')
    if not param['SNAP']:
        break
    if param['SNAP'].isdigit():
        param['SNAP'] = int(param['SNAP'])
        break
    else:
        print('\tError: Select a valid number (must be integer)\n')

print('Loading data...')

Tensors, list_data_info, _ = data_load_HUMAN_abs.main(param)

print('... data loaded!')

while True:
    param['n_modes'] = None # input('Select number of SVD modes. Continue with 5: ')
    if not param['n_modes']:
        param['n_modes'] = 5
        break
    if param['n_modes'].isdigit():
        param['n_modes'] = int(param['n_modes'])
        break
    else:
        print('\tError: Select a valid number (must be integer)\n')

while True:
    param['svd_library'] = None # input('Select the library to use to execute SVD (numpy, torch, torch_lower_precision) or continue with torch_lower_precision: ')

    if not param['svd_library']:
        param['svd_library'] = 'torch_lower_precision'
        break

    svd_libraries_options = ['numpy', 'torch', 'torch_lower_precision']
    svd_library_selection = [param['svd_library'] == svd_option for svd_option in svd_libraries_options]

    if not True in svd_library_selection:
        print('\tError: Select a valid library (numpy, torch, torch_lower_precision)\n')

    else:
        break

print('\n-----------------------------')
print('SVD summary:')
print(f"Number of modes to retain: {param['n_modes']}")
print("SVD library to use: " + param['svd_library'])

print('\n-----------------------------')
print('Outputs:\n')

param['filen'] = '/home/ander/Escritorio/Workshop2025-2026/Results_step4_abs' # input('Enter folder name to save the outputs or continue with default folder name: ')

if not param['filen']:
    param['filen'] = os.getcwd()

param['export_svd_data'] = 1 # input('Enter 0 to only export singular values, or 1 to also export SVD reconstructions and modes (default: 0): ')

if not param['export_svd_data']:
    param['export_svd_data'] = 0

param['export_V'] = 0 # input('Enter 0 to not export V matrices, or 1 otherwise (default: 0): ')

if not param['export_V']:
    param['export_V'] = 0

while True:
    param['decision_2'] = None # input('Select format of saved files (.mat, .npy) or press enter to not save: ')
    if not param['decision_2'] or param['decision_2'].strip().lower() in ['mat', '.mat', 'npy', '.npy']:
        break
    else:
        print('\tError: Please select a valid output format\n')

param['colormap'] = None # input('Select colormap or press enter to continue with gray: ')

if not param['colormap']:
    param['colormap'] = 'gray'

param['interactive_plot'] = None # input('Introduce whether activating interactive plot (0 or 1) of SVD modes or press enter to not activate: ')

if not(param['interactive_plot']):

    param['interactive_plot'] = 0

else:

    param['interactive_plot'] = int(param['interactive_plot'])


print('')

# endregion

# region Input parameter values for experiments

# region Original images (ONE SEQUENCE) high precision

# filetype = .dcm
# SNAP = 130
# wantedfile = /home/andres/andres/Databases/Data_DICOM/LAX DICOM_fixed.dcm
# filen = /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/LAX_torch_orig
# svd_library = torch

# endregion

# region Original images (ONE SEQUENCE) lower precision

# filetype = .dcm
# SNAP = 130
# wantedfile = /home/andres/andres/Databases/Data_DICOM/LAX DICOM_fixed.dcm
# filen = /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/LAX_torch_lower_precision_orig
# svd_library = torch_lower_precision

# endregion

# region DMD modes (ONE SEQUENCE) high precision

# filetype = folder
# wantedfile = /home/andres/andres/Results/Project_HODMD_IT_Python/LAX DICOM_ILAS 2023_DMD_modes_data
# filen = /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/LAX_torch_DMD
# svd_library = torch

# endregion

# region DMD modes (ONE SEQUENCE) lower precision

# filetype = folder
# wantedfile = /home/andres/andres/Results/Project_HODMD_IT_Python/LAX DICOM_ILAS 2023_DMD_modes_data
# filen = /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/LAX_torch_lower_precision_DMD
# svd_library = torch_lower_precision

# endregion


# region Original + DMD modes high precision

# filetype = folder
# wantedfile = /home/andres/andres/Results/Project_HODMD_IT_Python/LAX DICOM_ILAS 2023_orig_img_DMD_modes_data
# filen = /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/LAX_torch_orig_DMD
# svd_library = torch

# endregion

# region Original + DMD modes lower precision

# filetype = folder
# wantedfile = /home/andres/andres/Results/Project_HODMD_IT_Python/LAX DICOM_ILAS 2023_orig_img_DMD_modes_data
# filen = /home/andres/andres/Results/Project_v0.1-main_ModelFLOWs_app/LAX_torch_lower_precision_orig_DMD
# svd_library = torch_lower_precision

# endregion



# endregion




# region SVD module

# region SVD algorithm

# region Initialization of lists to assign names to files

if isinstance(Tensors, list):

    list_class_names = list_data_info[0]
    list_sequences_names = list_data_info[1]
    list_input_filenames = list_data_info[2]


else:

    Tensors = [[Tensors]]

    list_class_names = ['Class1']
    list_sequences_names = [[os.path.splitext(os.path.basename(param['wantedfile']))[0]]]
    list_input_filenames = [[list_data_info]]

# endregion

# region Make directories to save results

if not os.path.exists(os.path.join(param['filen'], 'svd_modes_images')):
    os.makedirs(os.path.join(param['filen'], 'svd_modes_images'))

if not os.path.exists(os.path.join(param['filen'], 'svd_modes_data')):
    os.makedirs(os.path.join(param['filen'], 'svd_modes_data'))

if not os.path.exists(os.path.join(param['filen'], 'svd_reconst_images')):
    os.makedirs(os.path.join(param['filen'], 'svd_reconst_images'))

if not os.path.exists(os.path.join(param['filen'], 'svd_reconst_data')):
    os.makedirs(os.path.join(param['filen'], 'svd_reconst_data'))

if not os.path.exists(os.path.join(param['filen'], 'List_input_filenames')):
    os.makedirs(os.path.join(param['filen'], 'List_input_filenames'))

if not os.path.exists(os.path.join(param['filen'], 'Singular_values')):
    os.makedirs(os.path.join(param['filen'], 'Singular_values'))

if not os.path.exists(os.path.join(param['filen'], 'RRMSE')):
    os.makedirs(os.path.join(param['filen'], 'RRMSE'))

if not os.path.exists(os.path.join(param['filen'], 'time_SVD')):
    os.makedirs(os.path.join(param['filen'], 'time_SVD'))

os.makedirs(os.path.join(param['filen'], 'V_matrices'), exist_ok=True)
os.makedirs(os.path.join(param['filen'], 'temporal_modes'), exist_ok=True)

# endregion

# region New: export used input arguments

utils.export_input_arguments(param, param['filen'])

# endregion


for idx_class, class_name in enumerate(list_class_names):

    class_name_extracted = os.path.basename(class_name)

    list_sequences_class = list_sequences_names[idx_class]

    if not os.path.exists(os.path.join(param['filen'], 'svd_modes_images', class_name_extracted)):
        os.makedirs(os.path.join(param['filen'], 'svd_modes_images', class_name_extracted))

    if not os.path.exists(os.path.join(param['filen'], 'svd_modes_data', class_name_extracted)):
        os.makedirs(os.path.join(param['filen'], 'svd_modes_data', class_name_extracted))

    if not os.path.exists(os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted)):
        os.makedirs(os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted))

    if not os.path.exists(os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted)):
        os.makedirs(os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted))



    for idx_sequence, sequence_name in enumerate(list_sequences_class):

        Tensor = Tensors[idx_class][idx_sequence]


        # sequence_name_extracted = sequence_name.split('_')[0]

        sequence_name_extracted = os.path.basename(sequence_name).split('--')[0]

        print(sequence_name_extracted)

        if not os.path.exists(os.path.join(param['filen'], 'svd_modes_images', class_name_extracted, sequence_name_extracted + '--svd_modes--images')):
            os.makedirs(os.path.join(param['filen'], 'svd_modes_images', class_name_extracted, sequence_name_extracted + '--svd_modes--images'))

        if not os.path.exists(os.path.join(param['filen'], 'svd_modes_data', class_name_extracted, sequence_name_extracted + '--svd_modes--data')):
            os.makedirs(os.path.join(param['filen'], 'svd_modes_data', class_name_extracted, sequence_name_extracted + '--svd_modes--data'))

        if not os.path.exists(os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted, sequence_name_extracted + '--svd_reconst--images')):
            os.makedirs(os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted, sequence_name_extracted + '--svd_reconst--images'))

        if not os.path.exists(os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted, sequence_name_extracted + '--svd_reconst--data')):
            os.makedirs(os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted, sequence_name_extracted + '--svd_reconst--data'))

        dims = Tensor.ndim
        shape = Tensor.shape

        # timestr = time.strftime("%Y-%m-%d_%H.%M.%S")
        # results_path = param['filen'] + '_' + timestr + '_SVD_solution_modes'


        if dims > 2:
            dims_prod = np.prod(shape[:-1])
            Tensor0 = np.reshape(Tensor, (dims_prod, shape[-1]))

        time_svd = time.time()

        U, S, V = svdtrunc(Tensor0, param['svd_library'])

        time_svd = time.time() - time_svd

        print('Time for SVD: ' + str(time_svd))

        S = np.diag(S)

        U = U[:,0:param['n_modes']]
        S = S[0:param['n_modes'],0:param['n_modes']] # NOTE: S contains the singular values?? YES
        V = V[0:param['n_modes'],:]

        Reconst = (U @ S) @ V

        svd_modes = np.dot(U, S)

        # endregion

        # region RRMSE calculation

        if dims <= 2:
            print(f'SVD modes shape: {svd_modes.shape}')
            print(f'Reconstructed tensor shape: {Reconst.shape}')

            # Usado para verificar. Borrar despues

            Norm2V = np.linalg.norm(Tensor.flatten(), 2)
            if (param['svd_library'] == 'torch_lower_precision') or (param['svd_library'] == 'torch'):
                diff = (Tensor - Reconst.numpy()).flatten()

            if param['svd_library'] == 'numpy':
                diff = (Tensor - Reconst).flatten()

            Norm2diff = np.linalg.norm(diff, ord=2)
            RelativeErrorRMS = Norm2diff/Norm2V
            print('\n' + f'The relative error (RMS) is: ' +  format(RelativeErrorRMS,'e'))

            fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')


            # ax.contourf(svd_modes, cmap=plt.get_cmap(param['colormap']))
            im_fig = ax.imshow(svd_modes, extent=[0, shape[1], shape[0], 0], cmap=plt.get_cmap(param['colormap']), aspect="auto")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im_fig, cax=cax)

            ax.set_title('SVD mode')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.show()

        elif dims > 2:
            newshape = []
            newshape.append(shape[:-1])
            newshape.append(svd_modes.shape[-1])
            newshape = list(newshape[0]) + [newshape[1]]
            svd_modes = np.reshape(svd_modes, np.array(newshape))
            print(f'SVD modes shape: {svd_modes.shape}')
            Reconst = np.reshape(Reconst, shape)
            print(f'Reconstructed tensor: {Reconst.shape}')

            # Usado para verificar. Borrar despues
            if (param['svd_library'] == 'torch_lower_precision') or (param['svd_library'] == 'torch'):
                RRMSE = np.linalg.norm(np.reshape(Tensor-Reconst.numpy(),newshape=(np.size(Tensor),1)),ord=2)/np.linalg.norm(np.reshape(Tensor,newshape=(np.size(Tensor),1)))

            if param['svd_library'] == 'numpy':
                RRMSE = np.linalg.norm(np.reshape(Tensor - Reconst, newshape=(np.size(Tensor), 1)),
                                       ord=2) / np.linalg.norm(np.reshape(Tensor, newshape=(np.size(Tensor), 1)))

            print(f'\nRelative mean square error made in the calculations: {np.round(RRMSE*100, 3)}%\n')

        # endregion

        # region Export results

        if param['decision_2']:

            if param['decision_2'].strip().lower() in ['npy', '.npy']:
                np.save(os.path.join(param['filen'], "Reconstruction_" + sequence_name_extracted + ".npy"), Reconst)
                np.save(os.path.join(param['filen'], "SVD_modes_" + sequence_name_extracted + ".npy"), svd_modes)

            if param['decision_2'].strip().lower() in ['.mat', 'mat']:
                mdic0 = {"Reconst": Reconst}
                mdic1 = {"svd_modes": svd_modes}

                file_mat0 = str(os.path.join(param['filen'], "Reconstruction_" + sequence_name_extracted + ".mat"))
                file_mat1 = str(os.path.join(param['filen'], "svd_modes_" + sequence_name_extracted + ".mat"))

                hdf5storage.savemat(file_mat0, mdic0, appendmat=True, format='7.3')
                hdf5storage.savemat(file_mat1, mdic1, appendmat=True, format='7.3')

        # region New: export metrics

        np.savetxt(os.path.join(param['filen'], 'RRMSE', 'RRMSE_' + param['svd_library'] + "_" + class_name_extracted + '_' + sequence_name_extracted + '.txt'), np.array([RRMSE]))
        np.savetxt(os.path.join(param['filen'], 'time_SVD', 'time_SVD_' + param['svd_library'] + "_" + class_name_extracted + '_' + sequence_name_extracted + '.txt'), np.array([time_svd]))

        # endregion

        # region New: export list of filenames

        df = pd.DataFrame(list_input_filenames[idx_class][idx_sequence], columns = ['Input_filename'])
        df.to_csv(os.path.join(param['filen'], 'List_input_filenames', 'List_input_filenames_' + class_name_extracted + '_' + os.path.basename(list_sequences_names[idx_class][idx_sequence]) + '.csv'))

        np.save(os.path.join(param['filen'], 'List_input_filenames', "List_input_filenames_" + class_name_extracted + '_' + os.path.basename(list_sequences_names[idx_class][idx_sequence]) + ".npy"),
                list_input_filenames[idx_class][idx_sequence])





        # endregion

        # region New: export all figures and corresponding data (SVD modes)


        print('Exporting SVD modes...')

        for mode_num in range(svd_modes.shape[-1]):

            if dims > 3:

                for mode_comp in range(svd_modes.shape[0]):

                    if dims == 4:
                        fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')


                        # ax.contourf(svd_modes[mode_comp, :, :, mode_num], cmap=plt.get_cmap(param['colormap']))


                        im_fig = ax.imshow(svd_modes[mode_comp, :, :, mode_num], extent=[0, shape[2], shape[1], 0], cmap=plt.get_cmap(param['colormap']), aspect="auto")

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im_fig, cax=cax)

                        ax.set_title(f'SVD modes - Component {mode_comp + 1} - Mode Number {mode_num + 1}')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')



                        mode_img_filename = os.path.join(param['filen'], 'svd_modes_images', class_name_extracted, sequence_name_extracted + '--svd_modes--images',
                                                         sequence_name_extracted + '--svd_mode--comp_' + str(mode_comp + 1).zfill(len(str(svd_modes.shape[0])) + 1)
                                                         + '_' + class_name_extracted + '_number_' + str(mode_num + 1).zfill(len(str(svd_modes.shape[-1])) + 1) +
                                                         '.png')

                        mode_data_filename = os.path.join(param['filen'], 'svd_modes_data', class_name_extracted, sequence_name_extracted + '--svd_modes--data',
                                                         sequence_name_extracted + '--svd_mode--comp_' + str(mode_comp + 1).zfill(len(str(svd_modes.shape[0])) + 1)
                                                         + '_' + class_name_extracted + '_number_' + str(mode_num + 1).zfill(len(str(svd_modes.shape[-1])) + 1) +
                                                         '.npy')

                        if param['export_svd_data']:
                            plt.savefig(mode_img_filename)

                        plt.close()

                        if param['export_svd_data']:
                            np.save(mode_data_filename, svd_modes[mode_comp, :, :, mode_num])




                    elif dims == 5:
                        fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode XY plane')


                        # ax.contourf(svd_modes[mode_comp, :, :, 0, mode_num], cmap=plt.get_cmap(param['colormap']))

                        im_fig = ax.imshow(svd_modes[mode_comp, :, :, 0, mode_num], extent=[0, shape[2], shape[1], 0], cmap=plt.get_cmap(param['colormap']), aspect="auto")


                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im_fig, cax=cax)


                        ax.set_title(f'SVD modes XY plane - Component {mode_comp + 1} - Mode Number {mode_num + 1}')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')




                        mode_img_filename = os.path.join(param['filen'], 'svd_modes_images', class_name_extracted, sequence_name_extracted + '--svd_modes--images',
                                                         sequence_name_extracted + '--svd_mode--XY plane comp_' + str(mode_comp + 1).zfill(len(str(svd_modes.shape[0])) + 1)
                                                         + '_' + class_name_extracted + '_number_' + str(mode_num + 1).zfill(len(str(svd_modes.shape[-1])) + 1) +
                                                         '.png')


                        mode_data_filename = os.path.join(param['filen'], 'svd_modes_data', class_name_extracted, sequence_name_extracted + '--svd_modes--data',
                                                         sequence_name_extracted + '--svd_mode--XY plane comp_' + str(mode_comp + 1).zfill(len(str(svd_modes.shape[0])) + 1)
                                                         + '_' + class_name_extracted + '_number_' + str(mode_num + 1).zfill(len(str(svd_modes.shape[-1])) + 1) +
                                                         '.npy')

                        if param['export_svd_data']:
                            plt.savefig(mode_img_filename)

                        plt.close()

                        if param['export_svd_data']:
                            np.save(mode_data_filename, svd_modes[mode_comp, :, :, 0, mode_num])


            else:

                fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')

                # ax.contourf(svd_modes[:, :, mode_num], cmap=plt.get_cmap(param['colormap']))
                im_fig = ax.imshow(svd_modes[:, :, mode_num], extent = [0, shape[1], shape[0], 0], cmap = plt.get_cmap(param['colormap']), aspect = "auto")

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_fig, cax=cax)

                ax.set_title(f'SVD modes - Mode Number {mode_num + 1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')



                mode_img_filename = os.path.join(param['filen'], 'svd_modes_images', class_name_extracted,
                                                 sequence_name_extracted + '--svd_modes--images',
                                                 sequence_name_extracted + '--svd_mode--' +
                                                 class_name_extracted + '_number_' + str(mode_num + 1).zfill(
                                                 len(str(svd_modes.shape[-1])) + 1) +
                                                 '.png')

                mode_data_filename = os.path.join(param['filen'], 'svd_modes_data', class_name_extracted,
                                                  sequence_name_extracted + '--svd_modes--data',
                                                  sequence_name_extracted + '--svd_mode--' +
                                                  class_name_extracted + '_number_' + str(mode_num + 1).zfill(
                                                  len(str(svd_modes.shape[-1])) + 1) +
                                                  '.npy')

                if param['export_svd_data']:
                    plt.savefig(mode_img_filename)

                plt.close()

                if param['export_svd_data']:
                    np.save(mode_data_filename, svd_modes[:, :, mode_num])


        print('...SVD modes exported!')

        # endregion



        # region New export all figures and corresponding data (reconstructed images)

        print('Exporting reconstructed images...')


        for n_frame in range(Reconst.shape[-1]):

            if dims > 3:

                for input_comp in range(Reconst.shape[0]):

                    if dims == 4:
                        fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Reconstructed images')

                        # ax.contourf(Reconst[input_comp, :, :, n_frame], cmap=plt.get_cmap(param['colormap']))
                        im_fig = ax.imshow(Reconst[input_comp, :, :, n_frame], extent=[0, shape[2], shape[1], 0], cmap=plt.get_cmap(param['colormap']), aspect="auto")

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im_fig, cax=cax)




                        if len(list_input_filenames[idx_class][idx_sequence]) > 1:
                            title_figure = "Reconstructed images - Component " + str(input_comp + 1) + " - Frame " + str(n_frame + 1) + \
                                           "\n" + os.path.join(os.path.basename(os.path.split(list_input_filenames[idx_class][idx_sequence][n_frame])[0]),
                                                               os.path.basename(list_input_filenames[idx_class][idx_sequence][n_frame]))

                        else:
                            title_figure = "Reconstructed images - Component " + str(input_comp + 1) + " - Frame " + str(n_frame + 1)


                        ax.set_title(title_figure)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')


                        file_reconstructed_image = os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted, sequence_name_extracted + '--svd_reconst--images',
                                                    sequence_name_extracted + '--svd_reconst--comp_' + str(input_comp + 1).zfill(len(str(Reconst.shape[0])) + 1)
                                                    + '_' + class_name_extracted + "_frame_" + str(n_frame + 1).zfill(len(str(Reconst.shape[-1])) + 1) +
                                                    '.png')

                        if param['export_svd_data']:
                            plt.savefig(file_reconstructed_image)
                        plt.close()

                        file_reconstructed_data = os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted, sequence_name_extracted + '--svd_reconst--data',
                                                               sequence_name_extracted + '--svd_reconst--comp_' + str(
                                                                   input_comp + 1).zfill(len(str(Reconst.shape[0])) + 1)
                                                               + '_' + class_name_extracted + "_frame_" + str(
                                                                   n_frame + 1).zfill(len(str(Reconst.shape[-1])) + 1) +
                                                               '.npy'
                                                               )

                        if param['export_svd_data']:
                            np.save(file_reconstructed_data, Reconst[input_comp, :, :, n_frame])

                    elif dims == 5:
                        fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Reconstructed images XY plane')

                        # ax.contourf(Reconst[input_comp, :, :, 0, n_frame], cmap=plt.get_cmap(param['colormap']))
                        im_fig = ax.imshow(Reconst[input_comp, :, :, 0, n_frame], extent = [0, shape[2], shape[1], 0], cmap = plt.get_cmap(param['colormap']), aspect = "auto")

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im_fig, cax=cax)

                        if len(list_input_filenames[idx_class][idx_sequence]) > 1:
                            title_figure = "Reconstructed images XY plane - Component " + str(input_comp + 1) + " - Frame " + str(n_frame + 1) + \
                                           "\n" + os.path.join(os.path.basename(os.path.split(list_input_filenames[idx_class][idx_sequence][n_frame])[0]),
                                                               os.path.basename(list_input_filenames[idx_class][idx_sequence][n_frame]))

                        else:
                            title_figure = "Reconstructed images XY plane - Component " + str(input_comp + 1) + " - Frame " + str(n_frame + 1)

                        ax.set_title(title_figure)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        #


                        file_reconstructed_image = os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted, sequence_name_extracted + '--svd_reconst--images',
                                                   sequence_name_extracted + '--svd_reconst--plane_comp_' + str(input_comp + 1).zfill(len(str(Reconst.shape[0])) + 1)
                                                   + '_' + class_name_extracted + "_frame_" + str(n_frame + 1).zfill(len(str(Reconst.shape[-1])) + 1) +
                                                   '.png')

                        if param['export_svd_data']:
                            plt.savefig(file_reconstructed_image)

                        plt.close()

                        file_reconstructed_data = os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted, sequence_name_extracted + '--svd_reconst--data',
                                                  sequence_name_extracted + '--svd_reconst--plane_comp_' + str(
                                                  input_comp + 1).zfill(len(str(Reconst.shape[0])) + 1)
                                                  + '_' + class_name_extracted + "_frame_" + str(
                                                  n_frame + 1).zfill(len(str(Reconst.shape[-1])) + 1) +
                                                  '.npy'
                                                  )

                        if param['export_svd_data']:
                            np.save(file_reconstructed_data, Reconst[input_comp, :, :, 0, n_frame])

            else:

                fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - Reconstructed images')
                # ax.contourf(Reconst[:, :, n_frame], cmap=plt.get_cmap(param['colormap']))

                im_fig = ax.imshow(Reconst[:, :, n_frame], extent = [0, shape[1], shape[0], 0], cmap = plt.get_cmap(param['colormap']), aspect = "auto")


                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im_fig, cax=cax)


                if len(list_input_filenames[idx_class][idx_sequence]) > 1:

                    title_figure = "Reconstructed images - Frame " + str(n_frame + 1) + \
                                   "\n" + os.path.join(os.path.basename(os.path.split(list_input_filenames[idx_class][idx_sequence][n_frame])[0]),
                                                       os.path.basename(list_input_filenames[idx_class][idx_sequence][n_frame]))

                else:

                    title_figure = "Reconstructed images - Frame " + str(n_frame + 1)

                ax.set_title(title_figure)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')

                file_reconstructed_image = os.path.join(param['filen'], 'svd_reconst_images', class_name_extracted, sequence_name_extracted + '--svd_reconst--images',
                                           sequence_name_extracted + '--svd_reconst--' + '_' + class_name_extracted + "_frame_"
                                           + str(n_frame + 1).zfill(len(str(Reconst.shape[-1])) + 1) +
                                           '.png')

                if param['export_svd_data']:
                    plt.savefig(file_reconstructed_image)

                plt.close()

                file_reconstructed_data = os.path.join(param['filen'], 'svd_reconst_data', class_name_extracted, sequence_name_extracted + '--svd_reconst--data',
                                           sequence_name_extracted + '--svd_reconst--' + class_name_extracted + "_frame_"
                                           + str(n_frame + 1).zfill(len(str(Reconst.shape[-1])) + 1) +
                                           '.npy')

                if param['export_svd_data']:
                    np.save(file_reconstructed_data, Reconst[:, :, n_frame])

        print('...reconstructed images exported!')


        # endregion

        # region Infinite loops to plot modes

        if param['interactive_plot']:


            print('Please CLOSE all figures to continue the run\n')


            while True:
                while True:
                    ModeNum = input(f'Introduce the mode number to plot (default mode 1). Maximum number of modes is {svd_modes.shape[-1]}: ')
                    if not ModeNum:
                        ModeNum = 0
                        break
                    elif ModeNum.isdigit():
                        if int(ModeNum) <= svd_modes.shape[-1]:
                            ModeNum = int(ModeNum)-1
                            break
                        else:
                            print('\tError: Selected value is out of bounds\n')
                    else:
                        print('\tError: Select a valid number format (must be integer)\n')
                if dims > 3:
                    while True:
                        ModComp = input(f'Introduce the component to plot (default component 1). Maximum number of components is {svd_modes.shape[0]}: ')
                        if not ModComp:
                            ModComp = 0
                            break
                        elif ModComp.isdigit():
                            if int(ModComp) <= svd_modes.shape[0]:
                                ModComp = int(ModComp)-1
                                break
                            else:
                                print('\tError: Selected value is out of bounds\n')
                        else:
                            print('\tError: Select a valid number format (must be integer)\n')

                elif dims==3:
                    fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')
                    # ax.contourf(svd_modes[:,:,ModeNum], cmap=plt.get_cmap(param['colormap']))

                    im_fig = ax.imshow(svd_modes[:, :, ModeNum], extent = [0, shape[1], shape[0], 0], cmap = plt.get_cmap(param['colormap']), aspect = "auto")

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im_fig, cax=cax)

                    ax.set_title(f'SVD modes - Mode Number {ModeNum+1}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    plt.show()

                if dims==4:
                    fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode')

                    # ax.contourf(svd_modes[ModComp,:,:,ModeNum], cmap=plt.get_cmap(param['colormap']))

                    im_fig = ax.imshow(svd_modes[ModComp, :, :, ModeNum], extent = [0, shape[2], shape[1], 0], cmap = plt.get_cmap(param['colormap']), aspect = "auto")

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im_fig, cax=cax)

                    ax.set_title(f'SVD modes - Component {ModComp+1} - Mode Number {ModeNum+1}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    plt.show()

                elif dims==5:
                    fig, ax = plt.subplots(num=f'CLOSE TO CONTINUE RUN - SVD mode XY plane')
                    # ax.contourf(svd_modes[ModComp,:,:,0,ModeNum], cmap=plt.get_cmap(param['colormap']))

                    im_fig = ax.imshow(svd_modes[ModComp, :, :, 0, ModeNum], extent = [0, shape[2], shape[1], 0], cmap = plt.get_cmap(param['colormap']), aspect = "auto")

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im_fig, cax=cax)

                    ax.set_title(f'SVD modes XY plane - Component {ModComp+1} - Mode Number {ModeNum+1}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    plt.show()

                while True:
                    Resp = input('Do you want to plot another mode? Yes or No (y/n). Press enter to continue with No: ')
                    if not Resp or Resp.strip().lower() in ['n', 'no']:
                        Resp = 0
                        break
                    elif Resp.strip().lower() in ['y', 'yes']:
                        Resp = 1
                        break
                    else:
                        print('\tError: Select yes or no (y/n)\n')

                if Resp == 0:
                    break

                if Resp == 1:
                    continue

        # endregion



        # region New: plot SVD modes vs Singular values
        print('Exporting Singular Values...')
        fig, ax = plt.subplots()
        markers = ['gx']
        labels = ["Singular values"]

        ax.plot(np.diag(S) / S[0,0], markers[0])

        ax.set_yscale('log') # Logarithmic scale in y axis
        ax.set_xlabel('SVD modes')
        ax.legend(labels, loc='center right', bbox_to_anchor=(1, 0.5))
        ax.set_title('SVD modes vs. Singular Values')

        if param['export_svd_data']:
            plt.savefig(os.path.join(param['filen'], 'Singular_values', 'SVD modes - Singular Values_' + class_name_extracted + '_' + sequence_name_extracted + '.png'), bbox_inches='tight')

        plt.close()

        # plt.show()



        df = pd.DataFrame(np.diag(S) / S[0,0], columns = ['Singular_value'])
        df.to_csv(os.path.join(param['filen'], 'Singular_values', 'Singular_values_normalized_' + class_name_extracted + '_' + sequence_name_extracted + '.csv'))

        np.save(os.path.join(param['filen'], 'Singular_values', 'Singular_values_normalized_' + class_name_extracted + '_' + sequence_name_extracted + '.npy'), np.diag(S) / S[0,0])

        df1 = pd.DataFrame(np.diag(S), columns=['Singular_value'])
        df1.to_csv(os.path.join(param['filen'], 'Singular_values', 'Singular_values_no-normalized_' + class_name_extracted + '_' + sequence_name_extracted + '.csv'))

        np.save(os.path.join(param['filen'], 'Singular_values', 'Singular_values_no-normalized_' + class_name_extracted + '_' + sequence_name_extracted + '.npy'), np.diag(S))

        # endregion



        # region New: export V matrices
        if param['export_V']:

            print('Exporting Temporal Modes...')
            np.save(os.path.join(param['filen'], 'V_matrices', 'V_' + class_name_extracted + '_' + sequence_name_extracted + '.npy'), V.numpy())

            np.save(os.path.join(param['filen'], 'temporal_modes', 'temporal_mode_' + class_name_extracted + '_' + sequence_name_extracted + '.npy'), S @ V.numpy())

        # endregion


        # endregion

        # endregion




