# region Import other required libraries
# !pip install h5py
# !pip install -qq -U tensorflow-addons
# !pip install scikit-plot

# !pip install tensorflow-gpu

# endregion

# region Import libraries


from tensorflow import keras
import tensorflow as tf

from tensorflow.keras import layers

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 20})

import numpy as np
import os
import time

import numpy.random as rng

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# endregion

np.random.seed(0)


# region Import local libraries


from Training import model as model

from Utils import utils as utils
from Utils import datasets as my_datasets
from Testing import results_functions as results_functions


# endregion


# region Input parameters

param = {}

print('Data parameters...')


# region Param path to training database


param['training_database_path'] = ['/home/usuario/andres/Databases/My_databases_ecos/ecos_orig/ecos_orig_Training']


# endregion


# region Param path to validation database


param['validation_database_path'] = ['/home/usuario/andres/Databases/My_databases_ecos/ecos_orig/ecos_orig_Validation']


# endregion


# region Model parameters


# region Param size of input data to the model


param['target_size'] = (256, 256)


# endregion


# region Specific parameters to the ViT


param['num_channels'] = 1
param['patch_size'] = 32
param['n_blocks'] = 8
param['n_heads'] = 4

param['projection_dim'] = 64
param['transformer_units'] = [128, 64]
param['mlp_head_units'] = [512, 256] # [2048, 1024]


# endregion


# endregion


# region Training parameters


# region Param batch size


param['batch_size'] = 64


# endregion


# region Callback parameters


# region Param Path to save model


param['model_save_path'] = '/home/usuario/andres/Models/DigitHEART/VIT/model_vit__modelflows-app'


# endregion


# endregion


# region Optimizer parameters

# region Param learning rate


param['lr'] = 0.001


# endregion

# region Param weight decay (case ViT for small datasets in Keras)


param['weight_decay'] = 0.0001


# endregion

# endregion


# region Fitting process parameters


# region Param Number of epochs


param['n_epochs'] = 200


# endregion


# region Param steps per epoch


param['steps_per_epoch'] = 500


# endregion


# region Param validation steps


param['validation_steps'] = 300


# endregion


# endregion

# endregion

print('Introduced all training parameters!')

print('Testing parameters: ')


# region Param path to pretrained model


param['model_path_test'] = '/home/usuario/andres/Models/DigitHEART/VIT/model_vit__modelflows-app__2025-03-27_13-08/pre-trained-vit.h5'


# endregion

# region Param path to test data


param['test_database_path'] = ['/home/usuario/andres/Databases/My_databases_ecos/ecos_orig/ecos_orig_Test']


# endregion


# region Device selection:
# -1 if 'cpu', >=0 for 'gpu' selection
param['device'] = 1


# endregion


# region Param size of the input data to the model for prediction


param['test_target_size'] = param['target_size']


# endregion


# region Param path where results will be saved


param['test_results_path'] = '/home/usuario/andres/Results/VIT/results_vit_modelflows-app'


# endregion

param['sequence_classification'] = 1

print('Introduced all testing parameters!')

# endregion


# region Device Initialization/check


device_name_tf = tf.test.gpu_device_name()
print(device_name_tf)


os.environ["CUDA_VISIBLE_DEVICES"]=str(param['device'])
if param['device'] < 0:
    print('CPU selected')
else:
    print('GPU '  + str(param['device']) + ' selected')


# endregion


# region Test part

print('Selected input parameters (test part): \n')

for param_name, param_value in param.items():
  print('- ' + param_name + ': ' + str(param_value))


# region Load model

trained_class_names = np.load(os.path.join(os.path.split(param['model_path_test'])[0], 'class_names.npy')).tolist()


print('Loading model...')


data_augmentation_model_loaded = keras.Sequential(
[
  layers.Normalization(),
  layers.Resizing(param['test_target_size'][0], param['test_target_size'][1]),
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(factor=0.02),
  layers.RandomZoom(height_factor=0.2, width_factor=0.2),
],
name="data_augmentation",
)

loaded_model = model.create_vit_classifier(data_augmentation_model_loaded, input_shape = (param['test_target_size'][0], param['test_target_size'][1], param['num_channels']), image_size = max(param['test_target_size'][0], param['test_target_size'][1]), patch_size=param['patch_size'], transformer_layers = param['n_blocks'], num_heads = param['n_heads'], projection_dim = param['projection_dim'], transformer_units = param['transformer_units'], mlp_head_units = param['mlp_head_units'], num_classes = len(trained_class_names), vanilla=False)
loaded_model.load_weights(param['model_path_test'])

loaded_model.summary()

num_channels_test_data = loaded_model.layers[0].input_shape[0][-1]

print('... model loaded!')

# endregion

# region Load test database and prediction


train_label_encoder = preprocessing.LabelEncoder()
train_label_encoder.fit_transform(trained_class_names)


start_loading_test_data_time = time.time()

# Load test data

x_test, y_test, x_test_filenames, y_test_class_names, test_class_names, test_label_encoder = my_datasets.load_dataset_organized_sequences_different_sources(
param['test_database_path'], img_size=param['test_target_size'], label_encoder=train_label_encoder)
loading_test_data_time = time.time() - start_loading_test_data_time
average_loading_test_data_time = loading_test_data_time / x_test.shape[0]


# Initialization
prediction_time = -1
average_prediction_per_sample = -1
prediction_time_individual = -1
average_prediction_per_sample_individual = -1


# Perform prediction
start_prediction_time = time.time()

test_predictions = loaded_model.predict(x_test, verbose = 0)

prediction_time = time.time() - start_prediction_time

average_prediction_per_sample = prediction_time/x_test.shape[0]

# Get the prediction confidences and the predicted classes
Pr_confidences = np.max(test_predictions, axis = 1)

# Get predicted classes
Prid_class = np.argmax(test_predictions, axis = 1)
Pred_class_names = train_label_encoder.inverse_transform(Prid_class)


# endregion


# region Results and statistics

# region Initialization


# Set the path to save results
timestamp_results_path = time.strftime("%Y-%m-%d_%H-%M")
test_results_fullpath = os.path.join(param['test_results_path'] + timestamp_results_path)

if not(os.path.exists(test_results_fullpath)):
  os.makedirs(test_results_fullpath)

# Save input parameters used
utils.export_input_arguments(param, test_results_fullpath)


# endregion


# region Prediction statistics

# Save prediction statistics
pred_stats_columns = ['Sample_name', 'GT_class_name', 'Predicted_class_name', 'GT_label', 'Predicted_label']
for trained_class_name in trained_class_names:
  pred_stats_columns.append('Prediction_class_' + trained_class_name)

pred_stats_df = pd.DataFrame(columns = pred_stats_columns)

for idx_test_sample, prid_class_sample in enumerate(Prid_class):
  sample_pred_stats = [x_test_filenames[idx_test_sample], y_test_class_names[idx_test_sample], Pred_class_names[idx_test_sample], y_test[idx_test_sample], Prid_class[idx_test_sample]]
  for idx_trained_class_name, trained_class_name in enumerate(trained_class_names):
    sample_pred_stats.append(test_predictions[idx_test_sample, idx_trained_class_name])

  pred_stats_df.loc[idx_test_sample] = sample_pred_stats

pred_stats_df.index = [os.path.basename(x_test_filename) for x_test_filename in x_test_filenames]


pred_stats_df.to_csv(os.path.join(test_results_fullpath, 'test_prediction_stats.csv'))

# endregion


# region Test times

# Save loading test data times
load_times_df = pd.DataFrame(data={'total_loading_test_data_time': loading_test_data_time, 'average_loading_test_data_time': average_loading_test_data_time, 'num_test_samples': x_test.shape[0]}, index = [0])
load_times_df.to_csv(os.path.join(test_results_fullpath, 'test_data_loading_times.csv'))


# Save prediction times
pred_times_df = pd.DataFrame(data={'total_prediction_time': prediction_time, 'average_prediction_per_sample': average_prediction_per_sample, 'total_prediction_time_individual': prediction_time_individual, 'average_prediction_per_sample_individual': average_prediction_per_sample_individual, 'num_test_samples': x_test.shape[0]}, index = [0])
pred_times_df.to_csv(os.path.join(test_results_fullpath, 'prediction_times.csv'))


# endregion


# region Classification report

# Save classification report
test_classification_report = classification_report(y_test, Prid_class, target_names=trained_class_names, output_dict = True)
print(classification_report(y_test, Prid_class, target_names=trained_class_names))
results_functions.export_test_classification_report_csv(test_classification_report, test_results_fullpath, trained_class_names)


# endregion


# region Confusion matrix

# Save confusion matrix
cm = confusion_matrix(y_test, Prid_class)
print(accuracy_score(y_test, Prid_class))
df_cm, cm_fig = results_functions.print_confusion_matrix(cm, trained_class_names)
df_cm.to_csv(os.path.join(test_results_fullpath, 'confusion_matrix.csv'))
cm_fig.savefig(os.path.join(test_results_fullpath, 'confusion_matrix.png'))

# endregion


# endregion


# region Sequence classification


# region Obtain pathology prediction for each sequence

if param['sequence_classification']:

  # region: Prediction statistics for sequence classification Initialization
  pred_stats_seq_columns = ['Seq_name', 'GT_name', 'Pred_name_mean', 'Pred_name_max', 'GT_label', 'Pred_label_mean', 'Pred_label_max', 'Frames_true', 'Frames_mean', 'Frames_max', 'Conf_mean_true', 'Conf_mean', 'Conf_max_true', 'Conf_max']

  for trained_class_name in trained_class_names:
      pred_stats_seq_columns.append('Perc__' + trained_class_name)

  for trained_class_name in trained_class_names:
      pred_stats_seq_columns.append('Frames_predicted__' + trained_class_name)

  for trained_class_name in trained_class_names:
      pred_stats_seq_columns.append('Conf_mean__' + trained_class_name)

  for trained_class_name in trained_class_names:
      pred_stats_seq_columns.append('Conf_max__' + trained_class_name)

  pred_stats_seq_df = pd.DataFrame(columns=pred_stats_seq_columns)

  # endregion


  print('Performing per-sequence classification...')

  x_test_sequence_filenames = []

  Prid_class_mean = -np.ones((Prid_class.shape), dtype = int)
  Prid_class_max = -np.ones((Prid_class.shape), dtype = int)

for idx_test_filename, x_test_filename in enumerate(x_test_filenames):

  all_data_sequence = x_test_filenames[idx_test_filename].split('/')[-2].split('--')[0:-1] + [x_test_filenames[idx_test_filename].split('/')[-3]]

  final_test_sequence_filename = '--'.join(all_data_sequence)

  x_test_sequence_filenames = x_test_sequence_filenames + [final_test_sequence_filename]

test_sequence_filenames = list(set(x_test_sequence_filenames))
test_sequence_filenames.sort()

# region Ground-truth of each sequence

y_seq_test_class_names = [sequence_name.split('--')[-1] for sequence_name in test_sequence_filenames]
y_seq_test = train_label_encoder.transform(y_seq_test_class_names)

# endregion

Prid_sequence_class_mean = -np.ones(len(test_sequence_filenames))
Prid_sequence_class_max = -np.ones(len(test_sequence_filenames))

for idx_test_sequence_filename, test_sequence_filename in enumerate(test_sequence_filenames):

  all_data_filenames_sequence_idx = np.array([i for i, x in enumerate(x_test_sequence_filenames) if x == test_sequence_filename])

  confidences_seq_mean = np.mean(test_predictions[all_data_filenames_sequence_idx, :], axis=0)
  confidences_seq_max = np.max(test_predictions[all_data_filenames_sequence_idx, :], axis=0)

  label_predicted_sequence_mean = np.argmax(confidences_seq_mean)
  label_predicted_sequence_max = np.argmax(confidences_seq_max)

  confidence_predicted_seq_mean = confidences_seq_mean[label_predicted_sequence_mean]
  confidence_predicted_seq_max = confidences_seq_max[label_predicted_sequence_max]

  y_test_in_seq = y_test[all_data_filenames_sequence_idx]
  Prid_class_in_seq = Prid_class[all_data_filenames_sequence_idx]

  Prid_class_in_seq_stats = np.unique(Prid_class_in_seq, return_counts=True)

  Prid_class_in_seq_all_frames = np.zeros(len(trained_class_names), dtype=int)
  Prid_class_in_seq_all_frames[Prid_class_in_seq_stats[0]] = Prid_class_in_seq_stats[1]

  Prid_class_in_seq_perc = Prid_class_in_seq_stats[1] / len(y_test_in_seq)

  Prid_class_in_seq_all_perc = np.zeros(len(trained_class_names))
  Prid_class_in_seq_all_perc[Prid_class_in_seq_stats[0]] = Prid_class_in_seq_perc

  Prid_class_mean[all_data_filenames_sequence_idx] = label_predicted_sequence_mean
  Prid_class_max[all_data_filenames_sequence_idx] = label_predicted_sequence_max

  Prid_sequence_class_mean[idx_test_sequence_filename] = label_predicted_sequence_mean
  Prid_sequence_class_max[idx_test_sequence_filename] = label_predicted_sequence_max

  seq_pred_stats = [test_sequence_filename,
                    y_seq_test_class_names[idx_test_sequence_filename],
                    trained_class_names[label_predicted_sequence_mean],
                    trained_class_names[label_predicted_sequence_max],
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

  for idx_trained_class_name, trained_class_name in enumerate(trained_class_names):
    seq_pred_stats.append(Prid_class_in_seq_all_perc[idx_trained_class_name])

  for idx_trained_class_name, trained_class_name in enumerate(trained_class_names):
    seq_pred_stats.append(Prid_class_in_seq_all_frames[idx_trained_class_name])

  for idx_trained_class_name, trained_class_name in enumerate(trained_class_names):
    seq_pred_stats.append(confidences_seq_mean[idx_trained_class_name])

  for idx_trained_class_name, trained_class_name in enumerate(trained_class_names):
    seq_pred_stats.append(confidences_seq_max[idx_trained_class_name])

  pred_stats_seq_df.loc[idx_test_sequence_filename] = seq_pred_stats

pred_stats_seq_df.to_csv(os.path.join(test_results_fullpath, 'test_seq_prediction_stats.csv'))
print('Done!')

Prid_class_names_mean = train_label_encoder.inverse_transform(Prid_class_mean)
Prid_class_names_max = train_label_encoder.inverse_transform(Prid_class_max)

# endregion


# region Plots about per-sequence prediction
if param['sequence_classification']:

    test_seq_results_path = os.path.join(test_results_fullpath, 'seq_analysis')

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
                 label="Prid_seq_mean: " + trained_class_names[label_predicted_sequence_mean],
                 linewidth = 1,
                 linestyle = 'None')

        # Line representing the predicted label in the sequence (max; cyan [0.5, 0.8, 0.9]) CORRECTED: BLACK
        plt.plot(range(2, len(all_data_filenames_sequence_idx), 6), label_predicted_sequence_max*np.ones(len(range(2, len(all_data_filenames_sequence_idx), 6)), dtype = int),
                 color=[0, 0, 0],
                 marker= "^",
                 label="Prid_seq_max: " + trained_class_names[label_predicted_sequence_max],
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

        for idx_trained_class_name, trained_class_name in enumerate(trained_class_names):


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
                     label = "Conf_mean_predicted: " + trained_class_names[label_predicted_sequence_mean], linewidth = 2, linestyle = 'dashdot')

            # Line representing the confidence of predicted class by max (cyan [0.5, 0.8, 0.9]) (CORRECTED: RED)
            plt.plot(range(len(all_data_filenames_sequence_idx)), confidence_predicted_seq_max*np.ones(len(all_data_filenames_sequence_idx), dtype = int), color = [1, 0, 0],
                     label = "Conf_max_predicted: " + trained_class_names[label_predicted_sequence_max], linewidth = 2, linestyle = 'dashed')

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
                         label="Conf_mean_predicted: " + trained_class_names[label_predicted_sequence_mean],
                         linewidth=2,
                         linestyle='dashdot')

                # Line representing the confidence of predicted class by max (cyan [0.5, 0.8, 0.9]) (CORRECTED: RED)
                plt.plot(range(len(all_data_filenames_sequence_idx)),
                         confidence_predicted_seq_max * np.ones(len(all_data_filenames_sequence_idx), dtype=int),
                         color=[1, 0, 0],
                         label="Conf_max_predicted: " + trained_class_names[label_predicted_sequence_max], linewidth=2,
                         linestyle='dashed')

                plt.legend()

                plt.savefig(os.path.join(test_seq_results_path,
                                         'Conf_per-image_GT_vs_' + trained_class_name + "_" + test_sequence_filename + '.png'))

                plt.close()


# endregion


# region Classification reports

if param['sequence_classification']:

  # Save classification reports

  print('Results with mean: ')
  test_classification_report_mean = classification_report(y_test, Prid_class_mean, target_names=trained_class_names, output_dict = True)
  print(classification_report(y_test, Prid_class_mean, target_names=trained_class_names))
  results_functions.export_test_classification_report_csv(test_classification_report_mean, test_results_fullpath, trained_class_names, 'mean')


  print('Results with mean (per-sequence): ')
  test_classification_report_mean_seq = classification_report(y_seq_test, Prid_sequence_class_mean, target_names=trained_class_names, output_dict = True)
  print(classification_report(y_seq_test, Prid_sequence_class_mean, target_names=trained_class_names))
  results_functions.export_test_classification_report_csv(test_classification_report_mean_seq, test_results_fullpath, trained_class_names, 'mean_seq')


  print('Results with max: ')
  test_classification_report_max = classification_report(y_test, Prid_class_max, target_names=trained_class_names, output_dict = True)
  print(classification_report(y_test, Prid_class_max, target_names=trained_class_names))
  results_functions.export_test_classification_report_csv(test_classification_report_max, test_results_fullpath, trained_class_names, 'max')


  print('Results with max (per-sequence): ')
  test_classification_report_max_seq = classification_report(y_seq_test, Prid_sequence_class_max, target_names=trained_class_names, output_dict = True)
  print(classification_report(y_seq_test, Prid_sequence_class_max, target_names=trained_class_names))
  results_functions.export_test_classification_report_csv(test_classification_report_max_seq, test_results_fullpath, trained_class_names, 'max_seq')

# endregion

# region Confusion matrixes

if param['sequence_classification']:

  # Save confusion matrixes
  print('Results with mean: ')
  cm_mean = confusion_matrix(y_test, Prid_class_mean)
  print(accuracy_score(y_test, Prid_class_mean))
  df_cm_mean, cm_fig_mean = results_functions.print_confusion_matrix(cm_mean, trained_class_names)
  df_cm_mean.to_csv(os.path.join(test_results_fullpath, 'confusion_matrix_mean.csv'))
  cm_fig_mean.savefig(os.path.join(test_results_fullpath, 'confusion_matrix_mean.png'))



  print('Results with mean (per-sequence): ')
  cm_mean_seq = confusion_matrix(y_seq_test, Prid_sequence_class_mean)
  print(accuracy_score(y_seq_test, Prid_sequence_class_mean))
  df_cm_mean_seq, cm_fig_mean_seq = results_functions.print_confusion_matrix(cm_mean_seq, trained_class_names)
  df_cm_mean_seq.to_csv(os.path.join(test_results_fullpath, 'confusion_matrix_mean_seq.csv'))
  cm_fig_mean_seq.savefig(os.path.join(test_results_fullpath, 'confusion_matrix_mean_seq.png'))




  print('Results with max: ')
  cm_max = confusion_matrix(y_test, Prid_class_max)
  print(accuracy_score(y_test, Prid_class_max))
  df_cm_max, cm_fig_max = results_functions.print_confusion_matrix(cm_max, trained_class_names)
  df_cm_max.to_csv(os.path.join(test_results_fullpath, 'confusion_matrix_max.csv'))
  cm_fig_max.savefig(os.path.join(test_results_fullpath, 'confusion_matrix_max.png'))



  print('Results with max (per-sequence): ')
  cm_max_seq = confusion_matrix(y_seq_test, Prid_sequence_class_max)
  print(accuracy_score(y_seq_test, Prid_sequence_class_max))
  df_cm_max_seq, cm_fig_max_seq = results_functions.print_confusion_matrix(cm_max_seq, trained_class_names)
  df_cm_max_seq.to_csv(os.path.join(test_results_fullpath, 'confusion_matrix_max_seq.csv'))
  cm_fig_max_seq.savefig(os.path.join(test_results_fullpath, 'confusion_matrix_max_seq.png'))


# endregion


# endregion


# endregion
