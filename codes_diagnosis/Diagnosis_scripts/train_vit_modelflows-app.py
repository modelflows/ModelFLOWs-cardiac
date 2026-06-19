# region Import libraries


from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

import os

import numpy as np
import numpy.random as rng


# endregion

np.random.seed(0)

# region Import local libraries


from Training import batch_generator as batch_generator
from Training import callbacks as callbacks
from Training import model as model

from Utils import datasets as my_datasets


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


# region Device selection:
# -1 if 'cpu', >=0 for 'gpu' selection
param['device'] = -1


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

# endregion

print('Introduced all training parameters!')

print('Selected input parameters (training part): \n')

for param_name, param_value in param.items():
  print('- ' + param_name + ': ' + str(param_value))


# region Device Initialization/check


device_name_tf = tf.test.gpu_device_name()
print(device_name_tf)


os.environ["CUDA_VISIBLE_DEVICES"]=str(param['device'])
if param['device'] < 0:
    print('CPU selected')
else:
    print('GPU '  + str(param['device']) + ' selected')


# endregion


# region Training part


# region Load Data


# region Normalizers for training and validation


data_augmentation = my_datasets.data_augmentation_pipeline(param['target_size'])


# endregion


# region Get input data


x_train, y_train, x_train_filenames, y_train_class_names, train_class_names, train_label_encoder = my_datasets.load_dataset_organized_sequences_different_sources(
    param['training_database_path'], img_size=param['target_size'],
    label_encoder=None)
x_val, y_val, x_val_filenames, y_val_class_names, val_class_names, val_label_encoder= my_datasets.load_dataset_organized_sequences_different_sources(
    param['validation_database_path'], img_size=param['target_size'],
    label_encoder=train_label_encoder)

data_augmentation.layers[0].adapt(x_train)


# endregion


# endregion


# region Batch generator


train_dataset = batch_generator.data_generator(x_train, y_train, train_class_names, param)
validation_dataset = batch_generator.data_generator(x_val, y_val, val_class_names, param)


# endregion


# region Model design


model_train = model.create_vit_classifier(data_augmentation, input_shape = (param['target_size'][0], param['target_size'][1], param['num_channels']), image_size = max(param['target_size'][0], param['target_size'][1]), patch_size=param['patch_size'], transformer_layers = param['n_blocks'], num_heads = param['n_heads'], projection_dim = param['projection_dim'], transformer_units = param['transformer_units'], mlp_head_units = param['mlp_head_units'], num_classes = len(train_class_names), vanilla=False)

model_train.summary()

print('Selected ViT model for small datasets in Keras')


# endregion


# Training setup


# region Compiling and optimizer


total_steps = int((len(x_train_filenames) / param['batch_size']) * param['n_epochs'])

optimizer = tfa.optimizers.AdamW(
    learning_rate=param['lr'], weight_decay=param['weight_decay']
)

model_train.compile(
optimizer=optimizer,
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)


# endregion


# region Callbacks

# Called during fitting process.

# endregion


# region Fitting process

print('This code will take hours to run on a CPU. We recommend you to skip this step here and load the proposed pretrained weights so as to see how testing works.')

model_history = model_train.fit(train_dataset, validation_data = validation_dataset,
                                   batch_size = param['batch_size'],
                                   epochs = param['n_epochs'],
                                   steps_per_epoch = param['steps_per_epoch'],
                                   validation_steps = param['validation_steps'],
                                   callbacks = callbacks.callbacks(param, model_train, train_class_names, validation_data = [x_val, y_val], total_steps = total_steps)
                                   )


# endregion


# endregion


