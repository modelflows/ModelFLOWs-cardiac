# region Import libraries

from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
import os
import sys
import subprocess

import psutil

import tensorflow.keras.backend as K
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, ModelCheckpoint, CSVLogger

# endregion

# region Import local libraries

from Utils import utils

# endregion


def check_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA GPU drivers and CUDA are installed.")


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

# CHECKED

# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.
def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    # target_lr = learning_rate_base = param['lr']
    learning_rate = 0.5 * target_lr * (
            1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmupCosineDecay(keras.callbacks.Callback):
    def __init__(self, model, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):
        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []
        self.model = model

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)


class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, input_data, image_dir):
        super().__init__()
        self.model = model
        self.input_data = input_data

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):

        y_pred = np.asarray(self.model.predict(self.input_data[0], verbose=0))

        y_true = self.input_data[1]
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
        plt.close(fig)

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_roc(y_true, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))
        plt.close(fig)


class Results_evaluation_callback(Callback):

    def __init__(self, checkpoints_folder):
        super(Callback, self).__init__()
        self.checkpoints_folder = checkpoints_folder

    def on_train_begin(self, logs={}):
        # self.times = []
        file_text = open(self.checkpoints_folder + '/results_evaluation.txt', mode='a+')
        file_text.write('n_iteration  training_time(s)  timestamp  total_time(s)\n')
        # file_text.write('n_iteration  val_acc_value  training_time(s) \n')

        self.training_begin_time = time.time()

        file_text.close()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        # self.times.append(time.time() - self.epoch_time_start)
        remember_the_time = time.time() - self.epoch_time_start
        remember_total_time = time.time() - self.training_begin_time
        remember_the_timestamp = '{}'.format(time.strftime("%Y-%m-%d_%H-%M"))

        file_text = open(self.checkpoints_folder + '/results_evaluation.txt', mode='a+')
        # file_text.write(str(epoch) + ': ' + str(val_acc) + ': ' + str(remember_the_time) + '\n')
        file_text.write(str(epoch) + ': ' + str(remember_the_time) + ': ' + remember_the_timestamp + ': ' + str(remember_total_time) + '\n')
        file_text.close()

        # Call the function to check GPU usage
        print('---------------------GPU USAGE---------------------')
        check_gpu_usage()
        print('---------------------END GPU USAGE---------------------')

        print('---------------------RAM USAGE---------------------')
        # Call the function to check RAM usage
        check_ram_usage()
        print('---------------------END RAM USAGE---------------------')



class LRTensorBoard(Callback):
    def __init__(self, log_dir):  # , **kwargs):  # add other arguments to __init__ if you need
        # super().__init__(log_dir=log_dir, **kwargs)
        super(Callback, self).__init__()
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        print(logs)
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def callbacks(param, model, info_classes, validation_data=None, total_steps=None):


    final_callbacks = []
    timestamp_model_path = time.strftime("%Y-%m-%d_%H-%M")
    model_complete_path = os.path.join(param['model_save_path'] + '__' + timestamp_model_path)

    num_folder = 0

    while(os.path.exists(model_complete_path)):
        model_complete_path = os.path.join(param['model_save_path'] + '__' + timestamp_model_path + '_' + str(num_folder))
        num_folder = num_folder + 1

    os.makedirs(model_complete_path)

    utils.export_input_arguments(param, model_complete_path)

    np.save(os.path.join(model_complete_path, 'class_names.npy'), info_classes)

    tensorboard_callback = TensorBoard(log_dir=model_complete_path, write_graph=True, write_images=True,
                                       histogram_freq=0)  # True, True,1
    final_callbacks.append(tensorboard_callback)

    model_saved_path = os.path.join(model_complete_path, 'saved_model_' + 'epoch_{epoch:03d}.h5')

    monitorized_metric = 'val_accuracy'

    print('Model Checkpoint')

    checkpoint_callback = ModelCheckpoint(model_saved_path, monitor=monitorized_metric, verbose=1,
                                          save_weights_only=True, save_best_only=True, include_optimizer=False)

    final_callbacks.append(checkpoint_callback)

    print('Done!')

    lr_tensorboard = LRTensorBoard(model_complete_path)
    final_callbacks.append(lr_tensorboard)

    results_callback = Results_evaluation_callback(model_complete_path)
    final_callbacks.append(results_callback)

    if total_steps is not None:
        warmup_epoch_percentage = 0.1
        warmup_steps = int(total_steps * warmup_epoch_percentage)

        scheduled_lrs = WarmupCosineDecay(model,
                                          total_steps=total_steps,
                                          warmup_steps=warmup_steps,
                                          hold=0,  # int(warmup_steps/2)
                                          start_lr=0.0,
                                          target_lr=param['lr'])

        final_callbacks.append(scheduled_lrs)

    if validation_data is not None:

        performance_cbk_validation = PerformanceVisualizationCallback(
            model=model,
            input_data=validation_data,
            image_dir=os.path.join(model_complete_path, 'performance_visualizations_validation'))

        final_callbacks.append(performance_cbk_validation)

    return final_callbacks