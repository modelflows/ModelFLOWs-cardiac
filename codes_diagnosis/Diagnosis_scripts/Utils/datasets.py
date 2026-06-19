from sklearn import preprocessing
import os
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import PIL

import cv2


# Recorre las carpetas - Lee las imagenes - Les asigna etiquetas - Las mete en un array NumPy listo para el modelo
# COMO SUPONEMOS QUE ESTAN ORGANIZADAS LAS CLASES
# dataset/
#    class_1/
#        sequence_1/
#            img1.png
#            img2.png
#        sequence_2/
#            ...
#    class_2/
#        sequence_1/

def load_dataset_organized_sequences_different_sources(img_dirs, img_size=(256, 256), num_channels=1, label_encoder=None, depth_final_data = np.float32, excluded_classes = []):

    # First count the total number of samples in the set
    total_num_samples = 0

    # region Get the information about the test samples (filenames, class names, and labels)
    all_database_images_filenames = []
    all_database_images_class_names = []
    class_names = []
    output_label_encoder = None

    for img_dir_idx, img_dir in enumerate(img_dirs): 
        #Obtiene las clases, todo dentro de la carpeta
        database_images_directory = glob(os.path.join(img_dir, '*'))

        # Case if we have folders with the class names
        if os.path.isdir(database_images_directory[0]):

            database_as_directories = True

        if database_as_directories:

            print('Situation of samples with known classes')

            # Just in case
            database_images_directory.sort()

            for idx_class, database_class_name_fullpath in enumerate(database_images_directory):
                
                # Por si hemos seleccionado algunas clases que queremos excluir, no cogerlas
                if not(os.path.basename(database_class_name_fullpath) in excluded_classes):

                    all_sequence_filenames_class = glob(os.path.join(database_class_name_fullpath, '*'))
                    all_sequence_filenames_class.sort() #Los ordena

                    # Guara el nombre de las clases
                    if img_dir_idx == 0:

                        class_names = class_names + [os.path.basename(database_class_name_fullpath)]
                    
                    # Obtener todas las imagenes de esa secuencia dentro de la carpeta de cada clase
                    for idx_sequence_class, sequence_filename_class in enumerate(all_sequence_filenames_class):

                        samples_sequence_class_filenames = glob(os.path.join(sequence_filename_class, '*'))
                        samples_sequence_class_filenames.sort()
                        # De la 71 y 72 obtenemos algo como ['healthy/sequence_01/img1.png','healthy/sequence_01/img2.png']
                        # Acumula todas las imagenes del dataset
                        all_database_images_filenames = all_database_images_filenames + samples_sequence_class_filenames

                        # A la lista de clases que asocia a cada imagen, coge el nombre de la clase, multiplica por el numero de imagenes de esa clase,
                        # ej: ['healthy'] * 3 = ['healthy', 'healthy', 'healthy'] y añade eso a a la lista.
                        all_database_images_class_names = all_database_images_class_names + [
                            os.path.basename(database_class_name_fullpath)] * len(samples_sequence_class_filenames)
                        
                        # Nos quedamos con listas de la siguiente forma tras recocorrer todo el dataset y clases:
                        # all_database_images_filenames = [img1,img2,img3,img4,...]
                        # all_database_images_class_names = [healthy,healthy,healthy,disease,...]


    total_num_samples = total_num_samples + len(all_database_images_filenames)

    print('Database with ' + str(total_num_samples) + ' samples and ' + str(len(database_images_directory)) + ' classes')

    if database_as_directories:

        if label_encoder is None: # Generalmente asume que es en entrenamiento la primera vez que vemos el encoder

            le = preprocessing.LabelEncoder()
            # Assume we have the same classes as in training, but if we do not have test samples in a certain class, the corresponding folder would be empty
            le.fit(all_database_images_class_names) 

            #Codifica las clases
            all_database_images_class_labels = le.transform(all_database_images_class_names)

            output_label_encoder = le #Guarda el encoder para reutilizarlo despues


        else: # Caso de validacion ya que se ha generado previamente en train el encoder (Reutilizamos mapeo train)

            all_database_images_class_labels = label_encoder.transform(all_database_images_class_names)

            output_label_encoder = label_encoder

        print('Loading data...')


        # Aqui comenzamos la carga real de imagenes
        for img_idx, img_path in enumerate(all_database_images_filenames):

            if '.npy' in img_path:

                if img_idx == 0: # Solo dice al principio (al ver primera imagen) al user que es formato numpy
                    print('Numpy format')

                image = np.load(img_path)

                if num_channels == 1:

                    # NOTE: THIS SEEMS TO WORK WITH OTHER AMPLITUDES OUT FROM 0-255
                    # Convierte a float32, redimensiona a 256x256 y añade dimension de canal (256,256,1)
                    image = np.expand_dims(cv2.resize(image.astype(depth_final_data), img_size), axis=2)

                if num_channels == 3:

                    image = cv2.resize(image.astype(depth_final_data), img_size)

            else:

                if num_channels == 1:
                    image = np.expand_dims(np.array(PIL.ImageOps.grayscale(PIL.Image.open(img_path)).resize(img_size)),
                                           axis=2)

                if num_channels == 3:
                    image = np.array(PIL.Image.open(img_path).resize(img_size))

            # Solo cuando carga la primera imagen,crea un array gigante donde meterá todas las imágenes.
            if img_idx == 0:                
                all_images = np.zeros((total_num_samples, img_size[0], img_size[1], num_channels),dtype=image.dtype)

            # Se trata de un tensor en 4D (N,H,W,C) -> (N,256,256,1 o 3)
            if img_idx % 20 == 0: #Mensaje cada 20 imagenes
                print('Number of loaded images: ' + str(img_idx))

        print('Data loaded!')


    return all_images, all_database_images_class_labels, all_database_images_filenames, all_database_images_class_names, class_names, output_label_encoder




# Data Augmentation module for default CNN and ViT for small datasets
# Define una función que crea un módulo de preprocesamiento y aumentación de datos
def data_augmentation_pipeline(target_size):

# SOLO SE APLICA A ENTRENAMIENTO, NO A TEST Y VAL, KERAS LO GESTIONA AUTOMATICAMENTE
# Con 1000 imagenes. En 200 epochs, el modelo puede haber visto 1000 versiones ligeramente distintas × 200 = 200.000
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(), # Normaliza los datos, X=(x−μ)/σ​
            layers.Resizing(target_size[0], target_size[1]), # Redimensiona la imagen
            layers.RandomFlip("horizontal"), # Se invierte la imagen horizontalmente (espejo)
            layers.RandomRotation(factor=0.02), # Rota la imagen con cierta probabilidad ±2%×360°≈±7° (En ese rango de aleatoriedad de rotacion, puede salir 0)
            layers.RandomZoom(height_factor=0.2, width_factor=0.2), # Hace zoom aleatorio hasta ±20%.
        ],
        name="data_augmentation",
    )


    return data_augmentation
