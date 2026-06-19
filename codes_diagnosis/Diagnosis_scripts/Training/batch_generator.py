# region Import libraries


from tensorflow import keras
import tensorflow as tf
import numpy as np
import numpy.random as rng


# endregion



# Esta funcion realiza las siguientes tareas:
# 1. Cada clase aporta el mismo número de muestras
# 2. Si sobran posiciones, se rellenan aleatoriamente
# 3. No se repiten muestras dentro del mismo batch

def get_batch(x, y, all_class_samples_indexes, num_samples_per_class_in_batch, num_remaining_samples, param):

  # Creamos los contenedores del batch (64, 256, 256, 1))
  batch_data = np.zeros((param['batch_size'], x.shape[1], x.shape[2], x.shape[3]), dtype = x[0].dtype)
  # (64,)
  targets = np.zeros(param['batch_size'], dtype = int)

  selected_class_samples_relative_per_class = []

  # Just in case: delete previous seeds
  rng.seed(None)

  # Get the samples for the batch (first, what we can take equally for each class)
  start_batch_idx = 0
  for idx_label, class_sample_indexes in enumerate(all_class_samples_indexes):

    # Seleccion muestras aleatorias sin reemplazo
    selected_class_sample_indexes_relative = np.sort(rng.choice(len(all_class_samples_indexes[idx_label]), num_samples_per_class_in_batch, replace=False))
    selected_class_samples_relative_per_class.append(selected_class_sample_indexes_relative)

    selected_class_sample_indexes_absolute = all_class_samples_indexes[idx_label][selected_class_sample_indexes_relative]
    # Copia imagenes al batch
    batch_data[start_batch_idx: start_batch_idx + len(selected_class_sample_indexes_absolute)] = x[selected_class_sample_indexes_absolute]
    targets[start_batch_idx: start_batch_idx + len(selected_class_sample_indexes_absolute)] = y[selected_class_sample_indexes_absolute]
    
    # Actualiza el indice 
    start_batch_idx = start_batch_idx + len(selected_class_sample_indexes_absolute)

  # Get the remaining samples (chosen randomly with the variety as wide as possible)
  selected_class_labels_remaining = np.sort(rng.choice(len(all_class_samples_indexes), num_remaining_samples, replace=False))

  # Todo el procedimiento para seleccionar aleatoriamente las muestras restantes del batch
  for idx_label, class_label in enumerate(selected_class_labels_remaining):

    selected_remaining_class_sample_index_relative = rng.choice([i for i in range(len(all_class_samples_indexes[class_label])) if i not in selected_class_samples_relative_per_class[class_label]])
    # print(selected_remaining_class_sample_index_relative)
    assert(selected_remaining_class_sample_index_relative not in selected_class_samples_relative_per_class[class_label])
    selected_remaining_class_sample_index_absolute = all_class_samples_indexes[class_label][selected_remaining_class_sample_index_relative]

    batch_data[start_batch_idx: start_batch_idx + 1] = x[selected_remaining_class_sample_index_absolute]

    targets[start_batch_idx: start_batch_idx + 1] = y[selected_remaining_class_sample_index_absolute]
    start_batch_idx = start_batch_idx + 1

  return batch_data, targets




# Funcion que realiza un batching balanceado por clase.
def data_generator(x, y, data_class_names, param): # En param esta batch_size

  all_class_samples_indexes = []

  #Calcula el minimo numero de muestras por clase en cada batch, ej: 65//2 = 32
  num_samples_per_class_in_batch = param['batch_size']//len(data_class_names)
  # Si el batch no es divisible exacto, imagenemos que barch size 65 => remaining = 1 que se reparte luego
  num_remaining_samples = param['batch_size'] - (num_samples_per_class_in_batch * len(data_class_names))

  print('\n')
  print("Batch size: " + str(param['batch_size']))
  print("Number of minimum samples per class in each batch: " + str(num_samples_per_class_in_batch))
  print("Number of remaining samples per class in each batch: " + str(num_remaining_samples) + "\n")

  # Numero de muestras por clase, ([0, 1,...], [3000, 2000],...)
  distribution_samples_per_class = np.unique(y, return_counts = True)

  for idx_label, class_label in enumerate(distribution_samples_per_class[0]):

    print("Class " + str(distribution_samples_per_class[0][idx_label]) + ": " + str(distribution_samples_per_class[1][idx_label]))

    # Guarda indices por clase tal que: Clase 0 → índices [0,1,4,6,10,...]
    class_sample_indexes = np.where(y == class_label)[0]

    all_class_samples_indexes.append(class_sample_indexes)

  while True:
    #Generacion del batch
    batch, targets = get_batch(x, y, all_class_samples_indexes, num_samples_per_class_in_batch, num_remaining_samples, param)

    # Devuelve un batch cada vez que el modelo lo pide.
    yield batch, targets


