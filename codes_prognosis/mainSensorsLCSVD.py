import matplotlib.pyplot as plt
import hdf5storage # Used later to save the reconstructed data
import numpy as np
import time
import sensors as ps
from mplcursors import cursor
#from BL_Re600_load import loadBL600 as BL
from VKI_tensors_load import loadVKIcyl as VKI
#from Jets_load import loadJetsLES as JET
#from Cyl2D_Re100_load import loadCyl2D as CYL2D
#from Cyl3D_load import loadCyl3D as CYL3D

# Busca la mejor colocación de sensores tal que la reconstrucción por lcSVD tenga un error aceptable.
def lcsvd_sensors(Tensor, n_sensors, sel):

    def sensor_opt(Tensor0, n_sensors):

        shape = Tensor0.shape

        if Tensor0.ndim == 3:
            Matrix = np.reshape(Tensor0, (np.prod(shape[:-1]), shape[-1]))

            mesh_size = shape[:-1]

        else:
            norm = np.linalg.norm(Tensor0.reshape(shape[0], np.prod(shape[1:])), axis = 0)

            norm = np.reshape(norm, (1, *shape[1:]))

            Matrix = np.reshape(norm, (np.prod(shape[1:-1]), shape[-1])) # Matrix loses 1 dimensions since there is now only 1 vel component (the norm)

            mesh_size = shape[1:-1]

        # Incorporar escalado de datos
        X_train = np.transpose(Matrix, (1, 0))

        n_samples, n_features = X_train.shape
        print('Number of samples:', n_samples)
        print('Number of possible sensors:', n_features)

        # Should we scale the data? The results seem to be OK...

        model = ps.SSPOR(basis = ps.basis.SVD(n_basis_modes = 2), n_sensors=n_sensors)

        model.fit(X_train)
        sensors = model.get_selected_sensors() # Sensor positions refering to the flattened out spatial mesh

        sens = np.zeros(n_features)
        sens[sensors] = 1
        matrix = np.array(np.reshape(sens, mesh_size)) # Matrix for zeros shaped Nx, Ny which contains 1 in the sensor positions

        x = []
        y = []
        z = []

        if len(shape) == 3: # For 2D data with only 1 component (Nx, Ny, Nt)
            for i in range(shape[1]):
                for j in range(shape[0]):
                    if matrix[j, i] == 1:
                        x.append(i) # X coordinates of the sensors
                        y.append(j) # Y coordinates of the sensors

        elif len(shape) in [4, 5]: # 2D data with more than 1 component, and 3D data
            for i in range(shape[2]):
                for j in range(shape[1]):
                    if matrix.ndim == 2:
                        if matrix[j, i] == 1:
                            x.append(i) # X coordinates of the sensors
                            y.append(j) # Y coordinates of the sensors
                    elif matrix.ndim == 3:
                        for k in range(shape[3]):
                            if matrix[j, i, k] == 1:
                                x.append(i) # X coordinates of the sensors
                                y.append(j) # Y coordinates of the sensors
                                z.append(k) # Z coordinates of the sensors

        if matrix.shape == 2:
            z = None

        return sensors, x, y, z


    def SVD_sensorChecker(tensor, sensors, sel):

        shape = tensor.shape
        dims_prod = np.prod(shape[:-1])
        Tensor = np.reshape(tensor, (dims_prod, shape[-1])) # dims_prod is the compression of the spatial data, shape[-1] y the shape of the spatial data
        Ared = Tensor[sensors, :]

        Ured, Sred, Vred = np.linalg.svd(Ared, full_matrices = False)

        Sred = np.diag(Sred)
        Ured = Ured[:, :sel]
        Sred = Sred[:sel, :sel]
        Vred = Vred.conj().T
        Vred = Vred[:, :sel]

        Q, R = np.linalg.qr(Ured)
        Ured = Ured @ np.linalg.inv(R[:sel, :])
        Q1, R1 = np.linalg.qr(Vred)
        Vred = Vred @ np.linalg.inv(R1[:sel, :])

        ss = Ured.conj().T @ Ared @ Vred
        ss1 = np.sign(np.diag(np.diag(ss)))

        Vred = Vred @ ss1

        # Calculate the original SVD modes using the reduced modes
        U = (Tensor @ Vred) @ np.linalg.inv(Sred)
        V = (Tensor[sensors, :].conj().T @ Ured) @ np.linalg.inv(Sred)
        S = Sred

        TensorCheck = np.reshape((U @ S) @ V.conj().T, (shape))

        return TensorCheck

    RelativeErrorRMS = 1

    i = 1

    while RelativeErrorRMS * 100 > 16.5: # 1 for Cyl2D, BL and JET, 2.5 for Cyl3D

        print(f'Iteration number {i}')

        i += 1

        sensors, x, y, z = sensor_opt(Tensor, n_sensors)

        t0 = time.time()

        TensorCheck = SVD_sensorChecker(Tensor, sensors, sel)

        t1 = time.time()

        print(f'Low Cost SVD execution time: {(t1 - t0):.5f} seconds')

        Norm2V = np.linalg.norm(Tensor.flatten(), 2)
        diff = (Tensor - TensorCheck).flatten()
        Norm2diff = np.linalg.norm(diff, ord=2)
        RelativeErrorRMS = Norm2diff/Norm2V

        print(f'Low Cost SVD reconstruction error for the sensors combination: {(RelativeErrorRMS * 100):.5f} %')

    print(f'Low Cost SVD reconstruction error for the optimal sensors combination: {(RelativeErrorRMS * 100):.5f} %')

    if Tensor.ndim == 3:
        fig, ax = plt.subplots(1, 1, figsize = (20, 7))
        plt.suptitle('Sensor positions')
        ax.contourf(Tensor[..., 0], alpha = 0.5)
        ax.scatter(x, y, color = 'k')
        cursor(hover=True)
        plt.show()

    if Tensor.ndim == 4:
        comps = Tensor.shape[0]
        fig, ax = plt.subplots(1, comps, figsize = (20, 7))
        plt.suptitle('Sensor positions')
        for col in range(comps):
            ax[col].contourf(Tensor[col, ..., 0], alpha = 0.5)
            ax[col].scatter(x, y, color = 'k')
        cursor(hover=True)
        plt.show()

    if Tensor.ndim == 5:
        comps = Tensor.shape[0]
        nz = int(Tensor.shape[3] / 2)
        fig, ax = plt.subplots(1, comps, figsize = (20, 7))
        plt.suptitle('Sensor positions | XY plane')
        for col in range(comps):
            ax[col].contourf(Tensor[col, ..., nz, 0], alpha = 0.5)
            ax[col].scatter(x, y, color = 'k')
        cursor(hover=True)
        plt.tight_layout()
        plt.show()

    return sensors, x, y, z

def LCSVD(tensor, sensors, Time, sel):
    # def LCSVD(tensor, Ared, sel):
    # Ared = np.reshape(Ared, (np.prod(Ared.shape[:-1]), Ared.shape[-1]))
    # sensors = Ared.shape[0]
    # time = Ared.shape[1]

    print(f'High resolution dataset shape: {tensor.shape}')

    shape = tensor.shape
    Tensor = np.reshape(tensor, (np.prod(tensor.shape[:-1]), tensor.shape[-1]))
    Ared = Tensor[sensors, :]
    Ared = Ared[:, Time]

    print(f'Low resolution dataset shape: {Ared.shape}') # This is the downsampled data, which has been downsampled with "interval" and "time", but will eventually
    # Be passed to this program as another dataset which will be experimental data that requires reconstruction

    T0 = time.time()

    # SVD applied to the reduced matrix
    Ured, Sred, Vred = np.linalg.svd(Ared, full_matrices = False)

    Sred = np.diag(Sred)
    Ured = Ured[:, :sel]
    Sred = Sred[:sel, :sel]
    Vred = Vred.conj().T
    Vred = Vred[:, :sel]

    Q, R = np.linalg.qr(Ured)
    Ured = Ured @ np.linalg.inv(R[:sel, :])
    Q1, R1 = np.linalg.qr(Vred)
    Vred = Vred @ np.linalg.inv(R1[:sel, :])

    ss = Ured.conj().T @ Ared @ Vred
    ss1 = np.sign(np.diag(np.diag(ss)))

    Vred = Vred @ ss1

    # Calculate the original SVD modes using the reduced modes
    U = (Tensor[:, Time] @ Vred) @ np.linalg.inv(Sred)
    V = (Tensor[sensors, :].conj().T @ Ured) @ np.linalg.inv(Sred)
    S = Sred

    TensorAprox = np.reshape((U @ S) @ V.conj().T, (shape))

    T1 = time.time()

    print(f'Low Cost SVD reconstruction completed in {(T1 - T0):.5f} seconds')

    print(f'lcSVD Reconstructed data shape: {TensorAprox.shape}')

    Norm2V = np.linalg.norm(Tensor.flatten(), 2)
    diff = (tensor - TensorAprox).flatten()
    Norm2diff = np.linalg.norm(diff, ord=2)
    RelativeErrorRMS = Norm2diff/Norm2V
    print(f'Low Cost SVD reconstruction error: {(RelativeErrorRMS * 100):.5f} %')

    if tensor.ndim == 3:
        fig, ax = plt.subplots(1, 2, figsize = (20, 7))
        ax[0].contourf(tensor[..., 0])
        ax[0].set_title('Ground Truth')
        ax[1].contourf(TensorAprox[..., 0])
        ax[1].set_title('Reduced + lcSVD')
        plt.show()

    if tensor.ndim == 4:
        comps = tensor.shape[0]
        for i in range(comps):
            fig, ax = plt.subplots(1, 2, figsize = (20, 7))
            plt.suptitle(f'Ground Truth vs Reduced + lcSVD | Component {i+1}')
            ax[0].contourf(tensor[i, :, :, 0])
            ax[1].contourf(TensorAprox[i, :, :, 0])
            plt.tight_layout()
            plt.show()

    if tensor.ndim == 5:
        comps = tensor.shape[0]
        nz = int(tensor.shape[3] / 2)
        for i in range(comps):
            fig, ax = plt.subplots(1, 2, figsize = (20, 7))
            plt.suptitle(f'Ground Truth vs Reduced + lcSVD | Component {i+1} | XY Plane')
            ax[0].contourf(tensor[i, :, :, nz, 0])
            ax[1].contourf(TensorAprox[i, :, :, nz, 0])
            plt.tight_layout()
            plt.show()

    return TensorAprox

def main():
    Tensor = VKI(2600)

    n_modes = 20
    n_sensors = 50

    sensors, x, y, z = lcsvd_sensors(Tensor, n_sensors, n_modes)

    # Experiment = hdf5storage.loadmat('experimental_data.mat)
    # Data_exp = list(Experiment.values())[-1]

    # TensorAprox = LCSVD(Tensor, Data_exp, n_modes)

    Time = list(np.arange(0, Tensor.shape[-1], 4)) # To select certain snapshots (Every 4th snapshot)

    # To select "points" amount of snapshots: Time = list(np.linspace(0, tensor.shape[-1]) - 1, points).astype(int))

    # Just as a reminder, in the future, the DNS tensor wont be downsampled, but an experimental dataset will be passed,
    # which will already be downsampled in space and time

    # Is it necessary to implement data scaling?? Results seem to be OK

    TensorAprox = LCSVD(Tensor, sensors, Time, n_modes)

if __name__ == '__main__':
    main()

