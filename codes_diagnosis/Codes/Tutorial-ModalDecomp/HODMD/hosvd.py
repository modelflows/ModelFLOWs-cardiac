#!/usr/bin/env Python 3.10.4
# -*- coding: utf-8 -*-

################### HOSVD ###################

__author__ = "Egoitz Maiora Otaegi"

"""
Perform HOSVD to input data and retain all the singular values.
"""

import numpy as np
import time             # To calculate execution time
import os

import scipy.linalg
import sklearn
import torch
import torch.linalg
import tensorflow as tf

torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def matlen(var):
    '''Equivalent to Matlab's length()'''
    if np.size(np.shape(var))==1:
        x = np.size(var)
    else:
        x = max(np.shape(var))
    return x

def unfold(A,dim):
    '''Turns tensor into matrix keeping the columns on dim'''
    ax=np.arange(A.ndim)
    return np.reshape(np.moveaxis(A,ax,np.roll(ax,dim)),(A.shape[dim],A.size//A.shape[dim]))

def fold(B,dim,shape):
    '''Reverse operation to the unfold function'''
    ax=np.arange(len(shape))
    shape=np.roll(shape,-dim)
    A=np.reshape(B,shape)
    return np.moveaxis(A,ax,np.roll(ax,-dim))

def tprod(S,U):
    '''Tensor product of an ndim-array and multiple matrices'''
    T = S
    shap = list(np.shape(S))
    for i in range(0,np.size(U)):
        x = np.count_nonzero(U[0][i])
        if not x==0:
            shap[i] = np.shape(U[0][i])[0]
            H = unfold(T,i)
            T = fold(np.dot(U[0][i],H),i,shap)
    return T

def svdtrunc(A, ns, svd_library = 'numpy', export_us=0, snd = 0, svd_driver = None):

    # IMPORTANT NOTE:
    # The matrixes are not unique, so depending on hardware and software, the computer singular vectors may be different
    # (in our case, it is like we have vectors with the sign changed). See the following link:
    # https://pytorch.org/docs/stable/generated/torch.linalg.svd.html

    start_svd = time.time()

    '''Truncated svd'''
    # NOTE: full_matrices to 'True' is not possible because we have memory errors

    if svd_library == 'numpy':

        U, S, _ = np.linalg.svd(A, full_matrices=False)

    if svd_library == 'hybrid':

        if not (device.type == 'cpu'):

            # NOTE: not possible to use the value gesvda in the field driver because of CUDA memory problems
            # Checked that the option 'gesvda' seems to be the best option
            if svd_driver == 'gesvd':
                svd_driver_used = 'gesvda'
            else:
                svd_driver_used = svd_driver

            U, S, _ = torch.linalg.svd(torch.tensor(A).to(device), driver = svd_driver_used, full_matrices=False)

        else:

            U, S, _ = torch.linalg.svd(torch.tensor(A).to(device), full_matrices=False)

        if export_us:
            index = 1
            while (os.path.exists('U_' + str(index) + '_' + svd_library + '_' + '.pt')):
                index = index + 1

            torch.save(U, 'U_' + str(index) + '_' + svd_library + '_' + '.pt')
            torch.save(S, 'S_' + str(index) + '_' + svd_library + '_' + '.pt')

        if not(device == 'cpu'):
          U = U.detach().cpu()
          S = S.detach().cpu()



    if svd_library == 'hybrid_lower_precision':

        if not (device.type == 'cpu'):

            # NOTE: not possible to use the value gesvda in the field driver because of CUDA memory problems
            # Checked that the option 'gesvda' seems to be the best option
            if svd_driver == 'gesvd':
                svd_driver_used = 'gesvda'
            else:
                svd_driver_used = svd_driver

            U, S, _ = torch.linalg.svd(torch.tensor(A.astype(np.float32)).to(device), driver = svd_driver_used, full_matrices=False)

        else:

            U, S, _ = torch.linalg.svd(torch.tensor(A.astype(np.float32)).to(device), full_matrices=False)

        if export_us:
            index = 1
            while (os.path.exists('U_' + str(index) + '_' + svd_library + '_' + '.pt')):
                index = index + 1

            torch.save(U, 'U_' + str(index) + '_' + svd_library + '_' + '.pt')
            torch.save(S, 'S_' + str(index) + '_' + svd_library + '_' + '.pt')

        if not(device == 'cpu'):
          U = U.detach().cpu()
          S = S.detach().cpu()





    if svd_library == 'torch':

        if not (device.type == 'cpu'):

            # NOTE: not possible to use the value gesvda in the field driver because of CUDA memory problems
            # Checked that the option 'gesvda' seems to be the best option
            if svd_driver == 'gesvd':
                svd_driver_used = 'gesvda'
            else:
                svd_driver_used = svd_driver

            U, S, _ = torch.linalg.svd(torch.tensor(A).to(device), driver = svd_driver_used, full_matrices=False)

        else:

            U, S, _ = torch.linalg.svd(torch.tensor(A).to(device), full_matrices=False)

        if export_us:
            index = 1
            while (os.path.exists('U_' + str(index) + '_' + svd_library + '_' + '.pt')):
                index = index + 1

            torch.save(U, 'U_' + str(index) + '_' + svd_library + '_' + '.pt')
            torch.save(S, 'S_' + str(index) + '_' + svd_library + '_' + '.pt')

        if not(device == 'cpu'):
          U = U.detach().cpu()
          S = S.detach().cpu()


    if svd_library == 'torch_new':

        U, S, _ = torch.svd(torch.tensor(A).to(device))

        if export_us:
            index = 1
            while (os.path.exists('U_' + str(index) + '_' + svd_library + '_' + '.pt')):
                index = index + 1

            torch.save(U, 'U_' + str(index) + '_' + svd_library + '_' + '.pt')
            torch.save(S, 'S_' + str(index) + '_' + svd_library + '_' + '.pt')

        if not(device == 'cpu'):
          U = U.detach().cpu()
          S = S.detach().cpu()


    if svd_library == 'torch_32f':

        U, S, _ = torch.linalg.svd(torch.tensor(A.astype(np.float32)).to(device), full_matrices=False)


        if not(device == 'cpu'):
          U = U.detach().cpu()
          S = S.detach().cpu()

        U = U.to(torch.float64)
        S = S.to(torch.float64)

    if svd_library == 'torch_32f_new':

        U, S, _ = torch.svd(torch.tensor(A.astype(np.float32)).to(device), some = True)

        if not (device == 'cpu'):
            U = U.detach().cpu()
            S = S.detach().cpu()

        U.to(torch.float64)
        S.to(torch.float64)


    if svd_library == 'tensorflow':


        S, U, _ = tf.linalg.svd(A)

        U = U.numpy()
        S = S.numpy()



        # U = tf.make_ndarray(tf.make_tensor_proto(U))
        # S = tf.make_ndarray(tf.make_tensor_proto(S))

    if svd_library == 'scipy':

        if snd == 1:
            U, S, _ = sklearn.utils.extmath.randomized_svd(A, n_components = ns)
        else:
            U, S, _ = scipy.linalg.svd(A, full_matrices=False, overwrite_a=False, check_finite=False, lapack_driver='gesdd') # gesdd, gesvd)


    svd_proc_time = time.time() - start_svd

    print("SVD execution time: " + str(svd_proc_time))

    if export_us:

        index = 1
        while(os.path.exists('U_' + str(index) + '_' + svd_library + '_' + '.npy')):
            index = index + 1

        np.save('U_' + str(index) + '_' + svd_library + '_' + '.npy', U)
        np.save('S_' + str(index) + '_' + svd_library + '_' + '.npy', S)


    return U, S, svd_proc_time


def HOSVD_function(T,n, svd_library = 'numpy', snd = 0, svd_driver = None):
    '''Perform hosvd to tensor'''
    M = np.shape(T)
    P = matlen(M)
    U = np.zeros(shape=(1,P), dtype=object)
    UT = np.zeros(shape=(1,P), dtype=object)
    sv = np.zeros(shape=(1,P), dtype=object)
    producto = n[0]
    
    total_hosvd_time = 0

    for i in range(1,P):
        producto = np.dot(producto,n[i])
    
    n = list(n)       # to be able to assign values
    for i in range(0,P):
        n[i] = np.amin((n[i],producto/n[i]))
        n[i]=int(n[i])
    #for i in range(0,P):
        #n[i] = np.amin((n[i],producto/n[i]))
        #n[i] = int(n[i])     # Indexes must be integers
        A = unfold(T,i)
        Uaux = []
        # SVD based reduction of the current dimension (i):
        Ui, svi, svi_time = svdtrunc(A, n[i], svd_library, snd, svd_driver)

        total_hosvd_time = total_hosvd_time + svi_time

        if n[i] < 2:
            Uaux = np.zeros((np.shape(Ui)[0],2))
            Uaux[:,0] = Ui[:,0]
            U[0][i] = Uaux
        else:
            U[0][i] = Ui[:,0:n[i]]
        # U[i] = Ui
        UT[0][i] = np.transpose(U[0][i])
        sv[0][i] = svi



    S = tprod(T, UT)
    TT = tprod(S, U)
    return TT, S, U, sv, n, total_hosvd_time

def HOSVD(Tensor,varepsilon1,nn,n,TimePos, svd_library = 'numpy', svd_driver = None):
    '''Perform hosvd to input data and retain all the singular values'''

    total_hosvd_time = 0


    snd = 0
    [TT,S,U,sv,n, hosvd_func_time] = HOSVD_function(Tensor,n, svd_library, snd, svd_driver)

    total_hosvd_time = total_hosvd_time + hosvd_func_time

    ## Set the truncation of the singular values using varepsilon1
    # (automatic truncation)
    for i in range(1,np.size(nn)):
        count = 0
        for j in range(0,np.shape(sv[0][i])[0]):
            if sv[0][i][j]/sv[0][i][0]>=varepsilon1:
                count = count+1
            else:
                break
        nn[i] = count
    print(f'Initial number of singular values: {n}')
    print(f'Number of singular values retained: {nn}')
    
    ## Perform HOSVD retaining n singular values: reconstruction of the modes
    snd = 0
    [TT,S2,U,sv,nn2, hosvd_func_time] = HOSVD_function(Tensor,nn, svd_library, snd, svd_driver)

    total_hosvd_time = total_hosvd_time + hosvd_func_time

    ## Construct the reduced matrix containing the temporal modes:
    UT = np.zeros(shape=np.shape(U), dtype=object)
    hatT = []
    for pp in range(0,np.size(nn2)):
        UT[0][pp] = np.transpose(U[0][pp])
    for kk in range(0,nn2[TimePos-1]):
        hatT.append(np.dot(sv[0][TimePos-1][kk],UT[0][TimePos-1][kk,:]))
    hatT = np.reshape(hatT, newshape=(len(hatT),np.size(hatT[0])))
    return hatT,U,S2,sv,nn2, total_hosvd_time