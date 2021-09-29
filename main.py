
import DataGeneration
import DNN
import numpy as np

import Plots

filename_data_train = 'DATA/data_train_01_dummy.txt'
filename_data_test = 'DATA/data_test_01_dummy.txt'
filename_dnn = 'NN/NN_01_dummy.h5'

CSNRdB = np.arange(20, 31, 5)
CSNR = 10 ** (CSNRdB/10)
sigma = np.sqrt(10 ** (-CSNRdB/10))

symbols = int(1e3)
alpha = 2

print('Generate Training Data...')
_, _ = DataGeneration.analog_generator(filename_data_train, CSNR, symbols, alpha, sigma)

neurons = 15

print('Train Deep Neural Network...')
DNN.analog_trainDNN(filename_data_train, filename_dnn, neurons)

symbols = int(1e3)

print('Generate Testing Data...')
SDR_AWGN, SDR_Ray = DataGeneration.analog_generator(filename_data_test, CSNR, symbols, alpha, sigma)

print('Test Deep Neural Network...')
SDR_DNN = DNN.analog_runDNN(filename_data_test, filename_dnn)

print('Ploting the Results')
Plots.PlotSDR(CSNRdB, SDR_AWGN, SDR_Ray, SDR_DNN)
