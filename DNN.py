
import DataGeneration
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense


def config_GPU():

    if tf.test.gpu_device_name() != '/device:GPU:0':
        print('WARNING: GPU device not found.')
    else:
        print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))
        physical_devices = tf.config.list_physical_devices('GPU')
        # print(physical_devices[0])
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def analog_trainDNN(filename_data, filename_dnn, neurons):

    # Make sure the GPU is configured
    config_GPU()

    # Load the data
    out_AWGN, out_Ray, out_delta_Ray, data_ray, CSNR, symbols = DataGeneration.load(filename_data)

    out_AWGN = np.reshape(out_AWGN, (2, symbols*len(CSNR))).transpose()
    out_Ray = np.reshape(out_Ray, (1, symbols*len(CSNR))).transpose()
    out_delta_Ray = np.reshape(out_delta_Ray, (1, symbols*len(CSNR))).transpose()

    data_DNN = np.append(out_AWGN, np.append(out_Ray, out_delta_Ray, axis=1), axis=1)
    np.random.shuffle(data_DNN)
    out_AWGN = data_DNN[:, 0:2]
    out_Ray = data_DNN[:, 2:3]
    out_delta_Ray = data_DNN[:, 3:4]

    # data_in = np.append(out_Ray, out_delta_Ray, axis=0)
    data_in = np.array(out_Ray)
    data_out = np.array(out_AWGN)
    # print(np.shape(data_in))

    # DNN
    net = Sequential([
        Input(shape=(1,)),
        Dense(neurons, activation='relu'),
        Dense(neurons, activation='relu'),
        Dense(neurons, activation='relu'),
        Dense(2),
    ])

    # net compile options
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    net.compile(opt, loss_fn)

    # net training options
    batch_size = 50
    epochs = 50
    verbose = 1
    validation_split = 0.2
    net.fit(data_in, data_out,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split)
    net.save(filename_dnn)

    # prediction = net.predict(data_in)
    # print(np.shape(prediction))


def analog_runDNN(filename_data, filename_dnn):

    # Make sure the GPU is configured
    config_GPU()

    # Load the data
    out_AWGN, out_Ray, out_delta_Ray, data_ray, CSNR, symbols = DataGeneration.load(filename_data)

    data_ray = np.array(data_ray)
    out_Ray = np.array(out_Ray)

    # data_in = np.append(out_Ray, out_delta_Ray, axis=2)
    data_in = np.reshape(out_Ray, (len(CSNR), symbols, 1))

    MSE_DNN = np.zeros(len(CSNR))
    SDR_DNN = np.zeros(len(CSNR))

    # Loads the net
    net = tf.keras.models.load_model(filename_dnn)

    for i in range(len(CSNR)):
        # Predicts
        DataOut_DNN = net.predict(data_in[i, :, :])
        # print(np.shape(DataOut_DNN))
        DataOut_DNN = np.array(DataOut_DNN).transpose()

        MSE_DNN[i] = 1 / 2 * np.mean((data_ray[0, :, i] - DataOut_DNN[0, :]) ** 2 + (data_ray[1, :, i] - DataOut_DNN[1, :]) ** 2)
        SDR_DNN[i] = 10 * np.log10(1 / MSE_DNN[i])

    return SDR_DNN

# analog_trainDNN('test_generation.txt', 'test_dnn.h5', 10, 15, 15)
