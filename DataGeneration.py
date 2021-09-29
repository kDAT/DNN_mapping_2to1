import numpy as np
import json

import Plots
import SpiralMapping


def analog_generator(filename, CSNR, symbols, alpha, sigma):
    rxSignal_aux = np.zeros((len(CSNR), symbols))
    rxSignal_ML_aux = np.zeros((len(CSNR), symbols))
    gamma = np.zeros(len(CSNR))
    delta = np.zeros(len(CSNR))
    txSignal = np.zeros((len(CSNR), symbols))

    rxSignal_Ray_aux = np.zeros((len(CSNR), symbols))
    rxSignal_ML_Ray_aux = np.zeros((len(CSNR), symbols))
    gamma_Ray = np.zeros(len(CSNR))
    delta_Ray = np.zeros((len(CSNR), symbols))
    txSignal_Ray = np.zeros((len(CSNR), symbols))
    h_aux = np.zeros((len(CSNR), symbols))

    data = np.zeros((2, symbols, len(CSNR)))
    DataOut = np.zeros((2, symbols, len(CSNR)))
    data_ray = np.zeros((2, symbols, len(CSNR)))
    DataOut_Ray = np.zeros((2, symbols, len(CSNR)))

    MSE_AWGN = np.zeros(len(CSNR))
    SDR_AWGN = np.zeros(len(CSNR))
    MSE_Ray = np.zeros(len(CSNR))
    SDR_Ray = np.zeros(len(CSNR))

    #
    for i in range(len(CSNR)):
        print(f'{i+1}/{len(CSNR)}')

        np.random.seed(987654321)
        delta[i] = 2 * np.pi * ((6 * 0.16 ** 2) / (CSNR[i])) ** (1 / 4)

        x = np.zeros((2, symbols))
        for k in range(symbols):
            s1 = np.random.randn()
            s2 = np.random.randn()
            # x[:, k] = [s1, s2]
            x[0, k] = s1
            x[1, k] = s2
            txSignal[i, k] = SpiralMapping.mapping(s1, s2, delta[i], alpha)
        data[:, :, i] = x
        gamma[i] = np.sqrt(np.mean(np.absolute(txSignal[i, :]) ** 2))

        # Received signal
        rxSignal_aux[i, :] = txSignal[i, :] + gamma[i] * sigma[i] * np.random.randn(1, symbols)

        # Decodificacao ML com filtro MMSE
        rxSignal_ML_aux[i, :] = rxSignal_aux[i, :] / (1 + 2 * sigma[i] ** 2)

        theta_est = np.sign(rxSignal_ML_aux[i, :]) * (np.absolute(rxSignal_ML_aux[i, :]) ** (1 / alpha))
        s1_hat = delta[i] / np.pi * np.sign(theta_est) * theta_est * np.sin(theta_est)
        s2_hat = delta[i] / np.pi * theta_est * np.cos(theta_est)
        # print(np.shape([s1_hat, s2_hat]))
        DataOut[:, :, i] = [s1_hat, s2_hat]

        # MSE_AWGN[i] = (1 / 2) * np.mean((data[0, :, i] - DataOut[0, :, i]) ** 2 + (data[1, :, i] - DataOut[1, :, i]) ** 2)
        MSE_AWGN[i] = np.mean(np.mean(np.square(data[:, :, i] - DataOut[:, :, i])))
        SDR_AWGN[i] = 10 * np.log10(1 / MSE_AWGN[i])

        # Rayleigh Channel
        np.random.seed(987654321)
        h_aux[i, :] = np.absolute(np.sqrt(0.5) * (np.random.randn(1, symbols) + 1.j * np.random.randn(1, symbols)))
        delta_Ray[i, :] = 2 * np.pi * ((6 * 0.16 ** 2) / (h_aux[i, :] ** 2 * CSNR[i])) ** (1 / 4)

        x = np.zeros((2, symbols))
        for k in range(symbols):
            s1 = np.random.randn()
            s2 = np.random.randn()
            x[:, k] = [s1, s2]
            txSignal_Ray[i, k] = SpiralMapping.mapping(s1, s2, delta_Ray[i, k], alpha)
        data_ray[:, :, i] = x
        gamma_Ray[i] = np.sqrt(np.mean(np.absolute(txSignal_Ray[i, :]) ** 2))

        np.random.seed(987654321)

        # Received signal
        rxSignal_Ray_aux[i, :] = txSignal_Ray[i, :] * h_aux[i, :] + gamma_Ray[i] * sigma[i] * np.random.randn(1, symbols)

        # Decodificacao ML com filtro MMSE
        rxSignal_ML_Ray_aux[i, :] = h_aux[i, :] * rxSignal_Ray_aux[i, :] / (h_aux[i, :] ** 2 + 2 * sigma[i] ** 2)

        theta_est = np.sign(rxSignal_ML_Ray_aux[i, :]) * (np.absolute(rxSignal_ML_Ray_aux[i, :]) ** (1 / alpha))
        s1_hat = delta_Ray[i] / np.pi * np.sign(theta_est) * theta_est * np.sin(theta_est)
        s2_hat = delta_Ray[i] / np.pi * theta_est * np.cos(theta_est)
        DataOut_Ray[:, :, i] = [s1_hat, s2_hat]

        MSE_Ray[i] = 1 / 2 * np.mean((data_ray[0, :, i] - DataOut_Ray[0, :, i]) ** 2 + (data_ray[1, :, i] - DataOut_Ray[1, :, i]) ** 2)
        SDR_Ray[i] = 10 * np.log10(1 / MSE_Ray[i])

    out_AWGN = DataOut  # (2, symbols, CSNR)
    out_Ray = rxSignal_ML_Ray_aux  # (CSNR, symbols)
    out_delta_Ray = delta_Ray  # (CSNR, symbols)

    save(filename, out_AWGN, out_Ray, out_delta_Ray, data_ray, CSNR, symbols)
    print('Generate Done!')

    return SDR_AWGN, SDR_Ray


def save(filename, out_AWGN, out_Ray, out_delta_Ray, data_ray, CSNR, symbols):
    # filename .txt
    data = {"out_AWGN": out_AWGN.tolist(),
            "out_Ray": out_Ray.tolist(),
            "out_delta_Ray": out_delta_Ray.tolist(),
            "data_ray": data_ray.tolist(),
            "CSNR": CSNR.tolist(),
            "symbols": symbols,
            }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    out_AWGN = data["out_AWGN"]
    out_Ray = data["out_Ray"]
    out_delta_Ray = data["out_delta_Ray"]
    data_ray = data["data_ray"]
    CSNR = data["CSNR"]
    symbols = data["symbols"]

    return out_AWGN, out_Ray, out_delta_Ray, data_ray, CSNR, symbols


# test
# CSNRdB = np.array([40, 50, 60])
# CSNR = 10 ** (CSNRdB / 10)
# sigma = np.sqrt(10 ** (-CSNRdB / 10))
# # print(sigma)
# filename = 'test_generation.txt'
# SDR_AWGN, SDR_Ray = analog_generator(filename, CSNR, 200, 2, sigma)
#
# Plots.PlotSDR(CSNRdB, SDR_AWGN, SDR_Ray)
#
# print('Loading...')
#
# out_AWGN, out_Ray, out_delta_Ray, data_ray, CSNR, symbols = load(filename)
# print(np.shape(out_AWGN))
# print(np.shape(out_Ray))
# print(np.shape(out_delta_Ray))
