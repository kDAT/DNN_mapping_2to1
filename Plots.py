
import numpy as np
import matplotlib.pyplot as plt
import SpiralMapping


def PlotSpiral(CSNRdB, alpha):
    # contourf
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)

    x1v, x2v = np.meshgrid(x1, x2)

    # CSNRdB = 20
    CSNR = 10**(CSNRdB/10)
    delta = 2*np.pi*((6*0.16**2)/CSNR)**(1/4)
    # alpha = 2

    yt = []
    line = 0
    for x1i, x2i in zip(x1v, x2v):
        yl = []
        for x1ii, x2ii in zip(x1i, x2i):
            y = SpiralMapping.mapping(x1ii, x2ii, delta, alpha)
            yl.append(y)
        yt.append(yl)
        print('line ' + str(line))
        line += 1

    print('Mapping Completed')

    plt.contourf(x1, x2, yt, levels=100)
    plt.colorbar()
    plt.title('CSNR = 20 dB')
    plt.show()


def PlotSDR(CSNRdB, SDR_AWGN, SDR_Ray, SDR_DNN=None):
    plt.plot(CSNRdB, SDR_AWGN)
    plt.plot(CSNRdB, SDR_Ray)
    if SDR_DNN is not None:
        plt.plot(CSNRdB, SDR_DNN)
    plt.grid()
    plt.show()
