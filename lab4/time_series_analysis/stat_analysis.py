import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import warnings

import math as mt


def ts_stat_characteristics(SL, text):

    Yout = LSM_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))

    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)

    print(text)
    print('кількість елементів вбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS, '\n')


    plt.hist(SL0, bins=50, facecolor="blue", alpha=0.5)
    plt.show()

    return


def LSM_Stat_characteristics (S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))

    for i in range(iter): 
        Yin[i, 0] = float(S0[i])  
        F[i, 1] = float(i)
        F[i, 2] = float(i ** 2)
        F[i, 3] = float(i ** 3)
        F[i, 4] = float(i ** 4)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)

    return Yout


def dfuller_test_1(data_np):
    '''
    Оцінювання стаціонарності за Дікі-Фуллера (ADF)
    '''

    print("Results of Dickey-Fuller Test_1:")

    series = pd.Series(data_np)


    result = adfuller(series)

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()

    plt.plot(series, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.legend()
    plt.title('Rolling Statistics')
    plt.show()

    return

def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):
    iter = len(S0)
    j_Wind=mt.ceil(iter-n_Wind)+1
    S0_Wind=np.zeros((n_Wind))

    j = 0
    for i in range(n_Wind):
        l = (j + i)
        S0_Wind[i] = S0[l]
        dS_standart = np.var(S0_Wind)
        scvS_standart = mt.sqrt(dS_standart)


    for j in range(j_Wind):
        for i in range(n_Wind):
            l=(j+i)
            S0_Wind[i] = S0[l]

        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)

        if scvS > (Q * scvS_standart):
            S0[l] = mS

    return S0