import numpy as np
import matplotlib.pyplot as plt

def ideal_square_model(k, n):
    S0=np.zeros(n)
    for i in range(n):
        S0[i]=(k*i*i)
    return S0

def noised_square_model(k, n, dm, dsig):
    N = np.random.normal(dm, dsig, n)
    I = ideal_square_model(k, n)
    noised_model = [x + y for x, y in zip(N, I)]
    return noised_model

def plot_model(X):
    plt.plot(X)
    plt.show()

def LSM(S0):

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))

    for i in range(iter):  
        Yin[i, 0] = float(S0[i])  
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)

    print('Регресійна модель:')
    print('y(t) = ', C[0,0], ' + ', C[1,0], ' * t', ' + ', C[2,0], ' * t^2')

    return Yout, C[0,0], C[1,0], C[2,0]

def plot_elbow(SL, t=None, max_degree=6):
    n = len(SL)
    if t is None:
        t = np.arange(n)

    res = []
    for m in range(1, max_degree + 1):
        coefs = np.polyfit(t, SL, m)
        fitted = np.polyval(coefs, t)
        mse = np.sum((SL - fitted) ** 2) / n
        res.append(mse)

    plt.plot(range(1, max_degree + 1), res, marker='o')
    plt.xlabel("Degree of polynomial")
    plt.ylabel("MSE")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()

    return res

