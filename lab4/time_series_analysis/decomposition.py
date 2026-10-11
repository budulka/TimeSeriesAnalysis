
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def decompose(series, period, seasonal, robust):
    from statsmodels.tsa.seasonal import STL
    stl = STL(series, period = period, seasonal=seasonal, robust=robust)
    result = stl.fit()
    return result

def plot_decomposition(result):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 4))

    ax1.plot(result.trend, label='Trend', color='red')
    ax1.set_title('Trend Component')
    ax2.plot(result.seasonal, label='Seasonal', color='blue')
    ax2.set_title('Seasonal Component')
    ax3.plot(result.resid, label='Residual')
    ax3.set_title('Residual Component')
    plt.tight_layout()
    plt.show()

def deseasonalize(series, result):
    deseasonalized = series - result.seasonal
    plt.figure(figsize=(7, 4))
    plt.plot(df.index, series, label='Original Time Series', color='blue')
    plt.plot(df.index, deseasonalized_series, label='Deseasonalized Time Series', color='green')
    plt.title('Original vs. Deseasonalized Time Series')
    plt.xlabel('Year')
    plt.ylabel('Data')
    plt.legend()
    plt.show()
    
    