import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


def prepare_series(model, freq=None):
    s = pd.Series(np.asarray(model).flatten())
    if freq:
        s.index = pd.date_range(start='2020-01-01', periods=len(s), freq=freq)
    return s


def decompose_series(series, period, seasonal=7, robust=True):
    stl = STL(series, period=period, seasonal=seasonal, robust=robust)
    return stl.fit()


def build_components_df(decomposition):
    df = pd.DataFrame({
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'observed': decomposition.observed
    })
    return df.dropna()


def plot_decomposition(series, result):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 4))
    ax1.plot(series.index, result.trend, label='Trend', color='red')
    ax1.set_title('Trend Component')
    ax2.plot(series.index, result.seasonal, label='Seasonal', color='blue')
    ax2.set_title('Seasonal Component')
    ax3.plot(series.index, result.resid, label='Residual')
    ax3.set_title('Residual Component')
    plt.tight_layout()
    plt.show()


def deseasonalize(series, result):
    deseasonalized_series = series - result.seasonal
    plt.figure(figsize=(7, 4))
    plt.plot(series.index, series, label='Original Time Series', color='blue')
    plt.plot(series.index, deseasonalized_series, label='Deseasonalized Time Series', color='green')
    plt.title('Original vs. Deseasonalized Time Series')
    plt.xlabel('Year')
    plt.ylabel('Data')
    plt.legend()
    plt.show()
    return deseasonalized_series


def compute_correlation_matrix(df):
    return df.corr()


def compute_covariance_matrix(df):
    return df.cov()


def plot_correlation_matrix(corr_matrix):
    plt.figure(figsize=(6, 5))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, round(corr_matrix.iloc[i, j], 2), ha='center', va='center')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_rolling_correlation(df, col1, col2, window=20):
    rolling_corr = df[col1].rolling(window).corr(df[col2])
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_corr, label=f'Rolling Correlation ({window})')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Correlation')
    plt.title(f'Rolling Correlation between {col1} and {col2}')
    plt.legend()
    plt.show()


def run_correlation_analysis(model, period, seasonal=7, robust=True, window=20):
    series = prepare_series(model)
    decomposition = decompose_series(series, period, seasonal, robust)
    plot_decomposition(series, decomposition)
    deseasonalize(series, decomposition)
    df = build_components_df(decomposition)
    corr_matrix = compute_correlation_matrix(df)
    cov_matrix = compute_covariance_matrix(df)
    print(corr_matrix)
    print(cov_matrix)
    plot_correlation_matrix(corr_matrix)
    plot_rolling_correlation(df, 'trend', 'residual', window)
    plot_rolling_correlation(df, 'seasonal', 'residual', window)

