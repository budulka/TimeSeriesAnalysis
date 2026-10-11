import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf


def fit_lognormal_model(series, degree=1):
    '''
    Модель: log y(t) = тренд(t) + e(t),
    тренд — поліном МНК, e(t) — скошений нормальний закон (skew-normal).
    '''
    y = np.asarray(series, dtype=float)
    t = np.arange(len(y))
    log_y = np.log(y)

    coefs = np.polyfit(t, log_y, degree)
    resid = log_y - np.polyval(coefs, t)
    a, loc, scale = stats.skewnorm.fit(resid)

    params = {'coefs': coefs, 'a': a, 'loc': loc, 'scale': scale}
    print('Тренд log y(t), коефіцієнти МНК:', coefs)
    print(f'Шум skew-normal: a={a:.3f}, loc={loc:.3f}, scale={scale:.3f}')
    return params


def generate_synthetic(params, n, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = stats.skewnorm.rvs(params['a'], params['loc'], params['scale'], size=n, random_state=rng)
    return np.exp(np.polyval(params['coefs'], t) + noise)


def describe_series(y):
    y = np.asarray(y, dtype=float)
    return {
        'середнє': np.mean(y),
        'медіана': np.median(y),
        'СКВ': np.std(y),
        'мінімум': np.min(y),
        'максимум': np.max(y),
        'асиметрія': stats.skew(y),
        'середнє log': np.mean(np.log(y)),
        'СКВ log': np.std(np.log(y)),
        'ACF(1)': acf(y, nlags=1)[1],
        'ADF p-value': adfuller(y)[1],
    }


def compare_with_real(real, synthetic):
    table = pd.DataFrame({'Реальні дані': describe_series(real),
                          'Синтетична модель': describe_series(synthetic)})
    table['Відхилення, %'] = (table['Синтетична модель'] - table['Реальні дані']) / table['Реальні дані'].abs() * 100
    table.loc[['ACF(1)', 'ADF p-value'], 'Відхилення, %'] = np.nan

    ks = stats.ks_2samp(np.log(real), np.log(synthetic))
    print(f'KS-тест (розподіли log y): statistic={ks.statistic:.4f}, p-value={ks.pvalue:.4f}')
    print('Розподіли статистично не відрізняються' if ks.pvalue > 0.05 else 'Розподіли статистично відрізняються')
    return table


def plot_real_vs_synthetic(real, synthetic):
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    ax[0].plot(real, linewidth=0.8)
    ax[0].set_title('Реальні дані')
    ax[1].plot(synthetic, linewidth=0.8, color='tab:orange')
    ax[1].set_title('Синтетична модель')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(min(np.log(real).min(), np.log(synthetic).min()),
                       max(np.log(real).max(), np.log(synthetic).max()), 40)
    ax[0].hist(np.log(real), bins=bins, alpha=0.6, label='Реальні дані')
    ax[0].hist(np.log(synthetic), bins=bins, alpha=0.6, label='Синтетична модель')
    ax[0].set_title('Гістограма log y')
    ax[0].legend()

    lags = np.arange(1, 31)
    ax[1].bar(lags - 0.2, acf(real, nlags=30)[1:], width=0.4, label='Реальні дані')
    ax[1].bar(lags + 0.2, acf(synthetic, nlags=30)[1:], width=0.4, label='Синтетична модель')
    ax[1].axhline(0, color='black', linewidth=0.8)
    ax[1].set_title('Автокореляційна функція')
    ax[1].legend()
    plt.tight_layout()
    plt.show()
