import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import scipy.stats as sc

data = pd.read_excel("dados.xlsx")
data.set_index(data["temp"], inplace=True)
del data["temp"]

def graph():
    #GRAFICO DE INDICES
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(data.index, data['ibov_mean'], label="ISE B3", color='black', linestyle='dashed', marker = "D", alpha = 0.7)
    ax1.set_xticks(range(len(data.index)))
    ax1.set_xticklabels(data.index, rotation=75)
    ax1.tick_params('y', labelcolor='black')
    ax2 = ax1.twinx()
    ax2.plot(data.index, data["ise_mean"], label="IBOVESPA", color='darkred', marker = "o", alpha = 0.7) 
    ax2.tick_params('y', labelcolor='darkred')
    plt.title("Índice de Sustentabilidade Empresarial x Índice Bovespa\n(2023 - 2024)", fontweight='bold')
    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 1))
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    plt.show()

    #GRÁFICO DE VARIAÇÃO PERCENTUAL
    diff = pd.DataFrame({f"ibov_diff": data["ibov_mean"].pct_change()*100, f"ise_diff": data["ise_mean"].pct_change()*100}).dropna()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(diff.index, diff[f"ibov_diff"], label = f"Δ% IBOVESPA", color='black',
                linestyle='dashed', marker = "D", alpha = 0.7)
    ax1.set_xticks(range(len(data.index)))
    ax1.set_xticklabels(data.index, rotation=75)
    ax1.tick_params('y', labelcolor='black')
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax2 = ax1.twinx()
    ax2.plot(diff.index, diff[f"ise_diff"], label = f"Δ% ISE B3", color='darkred',
                marker = "o", alpha = 0.7)
    ax2.set_yticklabels([])
    y1_lim = ax1.get_ylim()
    y2_lim = ax2.get_ylim()
    common_lim = (min(y1_lim[0], y2_lim[0]), max(y1_lim[1], y2_lim[1]))
    ax1.set_ylim(common_lim)
    ax2.set_ylim(common_lim)
    ax1.axhline(y=0, color='black', linewidth=1, linestyle='--')
    plt.title("Variação percentual dos índices\n(2023 - 2024)", fontweight='bold')
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 1))
    plt.show()

    #GRÁFICO DE DISPERSÃO
    reg = smf.ols('np.log(ibov_mean) ~ np.log(ise_mean)', data=data).fit()
    b1, b2 = reg.params
    data_ftd = data.sort_values(by=['ise_mean'])
    X = sm.add_constant(data_ftd['ise_mean'])
    preds = reg.get_prediction(X)
    pred_summary = preds.summary_frame(alpha=0.05)

    plt.figure(figsize=(13, 5))
    plt.scatter(data_ftd['ise_mean'], data_ftd["ibov_mean"], label='Original datas', color="black", facecolors='none')
    plt.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    plt.xlabel('ISE B3')
    plt.ylabel('IBOVESPA')
    plt.plot(data_ftd['ise_mean'], np.exp(b1 + b2 * np.log(data_ftd['ise_mean'])), label = "Estimated regression")
    plt.fill_between(data_ftd['ise_mean'], np.exp(pred_summary['mean_ci_lower']), np.exp(pred_summary['mean_ci_upper']), color='gray', alpha=0.2, label='95% Confidence Interval')

    plt.legend()
    plt.show()