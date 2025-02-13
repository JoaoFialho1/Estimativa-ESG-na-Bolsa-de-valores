import statsmodels.formula.api as smf
import scipy.stats as sc
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import numpy as np
import seaborn as sns

def reg(Y, X, data, alpha = 0):
    parametros = X.count(' + ') + 2
    if parametros > 2:
        X = [x.strip().replace('"', '').replace("'", '') for x in X.split(' + ')]
    else:
        #____Remover outliers identificados pelo Cook's Distance
        import statsmodels.api as sm
        Xr = sm.add_constant(data[X])
        model = sm.OLS(data[Y], Xr).fit()
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        data['cooks_d'] = cooks_d
        data = data[data['cooks_d'] <= 4 / len(data)]
        outliers_cooksd = data[data['cooks_d'] >= 4 / len(data)]
        print("OUTLIERS identificados, e removidos, pela Distância de Cook:", outliers_cooksd.index.tolist())
    #____FUNCAO
    reg = smf.ols(f"{Y} ~ {X}", data=data).fit()
    for i in range(0, parametros):
        locals()[f'β{i}'] = reg.params.iloc[i]
    sinais = {}
    for i in range(0, parametros):
        if locals()[f'β{i}'] > 0:
            sinais[f'β{i}'] = "+"
        else:
            sinais[f'β{i}'] = "-"
    funcao_str = f"Y = {locals()['β0']:.4f}"
    for i in range(1, parametros):
        funcao_str += f" {sinais[f'β{i}']} {abs(locals()[f'β{i}']):.4f} * ({X[i-1]})"
    print("--------------------------------------")
    print("FUNÇÃO")
    print()
    print(funcao_str)
    print()
    print(f"| σ(y) = {(reg.ssr/reg.df_resid)**0.5:.4f}")
    for i in range(0, parametros):
        print(f"| σ(β{i}) = {reg.bse.iloc[i]:.4f}")
    print()

    #_____ESTATISTICAS

    print("--------------------------------------")
    print("ESTATÍSTICAS")
    print()
    print(f"|  R² = {reg.rsquared:.4f}")
    print(f"|  R̅² = {reg.rsquared_adj:.4f}")
    print(f"|  AIC = {reg.aic:.4f}")
    print(f"|  BIC = {reg.bic:.4f}")
    print()
    print(f" - O R² é estatisticamente explicativo a um nivel de significância maior que {100*reg.f_pvalue:.2f}%")
    print(f"|  F = {reg.fvalue:.4f}")
    print(f"|  Pval(F) = {reg.f_pvalue:.4f}")
    print()
    print("--------------------------------------")
    print("TESTE T")
    print()
    for i in range(0, parametros):
        if i == 0:
            print(f" - O β{i} (intercepto) é estatisticamente explicativo a um nivel de significância maior que {100*reg.pvalues.iloc[i]:.2f}%")
        else:
            print(f" - O β{i} (associado a {X[i-1]}) é estatisticamente explicativo a um nivel de significância maior que {100*reg.pvalues.iloc[i]:.2f}%")
        print(f"|  T = {reg.tvalues.iloc[i]:.4f}")
        print(f"|  Pval(T) = {reg.pvalues.iloc[i]:.4f}")
        print()
    print("--------------------------------------")
    print("NORMALIDADE DOS RESÍDUOS")
    print()

    if alpha > 0:        
        if sc.shapiro(reg.resid)[1] <= alpha:
            print(f" - Ao alpha de {alpha*100:.2f}% os resíduos NÃO são normalmente distribuidos")
        else:
            print(f" - Ao alpha de {alpha*100:.2f}% os resíduos são normalmente distribuidos")
    print(f"|  Shapiro = {sc.shapiro(reg.resid)[0]:.4f}")
    print(f"|  Pval = {sc.shapiro(reg.resid)[1]:.4f}")
    print()

    if alpha > 0:        
        if sc.kstest(reg.resid, 'norm')[1] <= alpha:
            print(f" - Ao alpha de {alpha*100:.2f}% os resíduos NÃO são normalmente distribuidos")
        else:
            print(f" - Ao alpha de {alpha*100:.2f}% os resíduos são normalmente distribuidos")
    print(f"|  Kolmogorov = {sc.kstest(reg.resid, 'norm')[0]:.4f}")
    print(f"|  Pval = {sc.kstest(reg.resid, 'norm')[1]:.4f}")
    print()

    if alpha > 0:        
        if sc.jarque_bera(reg.resid)[1] <= alpha:
            print(f" - Ao alpha de {alpha*100:.2f}% os resíduos NÃO são normalmente distribuidos")
        else:
            print(f" - Ao alpha de {alpha*100:.2f}% os resíduos são normalmente distribuidos")
    print(f"|  Jarque = {sc.jarque_bera(reg.resid)[0]:.4f}")
    print(f"|  Pval = {sc.jarque_bera(reg.resid)[1]:.4f}")
    print()
    print("--------------------------------------")
    print()

    if alpha > 0:
        print(f"Ao alpha de: {alpha*100:.2f}%")
        if reg.f_pvalue <= alpha:
            print("- R² é estatisticamente explicativo")
        else:
            print("- R² não é estatisticamente explicativo")

        for i in range(1, parametros):
            if reg.pvalues.iloc[i] <= alpha:
                print(f"- {X[i-1]} tem efeito significativo sobre {Y}")
            else:
                print(f"- {X[i-1]} não tem efeito significativo sobre {Y}")

    #____GRÁFICOS DE RESÍDUOS
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    # QQ-Plot
    QQplot = ProbPlot(reg.get_influence().resid_studentized_internal)
    fig_qq = QQplot.qqplot(line=None, alpha=0.5, ax=axs[0, 0],
                            marker='o', markerfacecolor='none', markeredgecolor='black')
    x = np.linspace(min(QQplot.theoretical_quantiles), max(QQplot.theoretical_quantiles))
    axs[0, 0].plot(x, x, color='red', linewidth=0.5)
    axs[0, 0].set_title('Normal Q-Q plot')
    axs[0, 0].set_xlabel('Theoretical Quantiles')
    axs[0, 0].set_ylabel('Standardized Residuals')
    axs[0, 0].grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)

    # Residuals vs Fitted Plot
    sns.residplot(x=reg.fittedvalues, y=reg.resid, data=data, lowess=True, line_kws={'color': 'red', 'lw': 0.5},
                scatter_kws={'facecolors': 'none', 'edgecolors': 'black'}, ax=axs[0, 1])
    axs[0, 1].set_title('Residuals vs Fitted')
    axs[0, 1].set_xlabel('Fitted values')
    axs[0, 1].set_ylabel('Residuals')
    axs[0, 1].grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)

    # Scale-Location Plot
    sns.regplot(x=reg.fittedvalues, y=np.sqrt(np.abs(reg.get_influence().resid_studentized_internal)), 
                scatter=True, lowess=True, line_kws={'color': 'red', 'lw': 0.5}, scatter_kws={'facecolors': 'none', 'edgecolors': 'black'}, ax=axs[1, 0])
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('√|Standardized residuals|')
    axs[1, 0].grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)

    # Leverage Plot
    if parametros <= 2:
        leverage = influence.hat_matrix_diag
        resid_studentized = influence.resid_studentized_internal
        def one_line(x):
            return np.sqrt((1 * len(model.params) * (1 - x)) / x)
        def four_line(x):
            return np.sqrt((4 * len(model.params) * (1 - x)) / x)
        def show_cooks_distance_lines(tx, inc, color, label, ax):
            ax.plot(inc, tx(inc), label=label, color=color, ls='--')

        sns.scatterplot(x=leverage, y=resid_studentized, legend=False, alpha=1, edgecolor='black', facecolor='none', ax=axs[1, 1])
        sns.regplot(x=leverage, y=resid_studentized, scatter=False, lowess=True, line_kws={'color': 'red', 'lw': 0.5}, ax=axs[1, 1])
        show_cooks_distance_lines(one_line, np.linspace(.01, .14, 100), 'gray', 'Cook\'s Distance (D=1); (D=4)', axs[1, 1])
        show_cooks_distance_lines(four_line, np.linspace(.01, .14, 100), 'gray', '', axs[1, 1])
        axs[1, 1].axhline(y=0, color='gray', linestyle='dotted', linewidth=0.6)
        axs[1, 1].set_xlabel('Leverage')
        axs[1, 1].set_ylabel('Standardized Resíduals')
        axs[1, 1].set_title('Leverage vs Residuals')
        axs[1, 1].legend()
        axs[1, 1].grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()