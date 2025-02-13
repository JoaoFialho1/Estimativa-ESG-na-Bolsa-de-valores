import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller as adf

def adf_test(Y, X, data, type='c', lag=1, alpha=0):
    print()
    print(" TESTE DE ESTACIONARIEDADE")
    print(" H₀: Não estacionaria")
    print(" H₁: Estacionária")
    print(f" Ao alpha de: {alpha*100:.2f}%")
    print()
    print("--------------------------------------")
    ADF = adf(x=data[Y], maxlag=lag, regression=type, autolag=None)

    print(f"Y: {Y}")
    print(f'ADF: {ADF[0]}')
    print(f'P-valor: {ADF[1]}')
    print(f'N° Lags: {ADF[2]}')
    if ADF[1] >= alpha:
        print(" SERIE NÃO ESTACIONÁRIA")
    else:
        print(" SERIE ESTACIONÁRIA")
    print()
    print("--------------------------------------")
    ADF = adf(x=data[X], maxlag=lag, regression=type, autolag=None)
    print(f"X: {X}")
    print(f'ADF: {ADF[0]}')
    print(f'P-valor: {ADF[1]}')
    print(f'N° Lags: {ADF[2]}')
    if ADF[1] >= alpha:
        print(" SERIE NÃO ESTACIONÁRIA")
    else:
        print(" SERIE ESTACIONÁRIA")
    print()
    print("--------------------------------------")
    reg = smf.ols(f'{Y} ~ {X}', data=data).fit()
    ADF = adf(x=reg.resid, maxlag=lag, regression=type, autolag=None)
    print("RESÍDUOS")
    print(f'ADF: {ADF[0]}')
    print(f'P-valor: {ADF[1]}')
    print(f'N° Lags: {ADF[2]}')
    print()
    if ADF[1] >= alpha:
        print(" SERIE DOS RESÍDUOS NÃO É ESTACIONÁRIA")
        print(" - Regressão espúria")
        print(" - Não há evidência de cointegração")
    else:
        print(" SERIE DOS RESÍDUOS É ESTACIONÁRIA")
        print(" - Regressão NÃO espúria")
        print(" - Há evidência de cointegração")
    print()
    print("--------------------------------------")