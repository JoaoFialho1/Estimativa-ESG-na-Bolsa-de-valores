from functions import adf_test, reg, graph
import pandas as pd

data = pd.read_excel("dados.xlsx")
data.set_index(data["temp"], inplace=True)
del data["temp"]

# Stationarity
print('TESTE 1')
adf_test(Y="ibov_mean", X="ise_mean", data=data, alpha=0.05)

var_perc = pd.DataFrame({f"ibov_mean_var_perc": data["ibov_mean"].pct_change()*100,
                         f"ise_mean_var_perc": data["ise_mean"].pct_change()*100
                         }).dropna()
print('\nTESTE 2')
adf_test(Y="ibov_mean_var_perc", X="ise_mean_var_perc",
         data=var_perc, alpha=0.05)

# Regression
print('\n\nREGRESSION\n')
print("Tamanho da amostra:", data.shape[0])
diff = pd.DataFrame({f"ibov_diff": data["ibov_mean"].pct_change(
)*100, f"ise_diff": data["ise_mean"].pct_change()*100}).dropna()
reg(Y="ibov_diff", X='ise_diff', data=diff)

# Graphs
graph()