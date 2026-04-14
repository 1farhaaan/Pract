---------------------------------------------------------------- Practical - 1 --------------------------------------------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
observed = np.array([5, 15, 28, 45, 52, 62, 70, 77, 82, 85, 92, 96])

def modified_exponential(t, a, b):
  y = a * np.exp(-b * t)
  return y

params, _ = curve_fit(modified_exponential, time, observed, p0=[100, 0.1])

print(params)
print(_)

a, b = params
print(f"Fitted parameters: a = {a:.2f}, b = {b:.2f}")

fitted = modified_exponential(time, a, b)
print(fitted)

plt.figure(figsize=(10,6))
plt.scatter(time, observed, label="Observed Data", color="blue", marker="o")
plt.plot(time, fitted, label=f"Fitted Curve y={a:.2f}(1{b:.2f}^t)", color="red", 
         linestyle="--")
plt.title("Modified Exponential curve")
plt.xlabel("Time")
plt.ylabel("Observed Data")
plt.legend()
plt.grid()
plt.show()
-------------------------------------------------------------------- Practical - 2 ----------------------------------------------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

np.random.seed(42)
x_data = np.linspace(0, 10, 100)
a_true, b_true, c_true = 100, 2, 0.5

def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

y_data = gompertz(x_data, a_true, b_true, c_true) + np.random.normal(0, 2, len(x_data))
y_data

popt, pcov = curve_fit(gompertz, x_data, y_data, p0=[a_true, b_true, c_true])

a_fit, b_fit, c_fit = popt
print(f"Fitted parametes: a = {a_fit:.2f}, b = {b_fit:.2f}, c = {c_fit:.2f}")

plt.figure(figsize=(10,6))
plt.scatter(x_data, y_data, label="Data", color="orange", marker="o", alpha=0.5)
plt.plot(x_data, gompertz(x_data, *popt), label="Fitted Gompertz Curve", color="red")
plt.title('Gompertz Curve Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
----------------------------------------------------------------- Practical - 3 -------------------------------------------------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed(42)
x_data = np.linspace(0, 10, 100)
L, k, x0 = 10, 1, 5

def logistic_curve(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

y_data = logistic_curve(x_data, L, k, x0) + np.random.normal(0, 0.5, size=len(x_data))
y_data

popt, pcov = curve_fit(logistic_curve, x_data, y_data, p0=[L, k, x0])
L_fit, k_fit, x0_fit = popt

x_fit = np.linspace(0, 10, 100)
y_fit = logistic_curve(x_fit, L=L_fit, k=k_fit, x0=x0_fit)
y_fit 

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data', color='green',alpha=0.6)
plt.plot(x_fit, y_fit, label=f'Fitted Logistic Curve\nL={L_fit:.2f}, k={k_fit:.2f}, x0={x0_fit:.2f}')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
--------------------------------------------------------------- Practical - 4 --------------------------------------------------------------
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = [200, 150, 170, 200, 220, 250, 270, 300, 320, 350, 400, 450,
        130, 160, 180, 210, 260, 280, 310, 330, 360, 410, 460, 480
]

month = pd.date_range(start='2024-01-01', periods=len(data), freq='M')

print(month)

df = pd.DataFrame({'Date': month, 'Sales': data})
df.set_index('Date', inplace=True)

print(df)

df['12-month-MA'] = df.Sales.rolling(window=12).mean()
df

plt.figure(figsize=(10,6))
plt.plot(df['Sales'], label='Orginal Data', marker='o')
plt.plot(df['12-month-MA'], label='12 Month MA (Trend)', linestyle='--')
plt.title('Fitting trend Using Moving Average')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()
---------------------------------------------------------------- Practical - 5 -------------------------------------------------------------------
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import statsmodels.api as sm

data = {
    'Month': pd.date_range(start='2025-01-01', periods=24, freq='M'),
    'Sales': [120, 130, 150, 170, 200, 220, 210, 200, 180, 160, 140, 130, 140, 150, 170,
    190, 170, 190, 220, 240, 230, 220, 200, 180]
}

df = pd.DataFrame(data)
df

df['Month_num'] = np.arange(1, len(df) + 1)
df

x = sm.add_constant(df['Month_num'])
model = sm.OLS(df['Sales'], x).fit()
df['Trend'] = model.predict(x)
df

df['Ratio'] = (df['Sales'] / df['Trend']) * 100
df

df['Month_only'] = df['Month'].dt.month
df

season_indices = df.groupby('Month_only')['Ratio'].mean()

seasonal_indices = (season_indices / season_indices.sum()) * 100
print('Seasonal indices: (Ratio-To-Trend)', round(seasonal_indices, 2))

plt.figure(figsize=(10,6))
plt.plot(seasonal_indices.index, seasonal_indices.values, marker='o')
plt.title('Seasonal Indices (ratio-to-trend)')
plt.xlabel('Month')
plt.ylabel('Seasonal Indices')
plt.grid()
plt.show()

plt.figure()
plt.plot(df['Trend'], df['Sales'], label='Actual', marker='o')
plt.plot(df['Trend'], df['Trend'], label='Trend', linestyle='--')
plt.title('Trend vs Actual Data')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.grid()
plt.show()
---------------------------------------------------------------- Practical - 6 -------------------------------------------------------------
import numpy as np 
import pandas as pd

data = {
    'Date': pd.date_range(start='2023-01-01', periods=24, freq='M'),
    'Sales': [120, 130, 150, 170, 190, 200, 220, 250, 160, 140, 130,
             140, 190, 110, 190, 170, 200, 160, 150, 230, 220, 250, 200, 180]
}

df = pd.DataFrame(data)
df

df['Month_only'] = df['Date'].dt.month
df

df['MA_12'] = df['Sales'].rolling(window=12).mean()
df['CMA'] = df['MA_12'].rolling(window=2).mean()
df

df['RMA'] = (df['Sales'] / df['CMA']) * 100

seasonal_indices = df.groupby('Month_only')['RMA'].mean()

seasonal_indices = (seasonal_indices / seasonal_indices.sum()) * 1200

print('Seasonal Indices:\n', round(seasonal_indices, 2))

plt.figure()
plt.bar(seasonal_indices.index, seasonal_indices.values)
plt.title('Seasonal Indices')
plt.xlabel('Month')
plt.ylabel('Seasonal Index')
plt.grid(axis='y')
plt.show()
----------------------------------------------------------------- Practical - 7 --------------------------------------------------------------
import numpy as np
import pandas as pd 

data = {
    'Year': [2022, 2023, 2024],
    'Q1': [200, 210, 220],
    'Q2': [250, 260, 270],
    'Q3': [180, 190, 200],
    'Q4': [220, 230, 240]
}

df = pd.DataFrame(data)
df

link_relative = []
for i in range(len(df)):
    row = []
    if i > 0:
        row.append((df.loc[i, 'Q1'] / df.loc[i - 1, 'Q4']) * 100)
    for j in range(1, 4):
        row.append((df.iloc[i, j] / df.iloc[i, j - 1]) * 100)
    link_relative.append(row)

link_rel = pd.DataFrame(link_relative, columns=['Q1', 'Q2', 'Q3', 'Q4'])
link_rel

avg_link = link_rel.mean()

sum_avg = avg_link.sum()
adjustment_factor = 400 / sum_avg
seasonal_indices = avg_link * adjustment_factor

print("Average Link Relatives:")
print(avg_link)
print("\nAdjusted Seasonal Indices:")
print(seasonal_indices)

plt.figure() 
plt.plot(seasonal_indices.index, seasonal_indices.values, marker='o')
plt.title("Seasonal Indices (Quarterly - Link Relatives Method)")
plt.xlabel("Quarter")
plt.ylabel("Seasonal Index")
plt.grid(True)
plt.show()
---------------------------------------------------------------- Practical - 8 --------------------------------------------------------------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt

def variant_diff_variance(data):
    n = len(data)
    if n < 2:
        raise ValueError('Data must contain at least two observations.')

    diff_sq = [(data[i + 1] - data[i]) ** 2 for i in range(n - 1)]
    variance = (1 / (2 * (n - 1))) * sum(diff_sq)

    return variance

data = [10, 12, 15, 16, 12, 17,14]
var = variant_diff_variance(data)

print('Estimated Variance:',round(var, 2))
---------------------------------------------------------------- Practical - 9 ------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing


np.random.seed(42)
date_range = pd.date_range(start="2020-01-01", periods=50, freq="M")
data = np.cumsum(np.random.randn(50)) + 100  # Random walk with drift
df = pd.DataFrame(data, index=date_range, columns=["Value"])


plt.figure(figsize=(10, 5))
plt.plot(df, label="Original Data")
plt.title("Time Series Data")
plt.legend()
plt.show()

model_ses = SimpleExpSmoothing(df["Value"]).fit(smoothing_level=0.2, optimized=False)
df["SES_Forecast"] = model_ses.fittedvalues

model_holt = ExponentialSmoothing(df["Value"], trend="add").fit()
df["Holt_Forecast"] = model_holt.fittedvalues

model_holt_winters = ExponentialSmoothing(df["Value"], trend="add", seasonal="add", seasonal_periods=12).fit()
df["Holt_Winters_Forecast"] = model_holt_winters.fittedvalues


plt.figure(figsize=(12, 6))
plt.plot(df["Value"], label="Original Data", marker="o")
plt.plot(df["SES_Forecast"], label="SES Forecast", linestyle="--")
plt.plot(df["Holt_Forecast"], label="Holt Forecast", linestyle="--")
plt.plot(df["Holt_Winters_Forecast"], label="Holt-Winters Forecast", linestyle="--")
plt.title("Exponential Smoothing Forecasting")
plt.legend()
plt.show()
----------------------------------------------------------------- Practical - 10 -------------------------------------------------------------------------------------------------
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)
data = np.cumsum(np.random.randn(100))

window = 5
moving_avg = pd.Series(data).rolling(window=window).mean()

plt.plot(data, label='Actual')
plt.plot(moving_avg, label=f'Moving Average (Window={window})')
plt.legend()
plt.grid()
plt.show()

ses_model = SimpleExpSmoothing(data). fit(smoothing_level=0.2, optimized=False)
forecast = ses_model.forecast(10)

plt.plot(data, label='Actual')
plt.plot(ses_model.fittedvalues, label='SES Fitted', linestyle='--')
plt.plot(range(len(data), len(data) + 10), forecast, label='Forecast', 
         linestyle='dotted')
plt.legend()
plt.grid()
plt.show()

model = ARIMA(data, order=(2,1,2))
arima_fit = model.fit()

forecast = arima_fit.forecast(steps=10)

plt.plot(data, label='Actual')
plt.plot(range(len(data), len(data) + 10), forecast, label='Forecast', linestyle='dotted')
plt.legend()
plt.grid()
plt.show()
