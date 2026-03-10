import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

# Dataset
df = pd.read_csv('cleaned_financial_dataset.csv', parse_dates=['date'], index_col='date')

selected_columns = [
    "sp500 close", "nasdaq close", "CPI", "GDP", "silver close",
    "oil close", "platinum close", "palladium close", "gold open",
    "gold high", "gold low", "gold close", "gold volume"
]

df = df[selected_columns]

print(df.isnull().sum())

time_series = df['gold close']


# Plotting time series
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Gold Close Price', color='blue')
plt.title('Gold Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()


# Calculating rolling mean and variance
rolling_mean = time_series.rolling(window=30).mean()
rolling_var = time_series.rolling(window=30).var()


# Plotting rolling mean and rolling variance
plt.figure(figsize=(12, 6))
plt.plot(time_series, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_var, color='black', label='Rolling Variance')
plt.title('Rolling Mean & Rolling Variance')
plt.xlabel('Date')
plt.ylabel('Gold Close Price')
plt.legend()
plt.grid()
plt.show()


# Perform ADF test
def adf_test(timeseries):
    result = adfuller(timeseries.dropna())
    print("\nAugmented Dickey-Fuller Test:")
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    if result[1] > 0.05:
        print("Conclusion: The time series is non-stationary.")
    else:
        print("Conclusion: The time series is stationary.")
adf_test(time_series)


# Decompose the time series to find trend and seasonality
decomposition = STL(time_series, period=252)
res = decomposition.fit()

trend = res.trend
seasonal = res.seasonal
residual = res.resid

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axes[0].plot(time_series, label='Original', color='blue')
axes[0].legend(loc='best')
axes[1].plot(trend, label='Trend', color='red')
axes[1].legend(loc='best')
axes[2].plot(seasonal, label='Seasonality', color='green')
axes[2].legend(loc='best')
axes[3].plot(residual, label='Residuals', color='black')
axes[3].legend(loc='best')
plt.suptitle('Time Series Decomposition')
plt.show()

# Compute trend strength
var_residuals = np.var(residual, ddof=1)
var_total = np.var(trend + residual, ddof=1)
strength_of_trend = max(0, 1 - (var_residuals / var_total))

print(f"The strength of trend for this data set is {strength_of_trend:.4f}")

# Compute seasonality strength
var_residuals = np.var(residual, ddof=1)
var_total = np.var(seasonal + residual, ddof=1)
strength_of_seasonality = max(0, 1 - (var_residuals / var_total))

print(f"The strength of seasonality for this data set is {strength_of_seasonality:.4f}\n")

# ACF Function
def compute_acf(data, lag):
    n = len(data)
    data_mean = np.mean(data)
    data_centered = data - data_mean

    if lag >= 0:
        numerator = sum(data_centered[i] * data_centered[i + lag] for i in range(n - lag))
    else:
        numerator = sum(data_centered[i] * data_centered[i + lag] for i in range(-lag, n))

    denominator = sum(val ** 2 for val in data_centered)

    return numerator / denominator if denominator != 0 else 0

data = time_series
lags_a = 40

# Calculating ACF
acf_values_a = [round(compute_acf(data, lag), 2) for lag in range(-lags_a, lags_a + 1)]
lag_range_a = list(range(-lags_a, lags_a + 1))

print("ACF Values of a:", acf_values_a)
print("Lags for a:", lag_range_a)

plt.figure(figsize=(8, 5))
plt.stem(lag_range_a, acf_values_a)
plt.axhline(y=1.96/np.sqrt(len(data)), linestyle='--', color='red', label='95% Confidence Interval')
plt.axhline(y=-1.96/np.sqrt(len(data)), linestyle='--', color='red')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) for Closing Price of Gold')
plt.grid(True)
plt.show()


# Visualize correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)
plt.title('Correlation Matrix of Financial Variables')
plt.show()
