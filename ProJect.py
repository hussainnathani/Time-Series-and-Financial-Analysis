from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Load the dataset
df = pd.read_csv('financial_regression.csv', parse_dates=['date'], index_col='date')

selected_columns = [
    "sp500 close", "nasdaq close", "CPI", "GDP", "silver close",
    "oil close", "platinum close", "palladium close", "gold open",
    "gold high", "gold low", "gold close", "gold volume"
]

df = df[selected_columns]

df['gold close'] = df['gold close'].replace([float('inf'), float('-inf')], pd.NA)
df['gold close'] = df['gold close'].ffill()

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print("✅ The series is stationary (reject the null hypothesis).")
    else:
        print("❌ The series is non-stationary (fail to reject the null hypothesis).")

adf_test(df['gold close'])

import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr()

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)
plt.title('Correlation Matrix of Financial Variables')
plt.show()
