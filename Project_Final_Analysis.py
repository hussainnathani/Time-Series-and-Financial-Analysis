import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.linalg import toeplitz
from scipy.stats import t as t_dist, chi2
from ToolBox import manual_average_hstep_forecast, manual_naive_hstep, manual_drift_hstep_forecast, compute_gpac, \
    display_gpac_table, LM_algorithm, confidence_interval, poles_zeros, plot_sse, ggpac_table, plot_ggpac_table, \
    estimate_autocorrelations, estimate_acf, hgpac_table, levenberg_marquardt_bj, \
    confidence_intervals_bj, compute_error_bj, q_test, s_test, compute_residuals_from_lm, one_step_forecast_arma_plot, \
    h_step_forecast_arma_plot, forecast_arma, compute_residuals_bj, forecast_bj_1step, forecast_bj_hstep, ACF_PACF_Plot, \
    Cal_rolling_mean_var, run_adf_test, run_kpss_test, filter_significant_params, plt_hgpac_table, \
    compute_covariance_bj, evaluate_forecast, backward_elimination
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")  # Hides all warnings
np.seterr(all='ignore')

#%%

# 6. Description of the dataset

# Load data
df = pd.read_csv("D:\GWU\SEM 2\DATS 6313 (TS)\Project\cleaned_financial_dataset.csv", \
                 parse_dates=["date"], index_col="date")
series = df["gold close"]

# Selected columns
selected_columns = [
    "sp500 close", "nasdaq close", "CPI", "GDP", "silver close",
    "oil close", "platinum close", "palladium close", "gold open",
    "gold high", "gold low", "gold close", "gold volume"]

df = df[selected_columns]

#%%
# 6a. Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# 6d. Plot of the dependent variable (gold close)
plt.figure(figsize=(12, 6))
plt.plot(df["gold close"], label="Gold Close Price", color="blue")
plt.title("Gold Close Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

#%%
# 6e. ACF and PACF plot

ACF_PACF_Plot(df["gold close"].dropna(), lags=40, series_name = "Gold Price raw data")

#%%
# 6f. Correlation matrix with heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Financial Variables")
plt.show()

#%%%

# 6g. Train-Test Split (80/20)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")

#%%
# Stationarity

rolling_means = []
rolling_variances = []

Cal_rolling_mean_var(series, 'Gold Price')

run_adf_test(series)
print("\n")
run_kpss_test(series)

diff_series = series.diff().dropna()

Cal_rolling_mean_var(diff_series, '1st Order Differenced Gold Price Series')

print("\n")

run_adf_test(diff_series)
print("\n")
run_kpss_test(diff_series)


#%%

ACF_PACF_Plot(diff_series, lags=40, series_name = "Gold Price differenced data")

#%%
# 8. Time series Decomposition

period = 252

# ----------- Additive Decomposition ----------------
stl_add = STL(series, period=period, robust=True).fit()
trend_add = stl_add.trend
seasonal_add = stl_add.seasonal
resid_add = stl_add.resid

# ----------- Multiplicative Decomposition ----------------
log_series = np.log(series)
stl_mult = STL(log_series, period=period, robust=True).fit()
trend_mult = np.exp(stl_mult.trend)
seasonal_mult = np.exp(stl_mult.seasonal)
resid_mult = np.exp(stl_mult.resid)

# ----------- Plotting Decomposition ----------------
def plot_stl_components(original, trend, seasonal, resid, title):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(original, label='Original', color='blue')
    axes[0].legend(loc='best')
    axes[1].plot(trend, label='Trend', color='red')
    axes[1].legend(loc='best')
    axes[2].plot(seasonal, label='Seasonal', color='green')
    axes[2].legend(loc='best')
    axes[3].plot(resid, label='Residuals', color='black')
    axes[3].legend(loc='best')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_stl_components(series, trend_add, seasonal_add, resid_add, "Additive STL Decomposition")
plot_stl_components(series, trend_mult, seasonal_mult, resid_mult, "Multiplicative STL Decomposition")

# ----------- Trend and Seasonality Strength ----------------
def compute_strength(trend, seasonal, resid):
    var_resid = np.var(resid, ddof=1)
    trend_strength = max(0, 1 - (var_resid / np.var(trend + resid, ddof=1)))
    seasonality_strength = max(0, 1 - (var_resid / np.var(seasonal + resid, ddof=1)))
    return trend_strength, seasonality_strength

trend_strength_add, seasonality_strength_add = compute_strength(trend_add, seasonal_add, resid_add)
trend_strength_mult, seasonality_strength_mult = compute_strength(trend_mult, seasonal_mult, resid_mult)

print("\n")
print("Additive Decomposition:")
print(f"  Strength of Trend: {trend_strength_add:.4f}")
print(f"  Strength of Seasonality: {seasonality_strength_add:.4f}\n")

print("Multiplicative Decomposition:")
print(f"  Strength of Trend: {trend_strength_mult:.4f}")
print(f"  Strength of Seasonality: {seasonality_strength_mult:.4f}")

# Compare residual variance
var_resid_add = np.var(resid_add, ddof=1)
var_resid_mult = np.var(resid_mult, ddof=1)

print("\n")
print(f"Residual Variance (Additive): {var_resid_add:.4f}")
print(f"Residual Variance (Multiplicative): {var_resid_mult:.4f}")

#%%
# 9. Fit Holt-Winters model (seasonal period = 252 trading days/year)

holt_model = ExponentialSmoothing(
    train_df["gold close"],
    trend='mul',
    seasonal=None,
    initialization_method='estimated'
).fit(optimized=True)

forecast_hw = holt_model.forecast(len(test_df))
forecast_hw.index = test_df.index

mse = mean_squared_error(test_df["gold close"], forecast_hw)
mae = mean_absolute_error(test_df["gold close"], forecast_hw)
rmse = np.sqrt(mse)

print(f"Holt-Winters Forecast Evaluation:")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(train_df.index, train_df["gold close"], label='Train', color='blue')
plt.plot(test_df.index, test_df["gold close"], label='Test', color='black')
plt.plot(test_df.index, forecast_hw, label='Holt-Winters Forecast', color='green')
plt.title("Holt-Winters Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Gold Close Price")
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# 10. Select predictors (exclude gold close)

X = df.drop(columns=["gold close"])
y = df["gold close"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# VIF Analysis
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
print("\nVIF Analysis:")
print(vif_data)

# Condition Number
_, s, _, _ = np.linalg.lstsq(X_scaled, y.values, rcond=None)
condition_number = np.linalg.cond(X_scaled)
print(f"\nCondition Number: {condition_number:.2f}")

# PCA
pca = PCA()
pca.fit(X_scaled)
explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.legend()
plt.tight_layout()
plt.show()

# SVD (Singular Value Decomposition)
U, s_vals, VT = np.linalg.svd(X_scaled, full_matrices=False)
print("\nSingular Values (SVD):")
print(s_vals)

# Backward Stepwise Regression (basic version using AIC)
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

all_features = df.columns.drop("gold close")

# Exclude features related to 'gold open', 'gold high', 'gold low'
excluded_keywords = ["gold open", "gold high", "gold low"]
filtered_features = [col for col in all_features if col not in excluded_keywords]

X_filtered = df[filtered_features]
y = df["gold close"]

X_const = sm.add_constant(X_filtered)

be_model, final_features = backward_elimination(X_filtered, y)
print("\nFinal selected features:")
print(final_features)

#%%
# 11. Base Models

y = df["gold close"]

train_size = int(len(y) * 0.8)
train = y.iloc[:train_size]
test = y.iloc[train_size:]
h = len(test)

# 1. Average model
y_avg = manual_average_hstep_forecast(train, h)

# 2. Naïve model (last value)
y_naive = manual_naive_hstep(train, h)

# 3. Drift model
y_drift = manual_drift_hstep_forecast(train, h)

# 4. Simple Exponential Smoothing (SES)
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5, optimized=False)
y_ses = ses_model.forecast(h)

# --- Evaluation ---
def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def mae(true, pred):
    return mean_absolute_error(true, pred)

print("Base Model RMSEs and MAEs:")
print(f"  Average Model     : RMSE = {rmse(test, y_avg):.4f}, MAE = {mae(test, y_avg):.4f}")
print(f"  Naïve Model       : RMSE = {rmse(test, y_naive):.4f}, MAE = {mae(test, y_naive):.4f}")
print(f"  Drift Model       : RMSE = {rmse(test, y_drift):.4f}, MAE = {mae(test, y_drift):.4f}")
print(f"  SES Model         : RMSE = {rmse(test, y_ses):.4f}, MAE = {mae(test, y_ses):.4f}")

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train", color="blue")
plt.plot(test.index, test, label="Test", color="black")
plt.plot(test.index, y_avg, label="Average Forecast", linestyle="--")
plt.plot(test.index, y_naive, label="Naïve Forecast", linestyle="--")
plt.plot(test.index, y_drift, label="Drift Forecast", linestyle="--")
plt.plot(test.index, y_ses, label="SES Forecast", linestyle="--")
plt.title("Base Model Forecasts vs Actual")
plt.xlabel("Date")
plt.ylabel("Gold Close Price")
plt.legend()
plt.grid(True)
plt.show()

#%%
# 12. Multiple Linear Regression

# 12a: Model Training
X = df.drop(columns=["gold open", "gold high", "gold low", "gold close"])
y = df["gold close"]
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
ols_model = sm.OLS(y_train, X_train_const).fit()

# Step 12a: One-step ahead prediction
y_pred = ols_model.predict(X_test_const)

# 12b: F-test and t-tests
print(ols_model.summary())

# 12c: Cross-Validation for Time Series
tscv = TimeSeriesSplit(n_splits=5)
cv_mse = []
for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    model = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    y_pred_cv = model.predict(sm.add_constant(X_te))
    cv_mse.append(mean_squared_error(y_te, y_pred_cv))
cv_mse_mean = np.mean(cv_mse)
print(f"CV Mean MSE: {cv_mse_mean:.4f}")

# 12d: Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
aic = ols_model.aic
bic = ols_model.bic
r2 = ols_model.rsquared
r2_adj = ols_model.rsquared_adj
print("\n MLR Model Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"AIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {r2_adj:.4f}")

# 12e: ACF of Residuals
residuals = y_test - y_pred
ACF_PACF_Plot(residuals.dropna(), lags=40, series_name = "ACF of Residuals of MLR model")

# 12f: Q-statistic
ljung_box_result = acorr_ljungbox(residuals, lags=[20], return_df=True)
q_stat = ljung_box_result.loc[20, 'lb_stat']
p_val = ljung_box_result.loc[20, 'lb_pvalue']
print("Ljung-Box Q-Statistic (lag=20):", round(q_stat, 4))
print("p-value:", round(p_val, 4))

# 12g: Residual Mean and Variance
residual_mean = residuals.mean()
residual_variance = residuals.var()
print(f"Residual Mean: {residual_mean:.4f}")
print(f"Residual Variance: {residual_variance:.4f}")

# 12h Plot
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label='Train', color='blue')
plt.plot(y_test.index, y_test, label='Test', color='black')
plt.plot(y_test.index, y_pred, label='Predicted', color='green')
plt.title("MLR Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Gold Close Price")
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# 13.GPAC

y = diff_series

train_size = int(len(y) * 0.8)
y_train = y.iloc[:train_size]

acf_vals = sm.tsa.stattools.acf(y_train, nlags=50)

gpac_df = compute_gpac(acf_vals, max_j=7, max_k=7)
print("\nGPAC Table:")
print(gpac_df)
display_gpac_table(gpac_df)

ACF_PACF_Plot(diff_series, lags=40, series_name = "Gold Price differenced data")

#%%
# 14.

# ARMA Models

orders_to_try = [(1, 1), (2, 1), (2, 2), (2, 3), (3, 2), (4, 2)]

for ar_order, ma_order in orders_to_try:
    print(f"\n=== Trying ARMA({ar_order}, {ma_order}) ===")

    y = y_train  # y_train is the differenced series
    theta_est, SSE_history, cov_matrix = LM_algorithm(y, ar_order=ar_order, ma_order=ma_order)

    print("\nEstimated Parameters:")
    for i, est in enumerate(theta_est):
        label = f"AR[{i+1}]" if i < ar_order else f"MA[{i+1-ar_order}]"
        print(f"{label} = {est:.4f}")

    final_SSE = SSE_history[-1]
    N = len(y)
    num_params = len(theta_est)
    error_variance = final_SSE / (N - num_params)
    print(f"\nEstimated Variance of Error: {error_variance:.4f}")

    conf_int = confidence_interval(theta_est, cov_matrix)
    print("\nConfidence intervals:")
    for idx, ci in enumerate(conf_int):
        print(f"Param {idx+1}: [{ci[0]:.4f}, {ci[1]:.4f}]")

    sd = np.sqrt(np.diag(cov_matrix))
    print("\nStandard Deviation of Parameters:")
    print(np.round(sd, 4))

    theta_filtered, keep_indices, labels = filter_significant_params(theta_est, conf_int, ar_order)
    print("\nFiltered Parameters (CI excludes 0):")
    for lbl, val in zip(labels, theta_filtered):
        print(f"{lbl}: {val:.4f}")

    phi = [theta_est[i] for i in keep_indices if i < ar_order]
    theta = [theta_est[i] for i in keep_indices if i >= ar_order]

    poles, zeros = poles_zeros(phi, theta)
    print("\nPoles:", np.round(poles, 3))
    print("Zeros:", np.round(zeros, 3))

    plot_sse(SSE_history)

    e = compute_residuals_from_lm(y, phi, theta)
    q_test(e, lags=50, model_df=len(phi) + len(theta))
    ACF_PACF_Plot(e, lags=40, series_name="ACF of Residuals of ARMA model")

#%%

# ARIMA Models

orders_to_try = [(1,1,1), (2,1,1), (2,1,2), (3,1,1), (3,1,2), (3,1,3)]

for order in orders_to_try:
    model = ARIMA(df['gold close'], order=order)
    result = model.fit()
    print(f"ARIMA{order} — AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
    print("Confidence Intervals:")
    print(result.conf_int())

    residuals = result.resid
    p, d, q = order
    q_test(residuals, lags=50, model_df=p + q)


#%%

# Selected ARIMA (2,1,2)

final_ARIMA_model = ARIMA(df['gold close'], order=(2, 1, 2))
result = final_ARIMA_model.fit()

print(f"\nARIMA(2,1,2) — AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")

params = result.params
print("\nEstimated Parameters:")
for i, val in enumerate(params):
    print(f"{params.index[i]} = {val:.4f}")

stderr = result.bse
print("\nStandard Deviations:")
for i, val in enumerate(stderr):
    print(f"SD[{params.index[i]}] = {val:.4f}")

conf_int = result.conf_int()
print("\n95% Confidence Intervals:")
for i in range(len(conf_int)):
    lower, upper = conf_int.iloc[i]
    print(f"{conf_int.index[i]}: [{lower:.4f}, {upper:.4f}]")

residuals = result.resid

print("\n--- Q-Test for Residual Whiteness ---")
ljung_box = acorr_ljungbox(residuals, lags=[50], return_df=True)
Q_stat = ljung_box['lb_stat'].values[0]
p_val = ljung_box['lb_pvalue'].values[0]
print(f"Q-statistic: {Q_stat:.4f}")
print(f"p-value: {p_val:.4f}")
if p_val > 0.05:
    print("Result: Residuals are white (no autocorrelation).")
else:
    print("Result: Residuals show autocorrelation.")

ACF_PACF_Plot(residuals, lags=50, series_name="ACF of Residuals of ARIMA model")


#%%

# Selecting Input Feature for Box-jenkins (u)

excluded_keywords = ["gold open", "gold high", "gold low"]
filtered_features = [col for col in df.columns if col not in excluded_keywords]

X_filtered = df[filtered_features]

X_diff = X_filtered.diff().dropna()

correlation_matrix = X_diff.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of First-Differenced Features")
plt.tight_layout()
plt.show()

#%%
# 15. Box-Jenkins Model

input_var = 'silver close'  # Input feature (can also try 'platinum close', etc.)
raw_u = df[input_var].values

split = int(0.8 * len(y))

# Differencing input and output series
u = pd.Series(raw_u).diff().dropna().to_numpy()
y_mean = np.mean(diff_series[:split])
y = diff_series - y_mean

# Mean-center both series
u = u - np.mean(u[:split])
y = y - np.mean(y)

# Train/test split (80/20)
split = int(0.8 * len(y))
y_train, u_train = y[:split], u[:split]

K = 50
acf_u = estimate_acf(u, K)

Ru = toeplitz(acf_u[K:])

Ru_5x5 = Ru[:5, :5]
print("First 5x5 matrix of Ru(τ):")
print(np.round(Ru_5x5, 2))

print("\nShape of Ru(τ) matrix:", Ru.shape)

# Cross-correlation between input and output
Ru, Ruy = estimate_autocorrelations(u, y, K)

# Estimate impulse response coefficients (ĝ)
R_u_matrix = toeplitz(Ru[:K+1])
g_hat = np.linalg.solve(R_u_matrix, Ruy[:K+1])

print("First 5 estimated impulse response coefficients:")
print(np.round(g_hat[:5], 3))

# === G-GPAC: For u → y dynamics ===
acf_g = acf(g_hat, nlags=50, fft=False)

gpac_df_g = ggpac_table(acf_g, max_k=7, max_j=7)
print("G-GPAC Table (7x7) based on ĝ(k):\n")
print(gpac_df_g)
plot_ggpac_table(gpac_df_g)

# === H-GPAC: For y alone ===
acf_y = acf(y, nlags=50, fft=False)

gpac_df_h = hgpac_table(acf_y, max_k=7, max_j=7)
print("\nH-GPAC Table (7x7):")
print(gpac_df_h)
plt_hgpac_table(gpac_df_h)


#%%

# === Parameter Estimation and Diagnostics for Box-Jenkins Model ===

orders_to_try = [
    (1, 1, 1, 1),
    (2, 1, 2, 1),
    (2, 2, 2, 2),
    (2, 3, 3, 2),
    (3, 2, 2, 2)]

y = np.array(y).flatten()

for nb, nf, nc, nd in orders_to_try:
    print(f"\n=== Trying BJ Model with (nb={nb}, nf={nf}, nc={nc}, nd={nd}) ===")

    theta_init = np.zeros(nb + nf + nc + nd)

    # Estimate parameters using Levenberg-Marquardt
    theta_est, sse_track = levenberg_marquardt_bj(y, u, theta_init, nb, nf, nc, nd)

    ci = confidence_intervals_bj(theta_est, y, u, nb, nf, nc, nd)

    print("\nParameter Estimates with Std. Dev and 95% Confidence Intervals:")
    for i, (lb, est, ub) in enumerate(ci):
        std = (ub - lb) / (2 * 1.96)
        print(f"θ[{i}] = {est:.4f} | Std. Dev: {std:.4f} | CI: [{lb:.4f}, {ub:.4f}]")

    significant_indices = [i for i, (lb, est, ub) in enumerate(ci) if lb * ub > 0]  # CI does not include 0
    print("\nFiltered Significant Parameters (CI excludes 0):")
    for i in significant_indices:
        print(f"θ[{i}] = {theta_est[i]:.4f} | CI: [{ci[i][0]:.4f}, {ci[i][2]:.4f}]")

    b_params = [theta_est[i] for i in significant_indices if i < nb]
    f_params = [theta_est[i] for i in significant_indices if nb <= i < nb + nf]
    c_params = [theta_est[i] for i in significant_indices if nb + nf <= i < nb + nf + nc]
    d_params = [theta_est[i] for i in significant_indices if nb + nf + nc <= i]

    e = compute_error_bj(theta_est, y, u, nb, nf, nc, nd)

    # === Residual Diagnostics ===
    print("\n-- Q-Test (Residual Whiteness) --")
    model_df = nb + nf + nc + nd
    q_test(e, lags=50, model_df=model_df)

    print("\n-- S-Test (Residual-Uncorrelated-With-Input) --")
    s_test(e, u, theta_est, nb, nf)

#%%

# 16.

# ARMA(2,2) Final Selected Model

phi = [0.5160, 0.9614]
theta = [0.5145, 0.9625]

e = compute_residuals_from_lm(y, phi, theta)

split = int(0.8 * len(y))
y_train = y[:split]
y_test = y[split:]
e_train = e[:split]
h = len(y_test)

forecasts_diff = forecast_arma(y_train, phi, theta, e_train, steps=h)

# Invert differencing using last known actual
last_actual = series.iloc[split]
forecast_arma_original = np.cumsum(forecasts_diff) + last_actual

print("Variance of Residuals ARMA: ", np.var(e))
print("Variance of Forecast error ARMA: ",  np.var(y_test - forecast_arma_original))

#%%

# ARIMA(2,1,2) Final Selected Model

model = ARIMA(df['gold close'], order=(2, 1, 2))
result_arima = model.fit()
residuals = result_arima.resid

h = len(y_test)
forecast_arima = result_arima.forecast(steps=h)

y_test_aligned = np.array(y_test).flatten()
forecast_aligned = np.array(forecast_arima).flatten()

forecast_error_var = np.var(y_test_aligned - forecast_aligned)

print("Variance of Residuals ARIMA: ", np.var(residuals))
print("Variance of Forecast error ARIMA: ",  forecast_error_var)

#%%

# Box-jenkins (2,1,2,1) Final Selected Model

theta_bj = [-0.7306, -0.7199, -0.6864, 0.8737, 0.6782, -0.8603]
nb, nf, nc, nd = 2, 1, 2, 1

e_bj = compute_residuals_bj(theta_bj, y, u, nb, nf, nc, nd)

start = split
h = len(series[split:])
forecast_bj_diff = forecast_bj_hstep(y, u, e_bj, theta_bj, nb, nf, nc, nd, steps=h, start=start)

y_test_original = series.iloc[split:]
forecast_bj_diff = forecast_bj_diff + y_mean

# Invert differencing to original scale
last_actual_bj = series.iloc[start]
forecast_bj_original = y_test_original.iloc[0] + np.cumsum(-forecast_bj_diff) - (-forecast_bj_diff[0])
bias = np.mean(y_test_original[:h] - forecast_bj_original)
forecast_bj_corrected = forecast_bj_original + bias

print("Variance of Residuals Box-Jenkins: ", np.var(e_bj))
print("Variance of Forecast error Box-Jenkins: ",  np.var(y_test_original - forecast_bj_corrected))

#%%

# 17.

forecast_arma_original      # ARMA (2,2)
forecast_arima              # ARIMA (1,1,1)
forecast_bj_corrected       # Box-Jenkins (2,1,2,1)

#%%

# Comparing all models, based on Evaluation Metrics

y_test_original = series.iloc[split:]

results_arma = evaluate_forecast(
    y_test_original[:len(forecast_arma_original)],
    forecast_arma_original,
    num_params=len(phi) + len(theta))

results_arima = evaluate_forecast(
    y_test_original[:len(forecast_arima)],
    forecast_arima,
    num_params=3)

results_bj = evaluate_forecast(
    y_test_original[:len(forecast_bj_corrected)],
    forecast_bj_corrected,
    num_params=nb + nf + nc + nd)

comparison_df = pd.DataFrame({
    "ARMA (2,2)": results_arma,
    "ARIMA (2,1,2)": results_arima,
    "Box-Jenkins": results_bj
}).T.reset_index().rename(columns={"index": "Model"})

comparison_df = comparison_df.round(3)

# Format large/small numbers in scientific notation, else fixed-point
def format_cell(val):
    if isinstance(val, (float, int)):
        if abs(val) >= 1e5 or abs(val) < 1e-3:
            return f"{val:.3e}"
        else:
            return f"{val:.3f}"
    return str(val)

formatted_df = comparison_df.applymap(format_cell)
col_widths = [15] + [12] * (formatted_df.shape[1] - 1)
headers = formatted_df.columns.tolist()
header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
print(header_line)
for _, row in formatted_df.iterrows():
    row_line = "".join(str(val).ljust(w) for val, w in zip(row, col_widths))
    print(row_line)

#%%

# 19. Final Forecast Model using Selected Box Jenkins Model

# Plot actual vs forecast
plt.figure(figsize=(12, 6))
plt.plot(series.iloc[:split], label="Train", color="blue")
plt.plot(series.iloc[split:], label="Test (Actual)", color="black")
plt.plot(series.index[split:split+len(forecast_bj_corrected)], forecast_bj_corrected,
         label="Box-Jenkins Forecast", color="green", linestyle="--")

plt.title("Box-Jenkins Forecast vs Actual Gold Prices")
plt.xlabel("Date")
plt.ylabel("Gold Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%
print("END")
#%%