import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import toeplitz
from scipy.signal import dlti, dlsim
from scipy.stats import t as t_dist, chi2
from numpy.linalg import inv, LinAlgError
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, acf, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor




def ADF_Cal(x, column_name):
    result = adfuller(x)

    print(f"\nADF Test for {column_name}:")
    print("ADF Statistic: %.2f" % result[0])
    print("p-value: %.2f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print('\t%s: %.2f' % (key, value))

    if result[1] <= 0.05:
        print(f"{column_name} is stationary (p-value <= 0.05).")
    else:
        print(f"{column_name} is not stationary (p-value > 0.05).")

    return result

def run_adf_test(ts):
    result = adfuller(ts.dropna())
    print("ADF Test:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values: {result[4]}")
    if result[1] < 0.05:
        print("  Conclusion: Stationary (reject H0)")
    else:
        print("  Conclusion: Non-stationary (fail to reject H0)")

def run_kpss_test(ts):
    result = kpss(ts.dropna(), regression='c', nlags="auto")
    print("KPSS Test:")
    print(f"  KPSS Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values: {result[3]}")
    if result[1] > 0.05:
        print("  Conclusion: Stationary (fail to reject H0)")
    else:
        print("  Conclusion: Non-stationary (reject H0)")

# Function for Rolling mean and variance
def rolling_mean_var(data, column_name):

    rolling_means = []
    rolling_variances = []

    print(f"Calculating rolling mean and variance for {column_name}...")
    print("Sample Size | Rolling Mean | Rolling Variance")

    for i in range(1, len(data) + 1):
        current_data = data[column_name][:i]
        rolling_mean = current_data.mean()
        rolling_variance = current_data.var(ddof=0)

        rolling_means.append(rolling_mean)
        rolling_variances.append(rolling_variance)

        print(f"{i:11} | {rolling_mean:.4f}    | {rolling_variance:.4f}")

def Cal_rolling_mean_var(data, series_name="Series"):
    rolling_means = []
    rolling_variances = []

    print(f"\nCalculating rolling mean and variance for {series_name}...")

    for i in range(1, len(data) + 1):
        current_data = data[:i]
        rolling_mean = current_data.mean()
        rolling_variance = current_data.var(ddof=0)
        rolling_means.append(rolling_mean)
        rolling_variances.append(rolling_variance)

    print(f"Initial Mean: {rolling_means[0]:.4f}, Final Mean: {rolling_means[-1]:.4f}")
    print(f"Initial Var : {rolling_variances[0]:.4f}, Final Var : {rolling_variances[-1]:.4f}")

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(data) + 1), rolling_means, label="Rolling Mean", color="blue")
    plt.title(f"Rolling Mean - {series_name}")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(data) + 1), rolling_variances, label="Rolling Variance", color="blue")
    plt.title(f"Rolling Variance - {series_name}")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

# Function to compute ACF manually
def compute_acf(data, lag):
    n = len(data)
    x_mean = np.mean(data)

    if lag >= 0:
        numerator = np.sum((data[:n - lag] - x_mean) * (data[lag:] - x_mean))
    else:
        numerator = np.sum((data[-lag:] - x_mean) * (data[:n + lag] - x_mean))

    denominator = np.sum((data - x_mean) ** 2)
    return numerator / denominator

def calculate_acf(data, max_lag):
    n = len(data)
    mean = np.mean(data)
    var = np.sum((data - mean) ** 2)
    acf_values = []
    for lag in range(0, max_lag+1):
        numerator = np.sum((data[:n - lag] - mean) * (data[lag:] - mean))
        acf_values.append(numerator / var)
    return acf_values


def manual_average_1step_forecast(train):
    """Computes 1-step ahead predictions for training dataset using Average Method."""
    forecasts = [np.nan]
    for t in range(1, len(train)):
        avg = np.mean(train[:t])
        forecasts.append(avg)
    return np.array(forecasts)

def manual_average_hstep_forecast(train, h):
    """Computes h-step ahead forecasts for testing dataset using Average Method."""
    forecast_value = np.mean(train)
    return np.full(h, forecast_value)

def manual_q_value(errors, lags=5):
    """Computes Q-value manually based on sum of squared autocorrelations up to a given lag using the ACF function from Toolbox."""
    acf_values = calculate_acf(errors, lags)
    print(acf_values)
    q_value = len(errors) * sum(np.array(acf_values) ** 2)
    return q_value

def manual_naive_hstep(train, h):
    """Computes Naïve Method forecasts: last observed value as future prediction."""
    forecast_value = train[-1]  # Last observed value as the forecast
    return np.full(h, forecast_value)

def manual_naive_1step(train_set):
    forecast = [np.nan]
    forecast.extend(train_set[:-1])
    return np.array(forecast)

def manual_drift_1step_forecast(train):
    """Computes 1-step ahead forecasts using Drift Method."""
    forecasts = [np.nan, np.nan]
    for t in range(2, len(train)):
        slope = (train[t - 1] - train[0]) / (t - 1) if t > 1 else 0
        forecasts.append(train[t - 1] + slope)
    return np.array(forecasts)

def manual_drift_hstep_forecast(train, h):
    """Computes h-step ahead forecasts using Drift Method."""
    y_T = train[-1]  # Last observed value in training set
    y_1 = train[0]   # First observed value in training set
    T = len(train)    # Number of observations in training set
    return np.array([y_T + (i + 1) * (y_T - y_1) / (T - 1) for i in range(h)])

def manual_ses_forecast(train, alpha, h):
    """Computes 1-step and h-step ahead forecasts using Simple Exponential Smoothing (SES)."""
    forecasts = [train[0]]
    for t in range(1, len(train)):
        ses_value = alpha * train[t - 1] + (1 - alpha) * forecasts[-1]
        forecasts.append(ses_value)
    h_step_forecast = np.full(h, forecasts[-1])  # Forecasting constant value for h-steps
    return np.array(forecasts), h_step_forecast


def plot_acf(residuals, method_name, lags):
    """Plots the ACF of residual errors using the manual ACF function."""
    acf_values = compute_acf(residuals, lags)
    lag_range = list(range(-lags, lags + 1))

    print(f"ACF Values of {method_name}:", acf_values)

    symmetric_average_acf_values = np.concatenate((acf_values[::-1][:-1], acf_values))

    plt.figure(figsize=(8, 5))
    plt.stem(range(-7, 8), symmetric_average_acf_values)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('Magnitude')
    plt.title(f'ACF of residuals errors for {method_name}')
    plt.show()


# Define Moving Average function
def moving_average(y, m):
    if m in [1, 2]:
        raise ValueError("m=1,2 will not be accepted")


    if m % 2 == 0:
        folding_m = int(input(f"Enter an even folding order for {m}-MA: "))
        if folding_m % 2 != 0:
            raise ValueError("Folding order must be even")

        ma_first = np.convolve(y, np.ones(m) / m, mode='same')
        ma_final = np.convolve(ma_first, np.ones(folding_m) / folding_m, mode='same')
        return ma_final
    else:
        return np.convolve(y, np.ones(m) / m, mode='same')

def rolling_mean(data, column_name):

    rolling_means = []

    print(f"Calculating rolling mean for {column_name}...")

    for i in range(1, len(data) + 1):
        current_data = data[column_name][:i]
        rolling_mean = current_data.mean()

        rolling_means.append(rolling_mean)


def custom_standardize(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def svd_condition_number(X):

    U, singular_values, VT = np.linalg.svd(X, full_matrices=False)
    condition_number = singular_values[0] / singular_values[-1]

    return singular_values, condition_number


def add_intercept(X):

    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def normal_equation(X, Y):

    XTX = X.T.dot(X)
    XTY = X.T.dot(Y)
    beta = np.linalg.inv(XTX).dot(XTY)
    return beta

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def simulate_ar2(n, a1, a2):
    np.random.seed(6313)
    y = np.zeros(n)
    e = np.random.normal(0, 1, size=n)

    y[0] = 0
    y[1] = 0

    for t in range(2, n):
        y[t] = a1 * y[t - 1] + a2 * y[t - 2] + e[t]

    return np.round(y, 2)


def estimate_ar2_lse(n):
    y = simulate_ar2(n, -0.5, -0.2)

    y_lag1 = y[1:-1]
    y_lag2 = y[:-2]
    target = y[2:]

    X = np.column_stack((y_lag1, y_lag2))
    X = sm.add_constant(X)

    model = sm.OLS(target, X).fit()

    print(f"\nSample Size: {n}")
    print(f"True coefficients: a1 = -0.5, a2 = -0.2")
    print(f"Estimated coefficients: a1 = {model.params[1]:.2f}, a2 = {model.params[2]:.2f}")


def simulate_ar_process(n, order, coeffs):
    if len(coeffs) != order:
        raise ValueError("Number of coefficients must match the order of the AR process.")

    np.random.seed(6313)
    y = np.zeros(n)
    e = np.random.normal(0, 1, size=n)

    for t in range(order, n):
        y[t] = sum(coeffs[i] * y[t - (i + 1)] for i in range(order)) + e[t]

    return y


def estimate_ar_params(y, order):
    y_lags = np.column_stack([y[(order - i - 1):- (i + 1)] for i in range(order)])
    target = y[order:]

    X = sm.add_constant(y_lags)
    model = sm.OLS(target, X).fit()

    return model


def simulate_ma2(n):
    np.random.seed(6313)
    y = np.zeros(n)
    e = np.random.normal(0, 1, n)

    y[0] = 0
    y[1] = 0

    for t in range(2, n):
        y[t] = e[t] + 0.5 * e[t - 1] + 0.2 * e[t - 2]

    return np.round(y, 2)


def simulate_arma(n):
    np.random.seed(6313)
    y = np.zeros(n)
    e = np.random.normal(0, 1, n)

    y[0] = 0
    y[1] = 0

    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] - 0.25 * y[t - 2] + e[t] - 0.3 * e[t - 1] - 0.6 * e[t - 2]

    return np.round(y, 2)


def estimate_ru_matrix(u, K=50):
    """
    Builds the ACF matrix R_u of shape K x K using unbiased sample autocorrelation of u(t)
    """
    u_centered = u - np.mean(u)

    N = len(u_centered)
    acf_vals = [np.sum(u_centered[:N - lag] * u_centered[lag:]) / N for lag in range(K)]

    Ru = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            Ru[i, j] = acf_vals[abs(i - j)]

    return Ru


def estimate_impulse_response_method_A(u, y, K=50):
    """
    Estimate impulse response g(k) using least-squares method:
    g = Ru_inv * Ruy
    """
    N = len(u)

    # Center the signals
    u_centered = u - np.mean(u)
    y_centered = y - np.mean(y)

    # Step 1: Compute Ru(τ)
    Ru = np.zeros((K, K))
    acf_u = [np.sum(u_centered[:N - tau] * u_centered[tau:]) / (N - tau) for tau in range(K)]

    for i in range(K):
        for j in range(K):
            Ru[i, j] = acf_u[abs(i - j)]

    # Step 2: Compute Ruy(τ)
    Ruy = np.zeros(K)
    for tau in range(K):
        Ruy[tau] = np.sum(u_centered[:N - tau] * y_centered[tau:]) / (N - tau)

    # Step 3: Compute g(k) = Ru^(-1) * Ruy
    g_est = inv(Ru) @ Ruy

    return g_est

def compute_theoretical_acf_from_g(g, max_lag=20):
    """
    Computes theoretical ACF from impulse response g(k) using convolution
    r_y(tau) = sum_k g(k) * g(k+tau)
    """
    g = np.array(g)
    r = []

    for tau in range(max_lag + 1):
        total = 0
        for k in range(len(g) - tau):
            total += g[k] * g[k + tau]
        r.append(total)
    return r

def compute_gpac(acf, max_j=7, max_k=7):
    gpac = np.full((max_j, max_k), np.nan)

    for j in range(max_j):
        for k in range(1, max_k + 1):
            try:
                D = np.zeros((k, k))
                for i in range(k):
                    for l in range(k):
                        lag = abs(j + i - l)
                        D[i, l] = acf[lag] if lag < len(acf) else 0

                N = D.copy()
                for i in range(k):
                    lag = j + i + 1
                    N[i, -1] = acf[lag] if lag < len(acf) else 0

                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)

                if np.abs(det_D) < 1e-8:
                    gpac[j, k - 1] = np.nan
                else:
                    gpac[j, k - 1] = det_N / det_D

            except Exception:
                gpac[j, k - 1] = np.nan

    df = pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)], columns=[f"k={k}" for k in range(1, max_k + 1)])
    df = df.round(2)
    df[df.abs() < 1e-3] = 0.0
    return df

def display_gpac_table(gpac_df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(gpac_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('GPAC Table')
    plt.xlabel('k (Columns)')
    plt.ylabel('j (Rows)')
    plt.show()

def simulate_H_response_only(n_samples=50000, std_e=1):
    """
    Simulates only H(q)*e(t), to isolate the noise filter
    """
    np.random.seed(6313)
    e = np.random.normal(0, std_e, n_samples)
    C = [1, 0.2]
    D = [1, -0.6]

    H_sys = dlti(C, D, dt=1)
    _, He_response = dlsim(H_sys, e)
    return e, He_response.flatten()

def ACF_PACF_Plot(y, lags=20, series_name="raw data"):
    acf_vals = sm.tsa.stattools.acf(y, nlags=lags)
    pacf_vals = sm.tsa.stattools.pacf(y, nlags=lags)

    symmetric_acf = np.concatenate((acf_vals[:0:-1], acf_vals))
    symmetric_pacf = np.concatenate((pacf_vals[:0:-1], pacf_vals))
    lag_range = np.arange(-lags, lags + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # ACF plot
    axes[0].stem(lag_range, symmetric_acf)
    axes[0].set_title(f'ACF Plot of {series_name}')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF Value')
    axes[0].axhspan(-1.96 / np.sqrt(len(y)), 1.96 / np.sqrt(len(y)), alpha=0.2, color='blue')

    # PACF plot
    axes[1].stem(lag_range, symmetric_pacf)
    axes[1].set_title(f'PACF Plot of {series_name}')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF Value')
    axes[1].axhspan(-1.96 / np.sqrt(len(y)), 1.96 / np.sqrt(len(y)), alpha=0.2, color='blue')

    plt.tight_layout(pad=3)
    plt.show()


def LM_algorithm(y, ar_order, ma_order, max_iter=100, epsilon=1e-3, delta=1e-6):
    N = len(y)
    na, nb = ar_order, ma_order
    n = na + nb
    theta = np.zeros(n)
    mu, mu_max = 0.01, 1e10
    SSE_history = []

    for iteration in range(max_iter):
        e = np.zeros(N)
        for t in range(max(na, nb), N):
            yt_pred = -sum([theta[i] * y[t - i - 1] for i in range(na)])
            yt_pred += sum([theta[na+j] * e[t - j - 1] for j in range(nb)])
            e[t] = y[t] - yt_pred
        SSE = np.dot(e, e)
        SSE_history.append(SSE)

        X = np.zeros((N, n))
        for i in range(n):
            theta_temp = theta.copy()
            theta_temp[i] += delta
            e_temp = np.zeros(N)
            for t in range(max(na, nb), N):
                yt_pred_temp = -sum([theta_temp[k] * y[t - k - 1] for k in range(na)])
                yt_pred_temp += sum([theta_temp[na+m] * e_temp[t - m - 1] for m in range(nb)])
                e_temp[t] = y[t] - yt_pred_temp
            X[:, i] = (e - e_temp) / delta

        A = X.T @ X
        g = X.T @ e

        while True:
            try:
                delta_theta = inv(A + mu * np.eye(n)) @ g
            except LinAlgError:
                mu *= 10
                if mu > mu_max:
                    print("Matrix singular or mu too large. Exiting.")
                    return theta, SSE_history
                continue
            theta_new = theta + delta_theta

            e_new = np.zeros(N)
            for t in range(max(na, nb), N):
                yt_pred_new = -sum([theta_new[i] * y[t - i - 1] for i in range(na)])
                yt_pred_new += sum([theta_new[na+j] * e_new[t - j - 1] for j in range(nb)])
                e_new[t] = y[t] - yt_pred_new
            SSE_new = np.dot(e_new, e_new)

            if SSE_new < SSE:
                if np.linalg.norm(delta_theta) < epsilon:
                    cov_theta = (SSE_new / (N - n)) * inv(A)
                    print("Converged successfully.")
                    return theta_new, SSE_history, cov_theta
                theta = theta_new
                mu /= 10
                break
            else:
                mu *= 10
                if mu > mu_max:
                    print("Mu exceeded maximum limit. Exiting.")
                    return theta, SSE_history
                continue
    print("Max iterations reached.")
    cov_theta = (SSE / (N - n)) * inv(A)
    return theta, SSE_history, cov_theta


def confidence_interval(theta, cov_matrix):
    conf_intervals = []
    for i in range(len(theta)):
        conf = 2 * np.sqrt(cov_matrix[i, i])
        conf_intervals.append((theta[i] - conf, theta[i] + conf))
    return conf_intervals

def backward_elimination(data, target, sl=0.05):
    features = data.columns.tolist()
    while True:
        X_be = sm.add_constant(data[features])
        model = sm.OLS(target, X_be).fit()
        p_values = model.pvalues.iloc[1:]  # exclude intercept
        max_p = p_values.max()
        if max_p > sl:
            to_drop = p_values.idxmax()
            print(f"Dropped '{to_drop}' (p={max_p:.4f})")
            features.remove(to_drop)
        else:
            break
    return model, features

def poles_zeros(ar_coeffs, ma_coeffs):
    poles = np.roots(np.r_[1, -np.array(ar_coeffs)]) if len(ar_coeffs) else []
    zeros = np.roots(np.r_[1, np.array(ma_coeffs)]) if len(ma_coeffs) else []
    return poles, zeros

def plot_sse(SSE_history):
    plt.plot(SSE_history, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Sum Squared Error')
    plt.title('Convergence of LM Algorithm')
    plt.grid(True)
    plt.show()


def ggpac_table(acf_values, max_k=7, max_j=7):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k + 1):
            try:
                D = np.zeros((k, k))
                for i in range(k):
                    for m in range(k):
                        lag = abs(j + i - m)
                        D[i, m] = acf_values[lag] if lag < len(acf_values) else 0

                N = D.copy()
                for i in range(k):
                    lag = j + i + 1
                    N[i, -1] = acf_values[lag] if lag < len(acf_values) else 0

                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)

                if np.isclose(det_D, 0):
                    gpac[j, k - 1] = np.nan if np.isclose(det_N, 0) else np.inf
                else:
                    gpac[j, k - 1] = det_N / det_D
            except Exception:
                gpac[j, k - 1] = np.nan

    df = pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)], columns=[f"k={k+1}" for k in range(max_k)])
    df = df.round(3)
    df = df.replace(-0.0, 0.0)
    df = df.applymap(lambda x: 0.0 if np.isclose(x, 0, atol=1e-3) else x)
    return df

def plot_ggpac_table(gpac_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title("G-GPAC Table [from Impulse Response]", fontsize=14)
    plt.xlabel("k (AR order)")
    plt.ylabel("j (MA order)")
    plt.tight_layout()
    plt.show()

def estimate_autocorrelations(u, y, K):
    N = len(u)
    Ru = np.zeros(K+1)
    Ruy = np.zeros(K+1)
    for tau in range(K+1):
        Ru[tau] = np.sum(u[:N - tau] * u[tau:]) / (N - tau)
        Ruy[tau] = np.sum(u[:N - tau] * y[tau:]) / (N - tau)
    return Ru, Ruy

def estimate_acf(x, K):
    n = len(x)
    acf = np.zeros(2 * K + 1)
    mean = np.mean(x)

    for tau in range(-K, K + 1):
        if tau < 0:
            acf[K + tau] = np.sum((x[:tau] - mean) * (x[-tau:] - mean)) / n
        else:
            acf[K + tau] = np.sum((x[tau:] - mean) * (x[:n - tau] - mean)) / n
    return acf

def hgpac_table(acf_vals, max_k=7, max_j=7):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k + 1):
            try:
                D = np.zeros((k, k))
                for i in range(k):
                    for m in range(k):
                        lag = abs(j + i - m)
                        D[i, m] = acf_vals[lag] if lag < len(acf_vals) else 0

                N = D.copy()
                for i in range(k):
                    lag = j + i + 1
                    N[i, -1] = acf_vals[lag] if lag < len(acf_vals) else 0

                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)

                if np.isclose(det_D, 0):
                    gpac[j, k - 1] = np.nan if np.isclose(det_N, 0) else np.inf
                else:
                    gpac[j, k - 1] = det_N / det_D
            except Exception:
                gpac[j, k - 1] = np.nan

    df = pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)], columns=[f"k={k+1}" for k in range(max_k)])
    df = df.round(3)
    df = df.replace(-0.0, 0.0)
    df = df.applymap(lambda x: 0.0 if np.isclose(x, 0, atol=1e-3) else x)
    return df

def plt_hgpac_table(gpac_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title("H-GPAC Table", fontsize=14)
    plt.xlabel("k (AR order)")
    plt.ylabel("j (MA order)")
    plt.tight_layout()
    plt.show()

def compute_error_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    y_g = np.zeros(N)
    e = np.zeros(N)

    b = np.r_[1, theta[:nb - 1]]
    f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
    c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
    d = np.r_[1, -np.array(theta[nb - 1 + nf + nc:])]

    max_lag = max(len(b), len(f))

    for t in range(max_lag, N):
        y_g[t] = sum(b[i] * u[t - i] for i in range(len(b))) - sum(f[j] * y_g[t - j] for j in range(1, len(f)))

    residual = y - y_g

    max_h_lag = max(len(c), len(d))
    for t in range(max_h_lag, N):
        num = sum(d[j] * residual[t - j] for j in range(len(d)))
        den = sum(c[i] * e[t - i] for i in range(1, len(c)))
        e[t] = num - den

    return e[max_h_lag:]

def compute_jacobian_bj(theta, y, u, nb, nf, nc, nd, delta=1e-7):
    base_error = compute_error_bj(theta, y, u, nb, nf, nc, nd)
    X = np.zeros((len(base_error), len(theta)))

    for i in range(len(theta)):
        perturbed = theta.copy()
        perturbed[i] += delta
        perturbed_error = compute_error_bj(perturbed, y, u, nb, nf, nc, nd)
        X[:, i] = (base_error - perturbed_error) / delta

    return X

def levenberg_marquardt_bj(y, u, theta_init, nb, nf, nc, nd, mu_init=0.01, max_iter=100, epsilon=1e-3):
    theta = theta_init.copy()
    mu = mu_init
    sse_track = []

    for it in range(max_iter):
        e = compute_error_bj(theta, y, u, nb, nf, nc, nd)
        SSE = e @ e
        sse_track.append(SSE)

        X = compute_jacobian_bj(theta, y, u, nb, nf, nc, nd)
        A = X.T @ X
        g = X.T @ e

        delta_theta = np.linalg.inv(A + mu * np.eye(len(theta))) @ g
        theta_new = theta + delta_theta
        e_new = compute_error_bj(theta_new, y, u, nb, nf, nc, nd)
        SSE_new = e_new @ e_new

        print(f"Iter {it} | SSE: {SSE:.4f} | mu: {mu:.2e} | Δθ norm: {np.linalg.norm(delta_theta):.2e}")

        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon:
                print("Converged.")
                break
            theta = theta_new
            mu /= 10
        else:
            mu *= 10

    return theta, sse_track

def confidence_intervals_bj(theta_est, y, u, nb, nf, nc, nd):
    N = len(y)
    p = len(theta_est)
    e = compute_error_bj(theta_est, y, u, nb, nf, nc, nd)
    sse = np.sum(e ** 2)
    sigma2 = sse / (N - p)

    X = compute_jacobian_bj(theta_est, y, u, nb, nf, nc, nd)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    stderr = np.sqrt(np.diag(cov))

    tval = t_dist.ppf(0.975, N - p)
    lower = theta_est - tval * stderr
    upper = theta_est + tval * stderr

    return list(zip(lower, theta_est, upper))

# Q-test

def q_test(residuals, lags=50, model_df=0, alpha=0.05):

    residuals = np.asarray(residuals, dtype=np.float64)
    N = len(residuals)
    residuals -= np.mean(residuals)
    var_e = np.var(residuals)

    Q = 0
    for tau in range(1, lags + 1):
        autocov = np.sum(residuals[tau:] * residuals[:-tau]) / (N - tau)
        r_tau = autocov / var_e
        Q += (r_tau ** 2) / (N - tau)

    Q_stat = N * (N + 2) * Q
    dof = lags - model_df
    Q_crit = chi2.ppf(1 - alpha, df=dof)

    print(f"\n--- Q-Test Summary ---")
    print(f"Q-statistic              : {Q_stat:.4f}")
    print(f"Chi-square Critical (α={alpha}, dof={dof}) : {Q_crit:.4f}")
    print("Result                   :",
          "Residuals are white (Q < Q*)" if Q_stat < Q_crit
          else "Residuals show autocorrelation (Q > Q*)")

    return Q_stat, Q_crit, dof

# s-test

def s_test(e, u, theta_est, nb, nf, K=20, significance=0.05):
    N = len(e)
    e = e - np.mean(e)

    f = np.r_[1, theta_est[nb - 1: nb - 1 + nf]]

    alpha_t = np.zeros_like(u)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j] * alpha_t[t - j] for j in range(1, len(f)))

    alpha_t = alpha_t - np.mean(alpha_t)

    sigma_e = float(np.std(e))
    sigma_a = float(np.std(alpha_t))

    S = 0.0
    r_vals = []
    for tau in range(K + 1):
        R_ae = np.sum(alpha_t[:N - tau] * e[tau:]) / (N - tau)
        r_ae = R_ae / (sigma_a * sigma_e)
        r_vals.append(r_ae)
        S += r_ae ** 2

    S_stat = float(N * S)
    dof = K - (nb - 1) - nf
    S_crit = float(chi2.ppf(1 - significance, df=dof))

    print(f"S-stat: {S_stat:.4f}")
    print(f"Chi-square S* (α={significance}, DOF={dof}): {S_crit:.4f}")
    if S_stat < S_crit:
        print("G(q) is accurate (S < S*)")
    else:
        print("G(q) may be misspecified (S > S*)")

    return S_stat, S_crit, dof, r_vals


# ARMA FORECASTING

def forecast_arma(y, phi, theta, residuals, steps=1):
    p = len(phi)
    q = len(theta)
    y_hat = []
    y_hist = list(y)
    e_hist = list(residuals)

    for h in range(steps):
        # AR part: use last p y values
        ar_part = sum(phi[i] * y_hist[-i - 1] for i in range(p))

        # MA part: use last q residuals (real ones if available)
        ma_part = sum(theta[j] * e_hist[-j - 1] if len(e_hist) > j else 0 for j in range(q))

        y_next = ar_part + ma_part
        y_hat.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)  # assume zero residuals for future steps

    return np.array(y_hat)


# ARMA

def one_step_forecast_arma_plot(y, e, phi, theta, n_plot=20):

    y = np.asarray(y)
    e = np.asarray(e)
    p, q = len(phi), len(theta)
    N = len(y)
    lag = max(p, q)

    # compute one-step forecasts
    y_hat = np.zeros(N)
    for t in range(lag, N):
        ar = sum(phi[i] * y[t-i-1]   for i in range(p))
        ma = sum(theta[j] * e[t-j-1] for j in range(q))
        y_hat[t] = ar + ma

    actual   = y[lag:]
    forecast = y_hat[lag:]
    m = min(n_plot, len(actual))

    # plot
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(m), actual[:m],   'o-', label='Actual')
    plt.plot(np.arange(m), forecast[:m], 'x--', label='1-step Forecast')
    plt.title(f'One-Step Forecast vs Actual (first {m} points)')
    plt.xlabel('t')
    plt.legend()
    plt.grid(True)
    plt.show()

    res_var = np.var(actual[:m] - forecast[:m])
    print(f"1-step residual var       = {res_var:.4f}")

    return forecast

def h_step_forecast_arma_plot(y, e, phi, theta, h, y_actual=None):

    y = np.asarray(y)
    e = np.asarray(e)
    p, q = len(phi), len(theta)

    # prepare history
    y_hist = list(y)
    e_hist = list(e)
    forecasts = []

    for step in range(h):
        ar = sum(phi[i] * y_hist[-i-1] for i in range(p))
        ma = sum(theta[j] * e[-j-1] if step == 0 else 0 for j in range(q))
        y_next = ar + ma
        forecasts.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)

    # plot if true future provided
    if y_actual is not None:
        actual = np.asarray(y_actual)
        m = min(h, len(actual))
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(m), actual[:m],   'o-', label='Actual')
        plt.plot(np.arange(m), forecasts[:m], 'x--', label=f'{h}-step Forecast')
        plt.title(f'{h}-Step Forecast vs Actual')
        plt.xlabel('h')
        plt.legend()
        plt.grid(True)
        plt.show()

        fc_var = np.var(actual[:m] - forecasts[:m])
        print(f"{h}-step forecast var     = {fc_var:.4f}")

    return forecasts


def compute_residuals_from_lm(y, phi, theta):
    y = np.asarray(y)
    N = len(y)
    p = len(phi)
    q = len(theta)

    y_hat = np.zeros(N)
    residuals = np.zeros(N)

    # Start from max(p, q) to ensure lag availability
    for t in range(max(p, q), N):
        # AR component: use only past y values
        ar_part = sum(phi[i] * y[t - i - 1] for i in range(p))

        # MA component: use only past residuals
        ma_part = sum(theta[j] * residuals[t - j - 1] for j in range(q))

        y_hat[t] = ar_part + ma_part
        residuals[t] = y[t] - y_hat[t]

    return residuals

def compute_residuals_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    y_g = np.zeros(N)
    e = np.zeros(N)

    b = np.r_[1, theta[:nb - 1]]
    f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
    c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
    d = np.r_[1, -np.array(theta[nb - 1 + nf + nc:])]

    for t in range(max(len(b), len(f)), N):
        y_g[t] = sum(b[i] * u[t - i] for i in range(len(b))) - sum(f[j] * y_g[t - j] for j in range(1, len(f)))

    residual = y - y_g

    for t in range(max(len(c), len(d)), N):
        num = sum(d[j] * residual[t - j] for j in range(len(d)))
        den = sum(c[i] * e[t - i] for i in range(1, len(c)))
        e[t] = num - den

    return e

def compute_covariance_bj(theta_est, y, u, nb, nf, nc, nd):

    n = len(y)
    p = nb + nf + nc + nd
    eps = 1e-5

    J = np.zeros((n, p))
    f0 = compute_error_bj(theta_est, y, u, nb, nf, nc, nd)

    for i in range(p):
        theta_perturbed = theta_est.copy()
        theta_perturbed[i] += eps
        f1 = compute_error_bj(theta_perturbed, y, u, nb, nf, nc, nd)
        J[:, i] = (f1 - f0) / eps

    sigma2 = np.var(f0)

    cov_matrix = sigma2 * np.linalg.inv(J.T @ J)

    return sigma2, cov_matrix

def forecast_bj_1step(y, u, e, theta, nb, nf, nc, nd, steps=20, start=0):
    b1, f1, c1, d1 = theta
    N = len(y)
    yhat = []
    idx = []
    for t in range(start, start + steps):
        if t < 1 or t >= N - 1:
            continue
        val = b1 * u[t] - f1 * y[t] + c1 * e[t] - d1 * e[t - 1]
        yhat.append(val)
        idx.append(t)
    return np.array(yhat), idx


def forecast_bj_hstep(y, u, e, theta, nb, nf, nc, nd, steps=20, start=0):
    """
    General h-step forecast function for BJ models with arbitrary nb, nf, nc, nd.
    """
    b = theta[:nb]
    f = theta[nb:nb+nf]
    c = theta[nb+nf:nb+nf+nc]
    d = theta[nb+nf+nc:]

    y_hist = list(y[:start + nf])
    u_hist = list(u[:start + nb])
    e_hist = list(e[:start + max(nc, nd)])

    yhat = []

    for h in range(steps):
        t = start + h

        bu = sum(b[i] * (u_hist[t - i] if t - i >= 0 else 0) for i in range(nb))
        fy = sum(f[j] * y_hist[-j - 1] for j in range(nf))
        ce = sum(c[k] * e_hist[-k - 1] for k in range(nc))
        de = sum(d[l] * e_hist[-l - 1] for l in range(nd))

        y_next = bu - fy + ce - de

        yhat.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)
        u_hist.append(u[t] if t < len(u) else u[-1])

    return np.array(yhat)


def evaluate_forecast(true, pred, num_params):
    true = np.array(true)
    pred = np.array(pred)

    residuals = true - pred
    n = len(true)

    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)

    # R² and Adjusted R²
    sse = np.sum(residuals ** 2)
    sst = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - sse / sst
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - num_params - 1)

    # AIC and BIC
    aic = n * np.log(mse) + 2 * num_params
    bic = n * np.log(mse) + num_params * np.log(n)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Adj R²": r2_adj,
        "AIC": aic,
        "BIC": bic
    }


def filter_significant_params(theta_est, conf_intervals, ar_order):
    """
    Remove parameters whose confidence intervals include 0.
    Returns reduced parameter list and indices of kept parameters.
    """
    keep_indices = []
    for i, (lower, upper) in enumerate(conf_intervals):
        if lower > 0 or upper < 0:  # CI does NOT include 0
            keep_indices.append(i)

    # Filter theta
    theta_reduced = theta_est[keep_indices]

    # Optional: label retained parameters
    labels = []
    for i in keep_indices:
        label = f"AR[{i+1}]" if i < ar_order else f"MA[{i+1-ar_order}]"
        labels.append(label)

    return theta_reduced, keep_indices, labels






# END #