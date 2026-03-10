# Time-Series-and-Financial-Analysis
"Project Title: Time Series Analysis of Gold ETF Prices
Course: DATS 6313 - Time Series Analysis & Modeling
Author: Mohammed Hussain Nathani
GWID: G-36308827
Date: 05/04/2025"


Files Required:

Project_Final_Analysis.py — Main script containing all analysis and models.
ToolBox.py — Contains all custom functions used in the project.
cleaned_financial_dataset.csv — The input dataset used for modeling.


Requirements:

Make sure the following Python libraries are installed:
`pip install pandas numpy matplotlib seaborn statsmodels scikit-learn`


How to Run:

The main Python script Project_Final_Analysis.py is organized into sequential chunks, corresponding to each section in the project report (e.g., EDA, stationarity, decomposition, model fitting, evaluation).

You can run it using any of the following methods:
Option 1: In an IDE (Recommended)
Open Project_Final_Analysis.py in VS Code, PyCharm, or Jupyter Notebook.
Run the script chunk by chunk using #%% cell markers.
Make sure the dataset path in line 34 is correctly set:
`df = pd.read_csv("D:\\GWU\\SEM 2\\DATS 6313 (TS)\\Project\\cleaned_financial_dataset.csv", parse_dates=["date"], index_col="date")`
Update the path to match your local machine if needed.

Option 2: In Terminal
If you want to run the whole script at once:
Project_Final_Analysis.py
This may take time and produce a lot of output. It is recommended to run section by section.


Output:
The script will generate:
Plots for EDA, ACF/PACF, decomposition
Model summary outputs and metrics
Residual diagnostics
Final forecast comparison (ARMA, ARIMA, Box-Jenkins)
These outputs reproduce all results shown in the submitted report.


Notes:
All models (Base, ARMA, ARIMA, BJ) are manually implemented and evaluated.
Forecast accuracy is printed and plotted in the final sections.
Custom functions are defined in ToolBox.py and imported into the main script.
