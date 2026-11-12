
# Bitcoin Price Forecasting with ARIMA

## One-Day-Ahead Financial Asset Price Prediction Using Autoregressive Time Series Modeling

A quantitative financial forecasting project applying an Autoregressive Integrated Moving Average (ARIMA) model to historical Bitcoin prices, with the objective of predicting the asset's next-day closing price.

The project develops and evaluates a reproducible time series forecasting methodology based exclusively on historical market prices, combining statistical stationarity testing, autoregressive modeling, residual diagnostics and sequential out-of-sample validation.

**Key result: 2.86% mean absolute percentage forecasting error across 490 consecutive one-day-ahead predictions.**

## Project Overview

### Objective: Predicting the Next-Day Price of a Financial Asset

Financial asset prices evolve through complex market dynamics involving historical trends, volatility, liquidity, investor expectations and external economic factors.

Forecasting these prices is particularly challenging in cryptocurrency markets, where trading occurs continuously and substantial price fluctuations can develop over relatively short periods.

The objective of this project is to investigate whether **historical Bitcoin prices contain sufficient temporal information to support short-term price forecasting**, without incorporating additional explanatory variables.

The analytical framework addresses four objectives:

- **Time series analysis:** investigate the statistical properties of historical Bitcoin prices, including trends, non-stationarity and temporal dependence.
- **Autoregressive forecasting:** develop an ARIMA model capable of predicting the following day's closing price from previously observed prices.
- **Sequential validation:** evaluate the model through repeated one-day-ahead forecasts, incorporating each newly observed price into the historical information available for the next prediction.
- **Methodological transferability:** establish a reproducible forecasting framework that can be adapted and tested on other financial assets with sufficiently structured historical price series.

The project emphasizes the practical application of statistical time series forecasting to financial markets, moving beyond historical price description toward **systematic, quantitatively evaluated next-day prediction**.

## Why Bitcoin? A Financial Asset Forecasting Case Study

Bitcoin provides a relevant environment for investigating short-horizon financial forecasting.

Unlike traditional equities, whose valuation is directly connected to corporate earnings, dividends, financial statements and company-specific fundamentals, Bitcoin does not represent ownership of an underlying business.

Its market price reflects supply and demand, liquidity conditions, investor expectations, macroeconomic developments, market sentiment and other cryptocurrency-specific dynamics.

Bitcoin is therefore not independent of external market factors. However, its price series can be analyzed through a **univariate forecasting approach**, allowing the predictive contribution of historical price dynamics to be investigated separately from external explanatory variables.

This distinction is central to the project.

Rather than attempting to model every possible determinant of Bitcoin's market value, the analysis asks a more focused quantitative question:

**Can a statistical model based exclusively on previously observed prices generate sufficiently accurate predictions of Bitcoin's next-day closing price?**

ARIMA provides a structured methodology for examining this question.

Its applicability is not limited to cryptocurrencies. The same modeling and validation framework can be adapted to other financial instruments, although their statistical properties, market conditions and forecasting performance must be evaluated independently.

## The ARIMA Approach: Forecasting from Historical Price Dynamics

ARIMA is a statistical forecasting methodology designed to model temporal dependence in time series.

It combines three components:

**Autoregressive (AR)**

Uses previous observations to explain the current value of a time series.

**Integrated (I)**

Applies differencing to address non-stationarity and make the series more suitable for autoregressive modeling.

**Moving Average (MA)**

Models dependence on previous forecast errors when an additional moving-average component is required.

An ARIMA model is represented by three parameters:

**ARIMA(p, d, q)**

Where:

- **p** is the number of autoregressive lags.
- **d** is the order of differencing.
- **q** is the number of moving-average error terms.

In this project, the selected configuration is:

### ARIMA(1, 1, 0)

The model incorporates one autoregressive lag, first-order differencing and no moving-average component.

This specification provides a relatively parsimonious approach to modeling Bitcoin price dynamics.

The objective is not to reproduce every market movement, but to determine whether a simple autoregressive structure can produce useful short-term forecasts when evaluated against previously unseen observations.

## Dataset and Historical Market Coverage

The project uses historical daily Bitcoin market data from the `coin_Bitcoin.csv` dataset.

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Financial asset | Bitcoin |
| Symbol | BTC |
| Original observations | 2,991 |
| Historical period | April 29, 2013 – July 6, 2021 |
| Frequency | Daily |
| Market variables | Open, High, Low, Close |
| Forecast target | Daily closing price |
| Forecast horizon | One day ahead |

The original dataset also includes additional market information, but the forecasting model focuses on the historical closing-price series.

The date is converted into a datetime index, preserving the chronological structure required for time series analysis.

The modeling process is based on **historical prices rather than external macroeconomic, corporate or sentiment indicators**.

This isolates the univariate forecasting methodology and allows its performance to be evaluated independently of additional explanatory datasets.

## Methodology

### 1. Historical Bitcoin Price Dynamics

The first analytical stage examines the evolution of Bitcoin's daily closing price.

The historical series exhibits substantial long-term price appreciation alongside periods of pronounced volatility and rapid market movements.

These characteristics are particularly relevant for financial forecasting because many classical time series models require assumptions about the statistical behavior of the underlying data.

A central requirement is to determine whether the series is stationary or whether transformations are needed before constructing the forecasting model.

### 2. Stationarity Assessment: Augmented Dickey-Fuller Test

The project applies the **Augmented Dickey-Fuller (ADF) test** to investigate whether the Bitcoin closing-price series contains a unit root.

The null hypothesis is that the series contains a unit root, consistent with non-stationarity.

The initial test produces:

| Statistic | Result |
|---|---:|
| ADF statistic | -0.7973 |
| p-value | 0.8199 |
| Significance threshold | 0.05 |

The p-value is substantially higher than 0.05.

Consequently, the null hypothesis cannot be rejected.

The original Bitcoin closing-price series does not provide sufficient evidence of stationarity.

This result supports the need for a transformation before autoregressive modeling.

### 3. Logarithmic Transformation and First-Order Differencing

The project investigates two transformations of the original price series.

**Logarithmic Transformation**

The natural logarithm of the closing price is calculated to examine the behavior of the series on a transformed scale.

However, the ADF test still indicates non-stationarity.

**First-Order Differencing**

The project subsequently calculates the difference between consecutive logarithmic closing prices.

The transformed series is defined as:

`Close_log_diff(t) = log(Close(t)) - log(Close(t-1))`

The ADF test on the differenced logarithmic series indicates stationarity at the selected significance threshold.

This supports the use of one differencing operation.

**Selected integration order: d = 1.**

An important methodological distinction is that the stationarity analysis is performed on logarithmic differences, while the final ARIMA model is fitted directly to the closing-price series with first-order differencing.

The two transformations are related but not identical, and the notebook does not separately report an ADF test on the first differences of untransformed closing prices.

### 4. Autocorrelation Analysis and Model Specification

The project examines temporal dependence through the Autocorrelation Function (ACF).

The analysis evaluates the autocorrelation structure of the differenced logarithmic closing-price series across 40 lags.

The first lag is selected as the reference for the autoregressive specification.

The resulting model is:

**ARIMA(1, 1, 0)**

| Parameter | Value | Interpretation |
|---|---:|---|
| p | 1 | One autoregressive lag |
| d | 1 | First-order differencing |
| q | 0 | No moving-average error component |

The model is implemented using the `statsmodels` ARIMA framework.

This specification emphasizes simplicity and interpretability, providing a focused test of whether recent price information can support next-day forecasts.

The notebook selects the model based on stationarity and autocorrelation analysis rather than a systematic comparison of multiple ARIMA configurations using information criteria.

### 5. Model Estimation and Residual Diagnostics

The ARIMA model is fitted to the historical Bitcoin closing-price series.

The project uses the diagnostic plotting functions provided by `statsmodels` to examine the model's residual behavior.

The diagnostics include:

- Standardized residuals over time.
- Distribution of residuals.
- Residual quantile comparison.
- Residual autocorrelation.

Residual analysis is important because a forecasting model should not leave substantial systematic temporal patterns unexplained.

The diagnostic plots reveal periods of pronounced residual deviations, particularly during substantial Bitcoin market movements.

These observations highlight a limitation of a relatively simple autoregressive model: extreme market changes and abrupt shifts in price dynamics may not be fully captured by historical price dependence alone.

The diagnostics therefore complement the subsequent forecasting evaluation by identifying conditions under which the model may be less reliable.

## 6. One-Day-Ahead Forecasting with Walk-Forward Validation

### The Central Predictive Experiment

The most important component of the project is the **sequential forecasting procedure**, designed to simulate the progressive availability of new financial market observations.

Instead of fitting the model once and evaluating a single prediction, the methodology generates repeated one-day-ahead forecasts across a chronological test period.

This approach is commonly referred to as **walk-forward validation** or **rolling-origin forecasting with an expanding training window**.

It is particularly relevant to financial markets because new prices become available continuously, and a forecasting system must incorporate the latest observed information before generating subsequent predictions.

### Chronological Train-Test Split

The historical series is divided into two sequential subsets.

| Dataset | Observations |
|---|---:|
| Training history | 2,500 |
| Out-of-sample test period | 490 |
| **Total modeled observations** | **2,990** |

The original dataset contains 2,991 observations; one observation is removed during the construction of the differenced series.

The split preserves the chronological order of the observations.

This is essential for financial forecasting, where randomly mixing historical and future observations could compromise the validity of the predictive evaluation.

### Expanding-Window Forecasting Procedure

For each day in the test period, the following procedure is executed:

1. Fit the ARIMA(1, 1, 0) model using all historical closing prices available up to that point.
2. Generate a forecast for the following day's closing price.
3. Compare the forecast with the actual observed closing price.
4. Calculate the absolute percentage forecasting error.
5. Add the actual observed price to the historical dataset.
6. Refit the ARIMA model using the expanded history.
7. Repeat the process for the next trading observation.

Because Bitcoin trades continuously, the dataset provides consecutive daily observations rather than an equity-market trading-day calendar.

The model is re-estimated at every iteration.

This means that each new forecast incorporates the most recently observed market price without using future observations that were unavailable at the time of prediction.

### Why This Validation Approach Matters

The methodology evaluates the forecasting model under a sequence of progressively updated information sets.

This is more informative than assessing only how closely a fitted model reproduces its original training data.

The resulting 490 predictions provide an empirical basis for examining the model's short-term forecasting accuracy over an extended historical period.

The experiment demonstrates a **reproducible framework for one-step-ahead financial time series forecasting**.

## 7. Forecasting Performance and Results

### Mean Absolute Percentage Forecasting Error

For each forecast, the notebook calculates the absolute percentage difference between the predicted and observed closing price:

`Absolute Percentage Error = |Predicted Price - Actual Price| / Actual Price × 100`

The errors are averaged across the test observations.

### Main Quantitative Result

| Evaluation Metric | Result |
|---|---:|
| Forecasting model | ARIMA(1, 1, 0) |
| Forecast horizon | 1 day |
| Out-of-sample predictions | 490 |
| Validation methodology | Walk-forward, expanding window |
| Model re-estimation | After each new observation |
| **Mean Absolute Percentage Error (MAPE)** | **2.858838%** |

### Interpretation

The model achieves a **mean absolute percentage error of approximately 2.86%** across 490 consecutive one-day-ahead forecasts.

This indicates that the predicted closing prices differ from the actual observed prices by approximately 2.86% on average, expressed relative to the actual price.

The result is notable because the model relies exclusively on historical Bitcoin prices.

No external economic indicators, sentiment measures or additional financial-market variables are included.

The notebook also compares predicted and observed prices graphically, providing a visual assessment of forecasting behavior across the test period.

### What the Result Demonstrates

The experiment provides evidence that a parsimonious autoregressive model can generate relatively close next-day price estimates across the evaluated historical period.

However, the result should be interpreted within its specific scope.

A low percentage forecasting error does not automatically demonstrate profitable trading performance, superior directional prediction or forecasting skill relative to a naive benchmark.

In particular, financial asset prices often exhibit substantial persistence between consecutive observations.

A complete comparative evaluation would therefore also measure the model against a simple persistence forecast, such as predicting that tomorrow's closing price will equal today's closing price.

The notebook establishes the ARIMA model's observed forecasting error, but does not report this additional benchmark comparison.

## 8. Reproducibility and Transferability to Other Financial Assets

### A Generalizable Financial Forecasting Methodology

An important contribution of the project is the development of a forecasting workflow that is not intrinsically dependent on Bitcoin-specific explanatory variables.

The methodology is based on the statistical properties of a financial asset's historical price series.

Its principal components can therefore be reproduced on other assets:

**Historical Prices → Stationarity Assessment → Differencing → Autoregressive Model Specification → Residual Diagnostics → Chronological Validation → Sequential One-Day-Ahead Forecasting.**

Potential applications include:

- Other cryptocurrencies with continuous daily trading.
- Equity prices and exchange-traded funds.
- Foreign exchange rates.
- Commodity prices.
- Financial market indices.

The methodology can be transferred, but the selected ARIMA parameters and the reported accuracy cannot be assumed to remain valid for other assets.

Each new time series requires independent statistical assessment, model calibration and out-of-sample validation.

### Univariate Modeling Versus Multivariable Financial Forecasting

The project deliberately adopts a univariate approach.

This means that the forecast is derived from the asset's own historical closing prices rather than from additional explanatory variables.

For equities, forecasting may also involve corporate fundamentals, earnings expectations, industry conditions and macroeconomic indicators.

For cryptocurrencies, relevant external drivers may include market liquidity, monetary policy, investor sentiment, regulatory developments and broader financial-market conditions.

ARIMA does not assume that these factors are irrelevant or that Bitcoin is statistically independent of other assets.

Instead, the project tests whether useful short-horizon predictions can be obtained **without explicitly incorporating those factors**.

This provides a baseline for more complex forecasting approaches that may subsequently integrate additional variables through multivariate time series models or other Machine Learning methodologies.

### Why the Method Is Reproducible

The project uses:

- A defined historical dataset.
- Explicit statistical tests.
- A documented ARIMA specification.
- A chronological train-test split.
- A repeatable expanding-window evaluation.
- A transparent forecasting error calculation.

These elements allow the methodology to be replicated and adapted to other financial price series.

The main transferable result is the **analytical and validation framework**, rather than the specific forecasting error achieved on Bitcoin.

## Financial and Investment Applications

### Short-Term Asset Monitoring

One-day-ahead price forecasts can provide an additional quantitative reference for monitoring short-term market developments.

### Quantitative Investment Research

The methodology supports the systematic evaluation of temporal price dependence and the comparison of statistical forecasting approaches.

### Forecasting Model Benchmarking

ARIMA can serve as a statistical reference model when evaluating more complex forecasting techniques, including multivariate models and Machine Learning architectures.

### Market Risk Analysis

Forecast deviations and residual behavior can help identify periods in which observed price movements are not adequately explained by the selected historical model.

However, the project does not estimate volatility forecasts, Value at Risk or formal prediction intervals.

### Multi-Asset Research Frameworks

The reproducible methodology provides a foundation for evaluating short-horizon forecasting performance across multiple asset classes using consistent statistical and validation procedures.

## Key Findings

**1. Bitcoin's Historical Closing-Price Series Is Non-Stationary**

The Augmented Dickey-Fuller test on the original series produces a p-value of approximately 0.82, supporting the need for transformation before autoregressive modeling.

**2. First-Order Differencing Supports a Stationary Transformed Series**

The logarithmic first-difference series passes the notebook's stationarity test, motivating a first-order integration specification.

**3. A Parsimonious ARIMA Model Is Applied**

The project implements ARIMA(1, 1, 0), using one autoregressive lag and first-order differencing without a moving-average component.

**4. The Model Is Evaluated Through Sequential One-Day-Ahead Forecasts**

A walk-forward procedure repeatedly refits the model and incorporates each newly observed price into the expanding historical dataset.

**5. The Forecasting Experiment Covers 490 Out-of-Sample Observations**

The chronological test provides a substantial sequence of daily predictions rather than relying on a single forecast.

**6. The Model Achieves a Mean Absolute Percentage Error of 2.86%**

The result demonstrates relatively close next-day price estimates over the historical evaluation period, while remaining subject to market volatility and the limitations of a univariate model.

**7. The Methodology Can Be Adapted to Other Financial Assets**

The statistical workflow is reproducible and can be applied to other historical price series, provided that each asset undergoes independent model selection and validation.

## Technologies and Methodologies

**Programming and Data Analysis**

Python · Pandas · NumPy

**Time Series Modeling**

Statsmodels · ARIMA · Autoregressive Modeling · First-Order Differencing

**Statistical Testing**

Augmented Dickey-Fuller Test · Stationarity Analysis · Autocorrelation Function (ACF)

**Model Diagnostics**

Residual Analysis · Standardized Residuals · Residual Distribution · Residual Autocorrelation

**Forecasting Evaluation**

Chronological Train-Test Split · Walk-Forward Validation · Expanding Training Window · One-Step-Ahead Forecasting · Mean Absolute Percentage Error

**Visualization**

Matplotlib · Historical Price Charts · ACF Plots · ARIMA Diagnostic Plots · Predicted vs. Actual Price Comparison

**Development Environment**

Google Colab · Jupyter Notebook

## Project Files

### Notebook

[Forecasting_model_of_Bitcoin_price_prediction_based_on_time_series.ipynb](Forecasting_model_of_Bitcoin_price_prediction_based_on_time_series.ipynb)

The notebook contains the complete analytical workflow, including historical price analysis, stationarity testing, logarithmic transformation, differencing, autocorrelation analysis, ARIMA model estimation, residual diagnostics and sequential one-day-ahead forecasting.

### Dataset

The project uses the historical Bitcoin price dataset stored in this repository:

[datasets/coin_Bitcoin.csv](datasets/coin_Bitcoin.csv)

The notebook retrieves the dataset directly from GitHub:

https://raw.githubusercontent.com/Valeria-Raele/bitcoin-price-forecasting-arima/refs/heads/main/datasets/coin_Bitcoin.csv

The dataset is loaded programmatically without requiring manual file uploads during notebook execution.

## Analytical Workflow

**Historical Bitcoin Prices → Closing-Price Time Series → Augmented Dickey-Fuller Stationarity Test → Logarithmic Transformation → First-Order Differencing → Autocorrelation Analysis → ARIMA(1, 1, 0) Estimation → Residual Diagnostics → Chronological Train-Test Split → Walk-Forward One-Day-Ahead Forecasting → Percentage Error Evaluation → Financial Forecasting Interpretation.**

The distinctive contribution of the project is the integration of statistical time series analysis with a sequential forecasting procedure that continuously incorporates new market observations.

The methodology demonstrates how historical asset prices can be transformed into a reproducible, quantitatively evaluated framework for short-horizon financial forecasting.

## Project Scope and Limitations

This is an independent quantitative financial forecasting project focused on one-day-ahead Bitcoin closing-price prediction.

The evaluation covers historical daily observations from 2013 to 2021 and reports forecasting performance over a 490-observation chronological test period.

The model uses only historical closing prices and does not incorporate macroeconomic variables, market sentiment, trading volume or cross-asset relationships.

The reported mean absolute percentage error describes average relative price forecast deviations. It does not measure trading profitability, directional accuracy or performance relative to a naive persistence benchmark.

The stationarity assessment is performed on logarithmic differences, while the final ARIMA model operates on the original closing-price series with first-order differencing.

The notebook does not perform a systematic ARIMA hyperparameter search, provide calibrated forecast intervals or test the model on more recent market regimes.

The model's performance should therefore not be assumed to generalize unchanged to future Bitcoin prices or other financial assets.

Further research could incorporate naive forecasting benchmarks, alternative ARIMA specifications, rolling-window performance comparisons, volatility-aware models and additional explanatory variables.

The project's principal contribution is a **transparent, reproducible and transferable methodology for sequential financial asset price forecasting**, supported by an empirical evaluation of 490 one-day-ahead Bitcoin price predictions.
