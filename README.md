# AI-Based Short-Term Electricity Load Forecasting

This portfolio project builds an AI-based forecasting pipeline for short-term electricity load prediction using the UCI ElectricityLoadDiagrams20112014 dataset.

The project is designed for smart energy, green technology, and AI-driven energy management applications, especially for roles related to smart grid operation, building energy management, renewable energy operation, storage scheduling, and electricity price optimization.

## Project Goals

- Forecast future electricity load for a selected client or meter.
- Support two forecasting horizons:
  - 1-hour ahead forecasting: 4 future 15-minute steps.
  - 24-hour ahead forecasting: 96 future 15-minute steps.
- Use historical load, calendar features, weekday/weekend indicators, holiday flags, lag features, and rolling statistics.
- Compare simple baselines, machine learning models, and later deep learning models.
- Add robustness analysis under sensor noise, missing data, abnormal peaks, and distribution shift.

## Dataset

Dataset: UCI ElectricityLoadDiagrams20112014

Original source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

Citation: Trindade, A. (2015). ElectricityLoadDiagrams20112014 [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C58C86. License: CC BY 4.0.

The dataset contains electricity consumption time series for multiple clients. The original data is recorded at 15-minute intervals, which makes it suitable for short-term load forecasting.

Place the raw dataset file under:

```text
data/raw/
```

See `data/README.md` for download and placement instructions.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace notebooks\01_eda.ipynb
```

In VS Code, select `.venv\Scripts\python.exe` as the notebook kernel and open
`notebooks/01_eda.ipynb`.

## Forecasting Definition

### Prediction Task

Predict the future electricity load of a selected client or meter.

### Input

- Historical electricity load
- Time features:
  - hour
  - day of week
  - month
  - weekend indicator
- Holiday indicator
- Lag features
- Rolling mean and rolling standard deviation features

### Output

- Future electricity load value
- Main horizon: 1 hour ahead
- Extended horizon: 24 hours ahead

### Application Scenarios

- Smart grid dispatching
- Building energy management
- Renewable energy operation planning
- Energy storage scheduling
- Electricity price optimization

## Initial EDA Results

The first analysis selects `MT_252` as a representative client: among meters
with non-zero demand in at least 99% of intervals, its mean demand is closest
to the group median. This avoids both late-starting meters and unusually large
industrial loads.

Key findings:

- Data shape: 140,256 timestamps x 370 clients, from 2011-01-01 to 2015-01-01.
- Selected meter mean and standard deviation: 232.63 kW and 82.16 kW.
- Average off-peak hour: 03:00; average peak hour: 18:00.
- Weekend mean demand is only 1.32% above weekday mean demand for this meter.
- 1-hour lag autocorrelation: 0.896; 24-hour lag autocorrelation: 0.954.
- Brief near-zero drops in the example week are candidates for later anomaly
  and robustness analysis; they should not be labeled as faults without more
  operational context.

The strong 24-hour autocorrelation supports a seasonal-naive baseline and
`lag_96`. The high 1-hour autocorrelation supports `lag_4`. The small weekend
gap suggests that time-of-day and lag features may be more informative than a
weekend flag for this particular client.

![One-week load pattern](reports/figures/week_load_pattern.png)

![Average daily load pattern](reports/figures/average_daily_pattern.png)

![Weekday versus weekend load pattern](reports/figures/weekday_vs_weekend_pattern.png)

The machine-readable summary is saved at `reports/tables/eda_summary.csv`.

## Repository Structure

```text
energy-load-forecasting/
  README.md
  requirements.txt
  .gitignore
  data/
    README.md
    raw/
    processed/
  notebooks/
    01_eda.ipynb
  src/
    __init__.py
    config.py
    data_loader.py
    features.py
    evaluate.py
    forecasting.py
  reports/
    figures/
    tables/
  GITHUB_GUIDE.md
```

## Planned Roadmap

### Week 1: Data Understanding and EDA (Completed)

- Load the UCI electricity load dataset.
- Parse timestamps.
- Select one or several representative clients.
- Visualize daily and weekly load patterns.
- Identify peak and off-peak periods.

### Week 2: Baseline Models

- Naive baseline.
- Seasonal naive baseline.
- Ridge Regression.
- Time-series train/validation/test split.

### Week 3: Machine Learning Models

- Random Forest.
- XGBoost or LightGBM.
- Feature importance analysis.

### Week 4-5: Deep Learning Models

- LSTM.
- GRU.
- Optional Transformer model.

### Week 6: Robustness Analysis

- Noise disturbance.
- Missing data.
- Abnormal peaks.
- Distribution shift.

### Week 7: Dashboard

- Streamlit demo for visualizing forecasts, errors, and energy insights.

### Week 8: Portfolio Packaging

- Polish README.
- Add project summary.
- Prepare CV bullet points and interview pitch.

## How To Define Similar Projects For Codex

When asking Codex to help with a forecasting or AI project, define it like this:

```text
Project name:
Dataset:
Prediction target:
Time granularity:
Forecasting horizon:
Input features:
Output:
Model scope:
Application scenario:
Final use:
First step I want Codex to do:
```

For this project:

```text
Project name: AI-Based Short-Term Electricity Load Forecasting
Dataset: UCI ElectricityLoadDiagrams20112014
Prediction target: Electricity load of a selected client or meter
Time granularity: 15-minute raw data
Forecasting horizon: 1 hour and 24 hours
Input features: historical load, time features, weekday/weekend, holiday flag, lag features, rolling statistics
Output: future electricity load value
Model scope: naive baseline, seasonal naive, Ridge, Random Forest, XGBoost or LightGBM, later LSTM/GRU
Application scenario: smart grid dispatching, building energy management, storage and price optimization
Final use: GitHub portfolio, professor outreach, and Hong Kong green technology job applications
First step I want Codex to do: create project structure and EDA notebook
```

## Portfolio Positioning

This project connects a Computer Science background with green energy applications. It can be described as an AI-driven smart energy project that uses time-series forecasting and robustness evaluation to support reliable energy management decisions.

## Publish To GitHub

Follow `GITHUB_GUIDE.md`. Raw data and virtual environments are excluded, while
the executed notebook, charts, and summary table are included for reviewers.
