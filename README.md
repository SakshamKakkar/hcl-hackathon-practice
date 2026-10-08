# Smart Factory Insights (HCL hackathon practice project)

A practice project for HCL's campus hackathon. It uses synthetic manufacturing data, built in the same shape as a hackathon submission:

- OEE and bottleneck analysis in SQL
- A predictive-maintenance model
- A downtime-savings what-if, all in one Streamlit app

## Problem

Unplanned machine failures and slow stations cut throughput and raise cost. The plant needs to know three things:

1. Where efficiency is lost.
2. Which machines are about to fail.
3. What reducing downtime is worth.

## User stories

| # | As a... | I want... | So that... |
| --- | --- | --- | --- |
| 1 | Plant manager | OEE per station and the bottleneck | I know where to improve first |
| 2 | Maintenance engineer | A failure-risk score per reading | I fix machines before they break |
| 3 | Finance lead | Rupees saved from cutting downtime | I can justify the investment |

## How to run

```bash
pip install -r requirements.txt
python data/generate_data.py      # creates the two CSVs
python -m src.db                  # loads them into factory.db (SQLite)
python -m src.model               # compares models, trains the best one, saves failure_model.joblib
python -m pytest -q               # 6 tests
streamlit run app.py              # opens the demo in the browser
```

## Approach and trade-offs

- **SQL staging (SQLite):** raw CSV goes into tables, and KPIs are computed in SQL with a CTE and a window function. SQLite needs no server, which suits a one-day build. Production would use a warehouse.
- **OEE = Availability x Performance x Quality:** minutes and units are summed first and divided after. Averaging percentages would weight short and long shifts equally.
- **Bottleneck:** the station with the lowest units per run-hour. It caps the line, so improving any other station adds nothing.
- **Baseline first:** a majority-class dummy model scores F1 = 0. That is the bar to beat.
- **Class imbalance (about 7% failures):** models use `class_weight="balanced"` and are judged by F1, recall and precision. Accuracy is not used, because always predicting "no failure" already gets about 93%.
- **Time-based split:** the model trains on the earlier 80% of readings and is tested on the latest 20%, so it never sees the future.
- **Pipeline:** the imputer, scaler and encoder are fitted on training data only, which prevents leakage.
- **Physics features:** power = torque x angular speed, temperature difference, and wear x torque. Power is the most important feature.
- **Cost-based threshold:** a missed failure is assumed to cost Rs 50,000 and a false alarm Rs 2,000. The threshold is tuned on a validation slice of the training period, never on the test set.

## Results on the held-out latest 20%

| Model (5-fold cross-validated F1 on train) | F1 |
| --- | --- |
| Random Forest | 0.87 |
| Logistic Regression | 0.23 |
| Majority baseline | 0.00 |

Random Forest on the test set (threshold 0.31):

- precision 0.76
- recall 0.94
- F1 0.84
- ROC-AUC 0.98
- 84 failures caught, 5 missed, 26 false alarms

Logistic regression does badly because the failure rules are non-linear bands (power too low or too high). A linear model cannot draw two boundaries on one feature.

## Limitations

- The data is synthetic, so the real plant data would change these numbers.
- The cost figures are assumptions.
- The Random Forest is less interpretable than a linear model. Feature importance is shown to compensate.

## Project structure

```
data/generate_data.py   synthetic data generator
src/db.py               CSV -> SQLite, query helper
src/kpi.py              OEE, bottleneck, downtime, savings (SQL)
src/model.py            features, model comparison, threshold, training, prediction
app.py                  Streamlit demo (3 user stories)
tests/test_core.py      unit tests
CHANGE_DRILLS.md        practice change requests
```
