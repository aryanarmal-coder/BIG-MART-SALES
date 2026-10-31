# BigMart Sales Prediction

**Roll No:** L003

Predicting the sales of each product in each BigMart store, using the
[Analytics Vidhya Big Mart Sales III](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/) dataset.

## Problem

BigMart collected 2013 sales data for 1,559 products across 10 stores.
The task is to predict `Item_Outlet_Sales` (the **target variable**) for each product–store pair in the test set.
Submissions are scored with **RMSE** (Root Mean Squared Error).

| File | Rows | Columns |
|---|---|---|
| `data/train.csv` | 8,523 | 12 (includes sales) |
| `data/test.csv` | 5,681 | 11 (sales to be predicted) |

## Key finding from EDA

Sales = **price (MRP) × units sold**, and units sold depend almost entirely on the **store**.

| Store type | Average units sold per item |
|---|---|
| Grocery Store (OUT010, OUT019) | ≈ 2.4 |
| Supermarket Type 2 (OUT018) | ≈ 13.9 |
| Supermarket Type 1 (6 stores) | ≈ 15.6 – 17.0 |
| Supermarket Type 3 (OUT027) | ≈ 26.6 |

- Item type, fat content and price range do **not** change units sold (all ≈ 15).
- Every price range sells ≈ 15 units, so sales grow in a **straight line** with price.

![Sales vs MRP](results/sales_vs_mrp.png)

## Data checks

**Missing values:** `Item_Weight` (17%) and `Outlet_Size` (28%) are missing only for whole stores
(weight: OUT019, OUT027; size: OUT010, OUT017, OUT045). The model uses the store directly and
does not use weight or size, so no imputation is needed.

**Outliers:** three tests on units sold.

| Test | Limit | Rows flagged |
|---|---|---|
| IQR (Q3 + 1.5 × IQR) | 39.3 | 107 |
| Mean + 3σ | 43.0 | 54 |
| 99th percentile | 41.0 | 86 |

93 of the 107 IQR outliers are from OUT027, the largest supermarket. They are genuine high sales, so they were kept.

## Method

**Linear Regression on "MRP × Outlet" features** (no intercept).

For each of the 10 stores, a column holds the item's MRP if the row is from that store, and 0 otherwise.
This lets the model learn a separate price multiplier for each store:

```
Predicted sales = (store multiplier) × Item_MRP
```

### Encoding comparison (5-fold cross-validation)

| Encoding of the store | CV RMSE |
|---|---|
| Label encoding (OUT027 → 27) | 1,381.7 |
| One-hot encoding + MRP | 1,129.5 |
| **MRP × Outlet (final)** | **1,072.6** |

Label encoding implies a false order between stores. One-hot gives every store the same price slope.
MRP × Outlet gives each store its own slope, which matches the EDA.

### Why Linear Regression

- The relationship is linear (sales = units × price).
- Ridge and Lasso gave the identical RMSE (1,072.6): with 10 non-overlapping features and 8,523 rows there is no overfitting to correct.
- Tree models (Decision Tree, Random Forest, Gradient Boosting) scored worse (≈ 1,080 – 1,091) and are harder to explain.

## Results

| Metric | Value |
|---|---|
| 5-fold CV RMSE | **1,072.6** (folds: 1,088 / 1,067 / 1,066 / 1,067 / 1,075) |
| Baseline (always predict the average) | 1,706 |
| Improvement over baseline | 37% |
| Public leaderboard RMSE | **1,150.08** |

Learned store multipliers (≈ units sold per item):

| OUT019 | OUT010 | OUT018 | OUT045 | OUT046 | OUT013 | OUT049 | OUT017 | OUT035 | OUT027 |
|---|---|---|---|---|---|---|---|---|---|
| 2.42 | 2.44 | 14.17 | 15.56 | 15.90 | 16.42 | 16.81 | 16.84 | 17.09 | 26.25 |

Leaderboard screenshots are in [`results/`](results/).

## Limitations

The model predicts each store's average units sold. Actual sales vary widely around that average
(OUT027 ranges from about 3 to 57 units per item), and nothing in the data explains that spread,
so the error cannot fall much below ≈ 1,070 with these features.

## How to run

```bash
pip install -r requirements.txt
python bigmart_sales.py
```

Outputs are written to `results/`: `submission.csv` (predictions) and `sales_vs_mrp.png` (chart).
The full printed output of a run is saved in `results/output.txt`.

## Repository structure

```
bigmart-sales-prediction/
├── bigmart_sales.py        # full code: EDA, outlier checks, encoding comparison, model, submission
├── requirements.txt
├── README.md
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
└── results/
    ├── submission.csv
    ├── sales_vs_mrp.png
    ├── output.txt
    ├── leaderboard_score_1150.png
    └── leaderboard_score_1148.jpg
```
