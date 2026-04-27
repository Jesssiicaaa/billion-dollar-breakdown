# Global Unicorn Landscape: Billion Dollar Breakdown
### What Makes a $1B+ Company?

A full-stack data analysis and machine learning project examining 1,073 unicorn companies across 46 countries — built to demonstrate end-to-end data skills for data analyst and data science intern roles.

---

## Live Dashboards

| Dashboard | Link | Description |
|---|---|---|
| Tableau | *(add after publishing to Tableau Public)* | Interactive map, treemap, scatter, ML results, forecast |
| Google Sheets | *(add after running google_sheets.py)* | Live dataset with summaries and sparklines |

---

## Project Overview

This project investigates the global unicorn ecosystem — private companies valued at $1B or more — using every tool in the modern data stack.

**Research question:** Which industries, geographies, and founding patterns produce the highest-value unicorn companies relative to funding raised?

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python (pandas, numpy) | Data cleaning and feature engineering |
| Python (scikit-learn, XGBoost) | Machine learning models |
| Python (Prophet) | Time series forecasting |
| Python (matplotlib, seaborn) | Chart generation — 6 EDA charts |
| SQLite + SQL | 10 analytical queries and aggregations |
| Excel (openpyxl) | 2 styled workbooks with pivot tables and embedded charts |
| Google Sheets API (gspread) | Live collaborative dashboard pushed programmatically |
| Tableau | Interactive visual dashboard |
| Jupyter Notebook | Phase 1 exploratory analysis |
| Git + GitHub | Version control and portfolio hosting |

---

## Dataset

**Source:** `unicorn_companies.csv`
**Rows:** 1,073 unicorn companies
**Coverage:** 46 countries, 15 industries, founded 1919–2021

### Engineered features

| Feature | Description |
|---|---|
| `years_to_unicorn` | Years from founding to reaching $1B valuation |
| `funding_efficiency` | Valuation divided by funding raised |
| `is_high_value` | Binary flag: 1 if valuation >= $10B, 0 otherwise |
| `log_valuation` | Log-transformed valuation used as the regression target |
| `decade_founded` | Decade the company was founded |

---

## Project Structure

```
unicorn-analysis/
│
├── data/
│   ├── unicorn_companies.csv              # raw dataset
│   └── unicorn_companies_clean.csv        # cleaned and feature-engineered
│
├── notebook/
│   ├── phase1_eda.ipynb                   # Phase 1 — exploratory analysis
│   ├── clean.py                           # Phase 2a — cleaning and features
│   ├── sql_export.py                      # Phase 2b — export to SQLite
│   ├── excel_export.py                    # Phase 2c — export to Excel
│   ├── google_sheets.py                   # Phase 2e — push to Google Sheets API
│   ├── sql_analysis.py                    # Phase 3 Step 1 — 10 SQL queries
│   ├── eda_charts.py                      # Phase 3 Step 2 — 6 EDA charts
│   ├── pivot_tables.py                    # Phase 3 Step 3 — Excel pivot tables
│   ├── findings.py                        # Phase 3 Step 4 — findings report
│   ├── regression.py                      # Phase 4 Step 1 — valuation prediction
│   ├── binary_classifier.py               # Phase 4 Step 2 — high-value classifier
│   ├── multiclass_classifier.py           # Phase 4 Step 3 — continent classifier
│   ├── clustering.py                      # Phase 4 Step 4 — K-Means clustering
│   ├── timeseries.py                      # Phase 4 Step 5 — Prophet forecast
│   ├── model_comparison.py                # Phase 4 Step 6 — unified results
│   └── tableau_exports.py                 # Phase 5 Step 1 — Tableau-ready exports
│
├── charts/                                # 6 EDA chart PNGs
├── sql_results/                           # 10 query result CSVs
├── ml_outputs/                            # model results, charts, cluster assignments
├── tableau_exports/                       # clean CSVs for Tableau
├── outputs/                               # Excel workbooks
│
├── run_all.py                             # master pipeline — runs all phases in order
├── requirements.txt                       # all Python dependencies
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/unicorn-analysis.git
cd unicorn-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the raw dataset in the data/ folder

# 4. Run the full pipeline
python run_all.py

# Run a specific phase only
python run_all.py --phase 3
python run_all.py --from 4
```

> **Google Sheets:** Requires a one-time setup. Create a blank Google Sheet, share it with the `client_email` from `credentials.json`, paste the Spreadsheet ID into `google_sheets.py`, then run. Full instructions are in the script header.

> **Note:** Add `credentials.json` and `token.pickle` to your `.gitignore` — never commit credentials to GitHub.

---

## Key Findings

### Market overview
- **1,073** unicorn companies with a combined valuation of **$3,710B**
- Median valuation of **$2B** vs mean of **$3.5B** — the market is heavily right-skewed, driven by a small number of mega-unicorns pulling the average up
- Only **5.7% (61 companies)** have crossed the $10B threshold
- **Bytedance** is the most valuable at $180B — nearly double the next company

### Geography
- The **United States (52%)** and **China (16%)** account for 68% of all unicorns globally
- **India (6%)** is emerging as a third major hub, particularly in Fintech
- **North America (55%)**, **Asia (29%)**, and **Europe (13%)** dominate the continental split
- Africa and Oceania are significantly underrepresented relative to GDP — a gap worth watching

### Industry
- **Fintech** produces the most unicorns (224 companies, 21% of the dataset)
- **Artificial Intelligence** companies have the highest average valuation at **$4.5B**
- **Internet software & services** is the most capital-efficient industry at **28.5x** (valuation per $1B raised)

### Speed to unicorn
- The median path to unicorn status takes **6 years** from founding
- **10.5% (113 companies)** reached $1B within 2 years of founding
- **Auto & transportation** is the fastest industry at a median of **4.0 years**
- The slowest company — Otto Bock HealthCare — took **98 years**

### Growth trends
- **2021** was the peak year with **520 new unicorns** — driven by the post-pandemic VC surge and historically low interest rates
- Unicorn creation post-2015 was **45.7x higher** than pre-2015, reflecting the rapid maturation of the global venture capital ecosystem

---

## Charts & Visualisations

### Phase 3 — EDA Charts (Python: matplotlib & seaborn)

**Chart 1 — Valuation distribution**
Two panels side by side. The left panel is a log-scale histogram showing that the vast majority of unicorns cluster between $1B and $3B, with a long right tail toward ByteDance at $180B. The right panel is a box plot broken down by continent, also on a log scale, showing the spread and median valuation per region. Log scale is used on both because without it the extreme outliers compress everything else to the bottom and the distribution becomes unreadable. This chart answers: *how skewed is the valuation data and how does it differ by region?*

**Chart 2 — Top 15 countries by unicorn count**
A horizontal bar chart ranked from most to fewest unicorns. The top 3 bars (US, China, India) are highlighted in a darker green to draw the eye to the dominant players. Value labels sit at the end of each bar. Horizontal orientation is used because country names are long and would overlap on a vertical axis. This chart answers: *which countries dominate unicorn production?*

**Chart 3 — Industry breakdown (dual axis)**
A horizontal bar chart with a dot overlay. The bars show the number of companies per industry and the dots show the average valuation per industry — two different scales on two different axes. This dual-axis design lets you see two stories simultaneously: Fintech is the biggest industry by count, but AI commands the highest average valuation. Without the dual axis you would need two separate charts and the contrast would be harder to see. This chart answers: *which industries are biggest and which are most valuable?*

**Chart 4 — Funding vs valuation scatter**
A scatter plot with funding raised on the X axis and valuation on the Y axis, both on log scales, with each dot coloured by continent. The top 8 most valuable companies are annotated by name. Log scales are essential here — on a linear scale ByteDance and SpaceX push everything else into the bottom-left corner. The colour encoding by continent lets you see geographic clustering patterns without needing a separate chart. This chart answers: *is there a relationship between how much a company raises and how valuable it becomes?*

**Chart 5 — Unicorns per year (area chart)**
An area chart showing the count of new unicorns per year from 2007 to 2022, with a dashed secondary line showing the average valuation trend. The filled area under the count line emphasises volume accumulation over time. The 2021 spike to 520 is the most prominent feature and is annotated. The dual axis lets you see whether the 2021 surge was accompanied by higher or lower average valuations. This chart answers: *is unicorn creation accelerating and was 2021 an anomaly?*

**Chart 6 — Years to unicorn by industry (box plot)**
A horizontal box plot showing the distribution of years-to-unicorn for each industry, sorted by median from fastest to slowest. Box plots are used here rather than bars because the spread and outliers are as important as the median — some industries have very consistent timelines and others are wildly variable. A gradient fill from light to dark green indicates ranking. This chart answers: *which industries produce unicorns fastest and how consistent are they?*

---

### Phase 4 — ML Output Charts (Python: matplotlib)

**Feature importance chart (Step 1 — Regression)**
A horizontal bar chart showing how much each feature contributed to the XGBoost valuation prediction model. Features are ranked from most to least important. The most important feature bar is highlighted in a brighter accent colour. This is one of the most interview-friendly charts in the project because it translates a complex model into a simple ranked list that anyone can understand. This chart answers: *what actually drives a unicorn's valuation according to the model?*

**Actual vs predicted scatter — 3 panels (Step 1 — Regression)**
Three side-by-side scatter plots, one per model (Linear Regression, Random Forest, XGBoost), each showing predicted log-valuation against actual log-valuation on the test set. A perfect model would produce all dots along the diagonal dashed line. The R² score is shown in each panel title. The three-panel layout lets you see at a glance how much each model improves over the baseline. This chart answers: *how accurately does each model predict valuation and where does it fail?*

**Confusion matrices — 3 panels (Step 2 — Binary Classifier)**
Three side-by-side heatmaps, one per model, showing the count of true positives, true negatives, false positives, and false negatives. Darker green = higher count. The key thing to look for is false negatives (bottom-left cell) — predicting a company is not high-value when it actually is. This chart answers: *where does each classifier make mistakes and what type of mistakes?*

**ROC curves (Step 2 — Binary Classifier)**
All three models plotted on the same axes. The X axis is false positive rate, the Y axis is true positive rate. A perfect model hugs the top-left corner. The diagonal dashed line represents a random classifier with AUC=0.5. Each model's AUC is shown in the legend. Plotting all three together makes comparison immediate. This chart answers: *how well can each model distinguish high-value from standard unicorns across all classification thresholds?*

**Precision-recall curves (Step 2 — Binary Classifier)**
Similar to the ROC curve but more honest for imbalanced datasets. With only 61 high-value companies out of 1,073, the ROC curve can look deceptively good. The precision-recall curve penalises models that achieve high recall by flooding predictions — it shows the real trade-off between catching high-value companies and avoiding false alarms. This chart answers: *at what precision can each model reliably identify high-value unicorns?*

**Confusion matrix heatmap (Step 3 — Multiclass Classifier)**
A single heatmap for the best-performing model showing predicted vs actual continent. The colour represents the row-normalised proportion (how often each actual class was predicted as each other class) and the number inside each cell shows the raw count. This dual encoding lets you see both the pattern and the scale. This chart answers: *which continents does the model confuse with each other?*

**Per-class F1 grouped bar chart (Step 3 — Multiclass Classifier)**
A grouped bar chart with one group per continent and one bar per model within each group. This lets you compare all three models across all six classes simultaneously. Low-sample classes (Africa n=3, Oceania n=8) are annotated with a red warning label so the reader understands the low F1 scores for those classes are a data limitation, not a model failure. This chart answers: *which continents can be reliably predicted and which cannot?*

**Elbow method and silhouette scores (Step 4 — Clustering)**
Two side-by-side line charts. The left shows inertia (within-cluster variance) against K — you look for the elbow where adding more clusters stops helping. The right shows silhouette score against K — higher is better. A vertical dashed line marks the chosen K on both charts. Showing both together justifies the K choice more rigorously than either method alone. This chart answers: *how many clusters best fit the data?*

**PCA cluster scatter (Step 4 — Clustering)**
A scatter plot of all 1,073 companies reduced to 2 dimensions via PCA, coloured by cluster archetype. The top 5 most valuable companies are annotated by name. The variance explained by each PCA component is shown on the axis label. This is the most storytelling-friendly chart in the project — it turns an abstract algorithm output into a visual that non-technical audiences can immediately grasp. This chart answers: *what do the unicorn archetypes actually look like when visualised?*

**Cluster profiles bar chart (Step 4 — Clustering)**
Four small bar charts side by side, one per key metric (avg valuation, avg funding, avg years to unicorn, avg funding efficiency), each showing the value per cluster. This lets you characterise each archetype numerically. This chart answers: *what are the defining characteristics of each cluster?*

**Prophet forecast (Step 5 — Time Series)**
A line chart showing the historical monthly unicorn count as dots, the forecast as a solid line, and the 90% confidence interval as a shaded band. A vertical dashed line separates history from forecast. The 2021 spike is annotated. The confidence band widens into the future, which is honest — uncertainty grows the further out you forecast. This chart answers: *how many new unicorns should we expect per year over the next 3 years?*

**Prophet components (Step 5 — Time Series)**
Two stacked charts showing the trend component and the yearly seasonality component separately. Decomposing the forecast into components shows what the model actually learned — is the long-term trend rising or falling, and are certain months consistently higher than others? This chart answers: *what is the underlying trend and is there seasonal variation in unicorn creation?*

**Model comparison chart (Step 6 — Summary)**
Six panels arranged in a grid, each showing a different metric across the three supervised models. The layout lets a reader scan all model results in one view without switching between charts. This chart answers: *across all tasks, which model type performed best and by how much?*

---

### Phase 5 — Tableau Dashboard

**World map** — filled map coloured by total valuation per country, with unicorn count as a label. Shows geographic concentration immediately.

**Industry treemap** — rectangles sized by company count and coloured by average valuation. Shows both volume and value in one view.

**Valuation by continent box plot** — log-scale box plot showing the spread of valuations per continent, sortable by median.

**Top 15 countries bar chart** — horizontal bars with the top 3 highlighted in amber.

**Unicorns per year area chart** — with a reference line annotation on the 2021 spike.

**Funding vs valuation scatter** — log-log scatter coloured by continent with company-level tooltips.

**4 KPI tiles** — total unicorns, combined valuation, average years to unicorn, high-value count.

**Feature importance bar** — from the regression model results, coloured by model type.

**Cluster scatter** — funding vs valuation coloured by archetype with archetype labels.

**Prophet forecast line** — history in green, forecast in amber, confidence band shaded.

All charts are connected to continent, industry, and year range filters so clicking any element cross-filters the entire dashboard.

---

## Machine Learning Models

### Model 1 — Valuation Prediction (Regression)

Predicts `log(valuation)` from funding raised, industry, continent, years to unicorn, and founding decade. Log transformation applied to the target because raw valuation is heavily right-skewed — without it the model focuses almost entirely on the handful of mega-unicorns.

| Model | R² | RMSE | CV R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost** | **0.978** | **0.096** | **—** |

> Funding raised and industry are the strongest predictors. XGBoost significantly outperforms Linear Regression because non-linear interactions between features (e.g. the combination of industry and continent) cannot be captured by a linear model. Back-transform predictions with `np.expm1()` to get dollar values.

---

### Model 2 — High-Value Unicorn Classifier (Binary)

Predicts whether a company will reach ≥$10B. Target class is only 5.7% of the data (61 companies) — a severe class imbalance. Handled with `class_weight='balanced'` for scikit-learn models and `scale_pos_weight` for XGBoost. Stratified train/test split used to preserve the class ratio.

| Model | ROC-AUC | F1 | Avg Precision | CV F1 |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| **XGBoost** | **1.00** | **0.96** | **—** | **—** |

> ROC-AUC of 1.0 reflects the small test set size (~12 high-value companies in test). This is a strong result but should not be interpreted as a perfect model. The precision-recall curve is the more conservative and informative metric for this imbalanced problem.

---

### Model 3 — Continent Classifier (Multiclass)

Predicts which of 6 continents a unicorn belongs to based on industry, valuation, funding, and years to unicorn. Africa (n=3) and Oceania (n=8) have too few samples for reliable classification and are explicitly flagged in the output.

| Model | Accuracy | F1-macro | F1-weighted |
|---|---|---|---|
| Logistic Regression | — | — | — |
| **Random Forest** | **0.586** | **0.283** | **—** |
| XGBoost | — | — | — |

> F1-macro weights all 6 classes equally which heavily penalises Africa and Oceania misclassification despite their tiny sample sizes. F1-weighted, which accounts for class frequency, is the more practical metric for this model. North America and Asia are reliably predicted. The low-sample classes are a data limitation not a model failure.

---

### Model 4 — Unicorn Clustering (Unsupervised)

K-Means clustering on valuation, funding, years to unicorn, and funding efficiency. Funding efficiency was capped at the 99th percentile before fitting to prevent a single outlier (a company with $0 funding recorded) from dominating its own cluster. Optimal K chosen by comparing elbow method and silhouette scores simultaneously.

| Archetype | Count | Avg Valuation | Avg Funding | Avg Yrs to Unicorn |
|---|---|---|---|---|
| Steady growers | 958 | $2.6B | $0.5B | 5.9 |
| Mega-unicorns | 115 | $10.5B | $1.1B | 16.5 |

> Mega-unicorns take nearly 3x longer to reach status but arrive at 4x the valuation — suggesting a fundamentally different growth model from the typical unicorn. The PCA scatter with archetype labels is the most storytelling-friendly output in the project.

---

### Model 5 — Unicorn Creation Forecast (Prophet)

Monthly time series of new unicorn creation, forecast 3 years ahead with a 90% confidence interval. The 2021 surge (520 unicorns — nearly 5x the surrounding years) is handled as an explicit regressor so Prophet does not try to model it as natural seasonal behaviour.

| Year | Predicted | 90% Interval |
|---|---|---|
| 2023 | ~435 | 333 – 531 |
| 2024 | ~539 | 439 – 635 |
| 2025 | ~200 | 164 – 233 |

> The post-2021 reversion to mean is the expected and honest result. The widening confidence band into the future reflects genuine uncertainty — not a model weakness. Always mention the 2021 anomaly when presenting this forecast.

---

## SQL Analysis

10 queries against `unicorn.db` (SQLite). All results saved as CSVs in `sql_results/`.

| Query | What it finds |
|---|---|
| Q01 | Top 20 most valuable unicorns |
| Q02 | Country ranking by unicorn count and avg valuation |
| Q03 | Industry ranking by total valuation |
| Q04 | Funding efficiency leaders |
| Q05 | Fastest companies to reach unicorn status |
| Q06 | High-value rate by industry |
| Q07 | Unicorn creation by year |
| Q08 | Continent × industry cross-tab |
| Q09 | Late bloomers — founded before 2000, joined after 2015 |
| Q10 | Decade-over-decade founding patterns |

---

## Excel Outputs

| File | Contents |
|---|---|
| `unicorn_analysis.xlsx` | 5 styled sheets: raw data, industry summary, country summary, continent summary, yearly trends — with 2 embedded charts |
| `unicorn_pivot_tables.xlsx` | 5 pivot tables: industry × continent matrix with colour-scale heatmap, valuation tiers with data bars, decade × industry matrix, country league table with top-3 highlight, yearly growth with YoY % column |

Both files are built programmatically with openpyxl — conditional formatting, zebra row striping, frozen header rows, auto-fitted column widths, and embedded charts are all generated in code, not applied by hand.

---

## Google Sheets

The cleaned dataset and all summary tables are pushed to a live Google Sheet via the gspread API. The script uses batched API calls (`batch_update`) instead of row-by-row formatting to stay within Google's rate limit of 60 write requests per minute. A native `SPARKLINE()` formula is injected into the Yearly Trends sheet.

Key technical decisions made during implementation:
- Switched from service account Drive creation to opening a pre-existing shared sheet to avoid service account storage quota errors
- Replaced row-by-row `format_cell_range()` loops with a single `batch_update` call to fix 429 rate limit errors
- Fixed `ws.update()` argument order to use named parameters for gspread v6+ compatibility

---

## Limitations

- **Dataset size:** 1,073 rows is sufficient for tree-based ML but small for deep learning. Low-sample classes (Africa: 3, Oceania: 8) are flagged throughout and excluded from reliability claims.
- **Valuation accuracy:** Unicorn valuations are based on last funding round — they are estimates, not market prices.
- **2021 anomaly:** The post-pandemic VC surge is a historically unusual event treated explicitly in all trend analysis and forecasting.
- **Correlation vs causation:** This project identifies associations, not causal mechanisms. Funding efficiency predicting valuation does not mean raising less money causes higher valuations.
- **Static dataset:** The data has a fixed cutoff — new unicorns created after the dataset was compiled are not included.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
scikit-learn>=1.3.0
xgboost>=2.0.0
prophet>=1.1.4
gspread>=5.11.0
gspread-formatting>=1.1.2
google-auth>=2.22.0
notebook>=7.0.0
```