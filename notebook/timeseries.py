"""
Phase 4, Step 5 — Unicorn Creation Forecast (Time Series)
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: ML_outputs/step5_forecast_plot.png
        ML_outputs/step5_components_plot.png
        ML_outputs/step5_forecast_results.csv

Run:  python3 notebook/timeseries.py

Model: Facebook Prophet
  - Monthly unicorn count as the time series
  - 3-year forecast (2022–2025)
  - Seasonality components decomposed and plotted
  - 2021 spike handled with a special regressor note

Note on data: The 2021 spike (520 unicorns in one year vs ~100
in surrounding years) was driven by the post-pandemic VC surge.
Prophet will detect this as an outlier — the forecast will
naturally revert toward a mean. This is honest and expected.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
import os

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

os.makedirs("ML_outputs", exist_ok=True)

DARK   = "#1A3C34"
MID    = "#2D6A4F"
ACCENT = "#52B788"
LIGHT  = "#D8F3DC"
GRAY   = "#6C757D"
BG     = "#F8F9FA"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DEE2E6",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "font.family":       "DejaVu Sans",
    "grid.color":        "#DEE2E6",
    "grid.linewidth":    0.6,
})

# ── 1. Build monthly time series ──────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies_clean.csv", parse_dates=["date_joined"])

monthly = (df.groupby(df["date_joined"].dt.to_period("M"))["company"]
           .count()
           .reset_index())
monthly.columns = ["ds", "y"]
monthly["ds"] = monthly["ds"].dt.to_timestamp()
monthly = monthly.sort_values("ds").reset_index(drop=True)

print(f"Monthly time series: {len(monthly)} data points")
print(f"Date range: {monthly['ds'].min().date()} to {monthly['ds'].max().date()}")
print(f"Mean monthly count : {monthly['y'].mean():.1f}")
print(f"Max monthly count  : {monthly['y'].max()} "
      f"({monthly.loc[monthly['y'].idxmax(), 'ds'].strftime('%Y-%m')})")
print()

# ── 2. Flag 2021 as a known anomaly (covid VC surge) ─────────────────────────
# Prophet handles this better when we explicitly tell it about the event
# via a 0/1 regressor rather than letting it try to model the spike
monthly["covid_vc_surge"] = (
    (monthly["ds"].dt.year == 2021).astype(int)
)

# ── 3. Fit Prophet model ──────────────────────────────────────────────────────
m = Prophet(
    yearly_seasonality   = True,
    weekly_seasonality   = False,
    daily_seasonality    = False,
    seasonality_mode     = "additive",
    changepoint_prior_scale    = 0.3,   # more flexible trend
    seasonality_prior_scale    = 10,
    interval_width       = 0.90,        # 90% confidence interval
)
m.add_regressor("covid_vc_surge")
m.fit(monthly)
print("Prophet model fitted.\n")

# ── 4. Forecast 36 months ahead ───────────────────────────────────────────────
future = m.make_future_dataframe(periods=36, freq="MS")
future["covid_vc_surge"] = (
    (future["ds"].dt.year == 2021).astype(int)
)
forecast = m.predict(future)

# Clip negative forecasts to 0 (can't have negative unicorns)
forecast["yhat"]       = forecast["yhat"].clip(lower=0).round(1)
forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round(1)
forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round(1)

print("Forecast (next 12 months):")
future_only = forecast[forecast["ds"] > monthly["ds"].max()].head(12)
for _, row in future_only.iterrows():
    print(f"  {row['ds'].strftime('%Y-%m')}  "
          f"predicted: {row['yhat']:5.1f}  "
          f"[{row['yhat_lower']:5.1f} – {row['yhat_upper']:5.1f}]")

# ── 5. Cross-validation ───────────────────────────────────────────────────────
print("\nRunning cross-validation...")
try:
    df_cv = cross_validation(
        m,
        initial  = "730 days",
        period   = "90 days",
        horizon  = "180 days",
        parallel = None,
    )
    perf = performance_metrics(df_cv)
    print(f"  RMSE (horizon avg): {perf['rmse'].mean():.2f}")
    print(f"  MAE  (horizon avg): {perf['mae'].mean():.2f}")
    print(f"  MAPE (horizon avg): {perf['mape'].mean()*100:.1f}%")
    cv_success = True
except Exception as e:
    print(f"  Cross-validation skipped: {e}")
    cv_success = False

# ── 6. Charts ─────────────────────────────────────────────────────────────────

# Chart A: Main forecast plot (custom, not Prophet's default)
fig, ax = plt.subplots(figsize=(13, 6))

# Historical actuals
ax.scatter(monthly["ds"], monthly["y"],
           color=DARK, s=22, zorder=5, label="Actual monthly count")

# Forecast line
ax.plot(forecast["ds"], forecast["yhat"],
        color=MID, linewidth=2, label="Forecast")

# Confidence interval
ax.fill_between(forecast["ds"],
                forecast["yhat_lower"],
                forecast["yhat_upper"],
                color=ACCENT, alpha=0.2,
                label="90% confidence interval")

# Vertical line separating history from forecast
cutoff = monthly["ds"].max()
ax.axvline(cutoff, color=GRAY, linestyle="--",
           linewidth=1.2, alpha=0.7)
ax.text(cutoff, ax.get_ylim()[1] * 0.95,
        " Forecast →", fontsize=9, color=GRAY, va="top")

# Annotate 2021 spike
spike_date = pd.Timestamp("2021-01-01")
spike_val  = monthly[monthly["ds"].dt.year == 2021]["y"].max()
ax.annotate("2021 VC surge\n(pandemic-era anomaly)",
            xy=(spike_date, spike_val * 0.8),
            xytext=(pd.Timestamp("2019-01-01"), spike_val * 0.7),
            fontsize=8, color=DARK, style="italic",
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Date")
ax.set_ylabel("New unicorns per month")
ax.set_title("Unicorn creation forecast — Prophet model (3-year horizon)")
ax.legend(fontsize=9, framealpha=0.9)
ax.xaxis.grid(True, alpha=0.4)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("ml_outputs/step5_forecast_plot.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\nSaved → ml_outputs/step5_forecast_plot.png")

# Chart B: Trend + seasonality components
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
fig.suptitle("Prophet decomposition — trend & yearly seasonality",
             fontsize=13, fontweight="bold", color=DARK, y=1.01)

# Trend
ax = axes[0]
ax.plot(forecast["ds"], forecast["trend"],
        color=MID, linewidth=2)
ax.fill_between(forecast["ds"],
                forecast["trend_lower"],
                forecast["trend_upper"],
                color=ACCENT, alpha=0.2)
ax.axvline(cutoff, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)
ax.set_title("Long-term trend component")
ax.set_ylabel("Trend")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.grid(True, alpha=0.4)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)

# Yearly seasonality
ax = axes[1]
if "yearly" in forecast.columns:
    ax.plot(forecast["ds"], forecast["yearly"],
            color=MID, linewidth=2)
    ax.fill_between(forecast["ds"],
                    forecast.get("yearly_lower", forecast["yearly"]),
                    forecast.get("yearly_upper", forecast["yearly"]),
                    color=ACCENT, alpha=0.2)
    ax.set_title("Yearly seasonality component")
    ax.set_ylabel("Seasonality effect")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.grid(True, alpha=0.4)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
else:
    ax.text(0.5, 0.5, "Yearly seasonality not extracted\n(insufficient data)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color=GRAY)
    ax.set_title("Yearly seasonality component")

fig.tight_layout()
fig.savefig("ml_outputs/step5_components_plot.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved → ml_outputs/step5_components_plot.png")

# ── 7. Save forecast table ────────────────────────────────────────────────────
save_cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
forecast[save_cols].rename(columns={
    "ds":         "month",
    "yhat":       "predicted_unicorns",
    "yhat_lower": "lower_90",
    "yhat_upper": "upper_90",
}).to_csv("ml_outputs/step5_forecast_results.csv", index=False)
print("Saved → ml_outputs/step5_forecast_results.csv")

# ── 8. Yearly aggregated forecast ─────────────────────────────────────────────
future_forecast = forecast[forecast["ds"] > cutoff].copy()
future_forecast["year"] = future_forecast["ds"].dt.year
yearly_forecast = (future_forecast.groupby("year")
                   .agg(predicted=("yhat", "sum"),
                        lower=("yhat_lower", "sum"),
                        upper=("yhat_upper", "sum"))
                   .round(0).astype(int))

print("\n── Yearly forecast summary ──────────────────────────────────")
print(yearly_forecast.to_string())
print("\nNote: 2021's 520-unicorn spike is treated as a known anomaly.")
print("The forecast represents a normalised trajectory post-surge.")