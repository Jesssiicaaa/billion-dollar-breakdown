"""
Phase 2e — Google Sheets Live Dashboard
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: A live Google Sheet pushed via the Sheets API

Run:  python3 notebook/google_sheets.py

────────────────────────────────────────────────
ONE-TIME SETUP
────────────────────────────────────────────────
1. Create a blank Google Sheet manually in your Drive
2. Share it with the client_email from credentials.json
   and give it Editor access
3. Copy the Spreadsheet ID from the URL and paste it
   into SPREADSHEET_ID below
4. pip install gspread gspread-formatting pandas
────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import gspread
from gspread_formatting import (
    format_cell_range, CellFormat, TextFormat, Color,
    set_frozen, set_column_width
)
from google.oauth2.service_account import Credentials
import time
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────
CREDENTIALS_FILE = "notebook/credentials.json"
SPREADSHEET_ID = "1IUdiXZ5T7BgNLAhFZWuY-Q7ivnEdlOjRcRbv2vRja5s"   # ← paste your Sheet ID here

# ── Colour palette (RGB 0–1 scale) ────────────────────────────────────────────
DARK_GREEN  = Color(0.102, 0.235, 0.204)
MID_GREEN   = Color(0.176, 0.416, 0.310)
LIGHT_GREEN = Color(0.847, 0.953, 0.863)
WHITE       = Color(1, 1, 1)

# ── Preflight checks ──────────────────────────────────────────────────────────
if not os.path.exists(CREDENTIALS_FILE):
    print("ERROR: credentials.json not found.")
    sys.exit(1)

if not os.path.exists("data/unicorn_companies_clean.csv"):
    print("ERROR: data/unicorn_companies_clean.csv not found.")
    sys.exit(1)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/unicorn_companies_clean.csv", parse_dates=["date_joined"])
print(f"Loaded {len(df):,} rows from data/unicorn_companies_clean.csv")

# ── 2. Build summary tables ───────────────────────────────────────────────────
industry_summary = (
    df.groupby("industry")
    .agg(
        companies          = ("company",          "count"),
        avg_valuation_b    = ("valuation_b",      "mean"),
        median_valuation_b = ("valuation_b",      "median"),
        total_funding_b    = ("funding_b",        "sum"),
        avg_years_to_uni   = ("years_to_unicorn", "mean"),
        high_value_pct     = ("is_high_value",    "mean"),
    )
    .reset_index()
    .sort_values("companies", ascending=False)
    .round(2)
)
industry_summary["high_value_pct"] = (industry_summary["high_value_pct"] * 100).round(1)
industry_summary.columns = [
    "Industry", "# Companies", "Avg Valuation ($B)",
    "Median Valuation ($B)", "Total Funding ($B)",
    "Avg Years to Unicorn", "% High Value (>=10B)",
]

country_summary = (
    df.groupby(["country", "continent"])
    .agg(
        companies       = ("company",     "count"),
        avg_valuation_b = ("valuation_b", "mean"),
        total_funding_b = ("funding_b",   "sum"),
    )
    .reset_index()
    .sort_values("companies", ascending=False)
    .head(30)
    .round(2)
)
country_summary.columns = [
    "Country", "Continent", "# Companies",
    "Avg Valuation ($B)", "Total Funding ($B)",
]

continent_summary = (
    df.groupby("continent")
    .agg(
        companies        = ("company",          "count"),
        avg_valuation_b  = ("valuation_b",      "mean"),
        total_funding_b  = ("funding_b",        "sum"),
        avg_years_to_uni = ("years_to_unicorn", "mean"),
        high_value_count = ("is_high_value",    "sum"),
    )
    .reset_index()
    .sort_values("companies", ascending=False)
    .round(2)
)
continent_summary.columns = [
    "Continent", "# Companies", "Avg Valuation ($B)",
    "Total Funding ($B)", "Avg Years to Unicorn", "# High Value (>=10B)",
]

yearly_trends = (
    df.groupby("year_joined")
    .agg(
        new_unicorns    = ("company",     "count"),
        avg_valuation_b = ("valuation_b", "mean"),
        total_funding_b = ("funding_b",   "sum"),
    )
    .reset_index()
    .dropna(subset=["year_joined"])
    .sort_values("year_joined")
    .round(2)
)
yearly_trends["year_joined"] = yearly_trends["year_joined"].astype(int)
yearly_trends.columns = [
    "Year", "New Unicorns", "Avg Valuation ($B)", "Total Funding ($B)",
]

# Raw data — explicit selection to avoid column mismatch
raw_display = df[[
    "company", "valuation_b", "funding_b", "industry",
    "country", "continent", "year_founded", "year_joined",
    "years_to_unicorn", "funding_efficiency", "is_high_value",
]].copy()
raw_display["date_joined"] = df["date_joined"].dt.strftime("%Y-%m-%d")
raw_display.columns = [
    "Company", "Valuation ($B)", "Funding ($B)", "Industry",
    "Country", "Continent", "Year Founded", "Year Joined",
    "Years to Unicorn", "Funding Efficiency", "High Value?", "Date Joined",
]
raw_display = raw_display.fillna("")

# ── 3. Authenticate & open sheet ─────────────────────────────────────────────
print("\nAuthenticating with Google...")
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds  = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
client = gspread.authorize(creds)
print("Authenticated successfully.")

sh = client.open_by_key(SPREADSHEET_ID)
print(f"Opened: {sh.url}\n")

# Clear all existing sheets and start fresh
for ws in sh.worksheets():
    if ws.title != "Sheet1":
        sh.del_worksheet(ws)
sh.sheet1.clear()
sh.sheet1.update_title("Overview")

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_update(ws, data, range_start="A1"):
    """Push data in one batch call with Sheets-safe types."""
    def clean(v):
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return float(v) if not np.isnan(v) else ""
        if isinstance(v, float) and np.isnan(v): return ""
        try:
            if pd.isna(v): return ""
        except (TypeError, ValueError):
            pass
        return v
    cleaned = [[clean(cell) for cell in row] for row in data]
    # Fixed argument order for newer gspread versions
    ws.update(values=cleaned, range_name=range_start)
    time.sleep(1.2)

def style_header(ws, n_cols, row=1, bg=DARK_GREEN):
    """Style the header row in one API call."""
    col_letter = chr(ord("A") + n_cols - 1)
    fmt = CellFormat(
        backgroundColor=bg,
        textFormat=TextFormat(bold=True, foregroundColor=WHITE, fontSize=11),
        horizontalAlignment="CENTER",
    )
    format_cell_range(ws, f"A{row}:{col_letter}{row}", fmt)
    time.sleep(0.5)

def style_data_rows_batched(ws, n_rows, n_cols, start_row=2):
    """
    Style alternating rows using TWO batch API calls instead of one per row.
    Even rows = light green, odd rows = white.
    This replaces the row-by-row loop that caused the 429 rate limit error.
    """
    col_letter = chr(ord("A") + n_cols - 1)
    end_row    = start_row + n_rows - 1

    # All data rows: white base first (1 call)
    fmt_white = CellFormat(backgroundColor=WHITE)
    format_cell_range(ws, f"A{start_row}:{col_letter}{end_row}", fmt_white)
    time.sleep(0.6)

    # Even rows: light green overlay (1 call using banding via repeatCell)
    # Build a list of even row ranges and format them in one batch_update
    sheet_id = ws._properties["sheetId"]
    even_row_requests = []
    for i in range(n_rows):
        if i % 2 == 0:
            row_idx = start_row - 1 + i   # 0-indexed for the API
            even_row_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId":          sheet_id,
                        "startRowIndex":    row_idx,
                        "endRowIndex":      row_idx + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex":   n_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {
                                "red":   0.847,
                                "green": 0.953,
                                "blue":  0.863,
                            }
                        }
                    },
                    "fields": "userEnteredFormat.backgroundColor",
                }
            })

    if even_row_requests:
        ws.spreadsheet.batch_update({"requests": even_row_requests})
    time.sleep(0.8)

def push_dataframe(ws, df_in, header_color=MID_GREEN):
    """Write a DataFrame to a sheet with header styling and batched row formatting."""
    headers = [df_in.columns.tolist()]
    rows    = df_in.values.tolist()

    # 1 call — write all data
    safe_update(ws, headers + rows)

    # 1 call — freeze header
    set_frozen(ws, rows=1)

    # 1 call — style header row
    style_header(ws, len(df_in.columns), row=1, bg=header_color)

    # 2 calls total — style all data rows (batched, not per-row)
    style_data_rows_batched(ws, len(rows), len(df_in.columns))

    # Column widths — one call per column (unavoidable but small)
    for i in range(len(df_in.columns)):
        col_letter = chr(ord("A") + i)
        max_len = max(
            len(str(df_in.columns[i])),
            df_in.iloc[:, i].astype(str).str.len().max() if len(df_in) else 0,
        )
        set_column_width(ws, col_letter, int(min(max(max_len * 8 + 20, 80), 280)))
        time.sleep(0.1)

    time.sleep(0.8)

# ── 4. Overview sheet ─────────────────────────────────────────────────────────
print("Writing Overview...")
ws_overview = sh.sheet1

overview_data = [
    ["Unicorn Companies - Data Analysis Portfolio"],
    [""],
    ["Project",      "Global Unicorn Landscape: What Makes a $1B+ Company?"],
    ["Author",       "Kess Waters"],
    ["Dataset",      "unicorn_companies_clean.csv - 1,074 companies, 46 countries"],
    ["Last Updated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
    [""],
    ["Key Metrics", ""],
    ["Total unicorns",        f"{len(df):,}"],
    ["Countries represented", f"{df['country'].nunique()}"],
    ["Industries",            f"{df['industry'].nunique()}"],
    ["Avg valuation",         f"${df['valuation_b'].mean():.1f}B"],
    ["Median valuation",      f"${df['valuation_b'].median():.1f}B"],
    ["Largest unicorn",       f"{df.loc[df['valuation_b'].idxmax(), 'company']} (${df['valuation_b'].max():.0f}B)"],
    ["High-value (>=10B)",    f"{df['is_high_value'].sum()} companies ({df['is_high_value'].mean()*100:.1f}%)"],
    ["Avg years to unicorn",  f"{df['years_to_unicorn'].mean():.1f} years"],
    [""],
    ["Tools Used", ""],
    ["Python",        "pandas, numpy, scikit-learn, XGBoost, Prophet, gspread"],
    ["SQL",           "SQLite - aggregations, rankings, time-to-unicorn analysis"],
    ["Excel",         "openpyxl - styled pivot tables and embedded charts"],
    ["Google Sheets", "gspread API - live collaborative dashboard (this file)"],
    ["Tableau",       "Interactive dashboard - map, treemap, ML results"],
    [""],
    ["Sheets in this workbook", ""],
    ["Raw Data",          "Full 1,074-row cleaned dataset"],
    ["Industry Summary",  "Aggregated stats per industry"],
    ["Country Summary",   "Top 30 countries by unicorn count"],
    ["Continent Summary", "Continental breakdown"],
    ["Yearly Trends",     "Unicorn creation over time"],
]

safe_update(ws_overview, overview_data)

# Style title and metric labels
format_cell_range(ws_overview, "A1", CellFormat(
    textFormat=TextFormat(bold=True, fontSize=16, foregroundColor=DARK_GREEN),
))
format_cell_range(ws_overview, "A9:A16", CellFormat(
    textFormat=TextFormat(bold=True),
))
set_column_width(ws_overview, "A", 220)
set_column_width(ws_overview, "B", 380)
time.sleep(0.8)
print("  Overview done")

# ── 5. Remaining sheets ───────────────────────────────────────────────────────
sheets_to_create = [
    ("Raw Data",          raw_display,        DARK_GREEN),
    ("Industry Summary",  industry_summary,   MID_GREEN),
    ("Country Summary",   country_summary,    MID_GREEN),
    ("Continent Summary", continent_summary,  MID_GREEN),
    ("Yearly Trends",     yearly_trends,      MID_GREEN),
]

for sheet_name, data, hdr_color in sheets_to_create:
    print(f"Writing {sheet_name}...")
    ws = sh.add_worksheet(
        title=sheet_name,
        rows=max(len(data) + 10, 50),
        cols=max(len(data.columns) + 2, 15),
    )
    push_dataframe(ws, data, header_color=hdr_color)
    print(f"  {sheet_name} done  ({len(data):,} rows x {len(data.columns)} cols)")

# ── 6. Sparkline on Yearly Trends ─────────────────────────────────────────────
print("\nAdding sparkline to Yearly Trends...")
ws_trends = sh.worksheet("Yearly Trends")
n_trends  = len(yearly_trends) + 1
ws_trends.update(values=[["Trend (sparkline)"]], range_name="F1")
ws_trends.update(
    values=[[f'=SPARKLINE(B2:B{n_trends},{{"charttype","line";"color","#2D6A4F";"linewidth",2}})']],
    range_name="F2"
)
format_cell_range(ws_trends, "F1", CellFormat(
    textFormat=TextFormat(bold=True, foregroundColor=WHITE, fontSize=11),
    backgroundColor=MID_GREEN,
    horizontalAlignment="CENTER",
))
time.sleep(0.5)
print("  Sparkline done")

# ── 7. Done ───────────────────────────────────────────────────────────────────
print("\nDone")
print(f"Spreadsheet : {sh.url}")
print(f"Sheets      : {[ws.title for ws in sh.worksheets()]}")
print("\nCopy this URL into your GitHub README.")