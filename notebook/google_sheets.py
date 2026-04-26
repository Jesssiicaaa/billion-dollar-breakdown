"""
Phase 2e — Google Sheets Live Dashboard
Unicorn Companies Project
------------------------------------------------
Input:  data/unicorn_companies_clean.csv
Output: A live Google Sheet pushed via the Sheets API

Run:  python3 notebook/google_sheets.py

────────────────────────────────────────────────
ONE-TIME SETUP (do this before running the script)
────────────────────────────────────────────────
1. Go to https://console.cloud.google.com
2. Create a new project (e.g. "unicorn-project")
3. Enable these two APIs:
     - Google Sheets API
     - Google Drive API
4. Go to IAM & Admin → Service Accounts → Create Service Account
     - Name it anything (e.g. "unicorn-sheets-bot")
     - Role: Editor
5. Click the service account → Keys → Add Key → JSON
     - Download the file and rename it: credentials.json
     - Place credentials.json in the SAME folder as this script
6. Install dependencies:
     pip install gspread gspread-formatting pandas

That's it. The script handles everything else automatically.
────────────────────────────────────────────────

Sheets created inside the workbook:
  1. Overview         — key metrics and project description
  2. Raw Data         — full 1,074-row cleaned dataset
  3. Industry Summary — aggregated stats per industry
  4. Country Summary  — top 30 countries
  5. Continent Summary— continental breakdown
  6. Yearly Trends    — unicorn creation over time
"""

import pandas as pd
import numpy as np
import gspread
from gspread_formatting import (
    format_cell_range, CellFormat, TextFormat, Color,
    set_frozen, set_column_width
)
from google.oauth2.service_account import   Credentials
import time
import sys
import os

from googleapiclient.discovery import build

# ── Config ────────────────────────────────────────────────────────────────────
CREDENTIALS_FILE = "notebook/credentials.json"
SPREADSHEET_NAME = "Unicorn Companies — Data Analysis Portfolio"

# Colour palette (RGB 0-1 scale for Google Sheets API)
DARK_GREEN  = Color(0.102, 0.235, 0.204)   # #1A3C34
MID_GREEN   = Color(0.176, 0.416, 0.310)   # #2D6A4F
ACCENT      = Color(0.322, 0.718, 0.533)   # #52B788
LIGHT_GREEN = Color(0.847, 0.953, 0.863)   # #D8F3DC
WHITE       = Color(1, 1, 1)
LIGHT_GRAY  = Color(0.973, 0.976, 0.980)   # #F8F9FA

# ── Preflight check ───────────────────────────────────────────────────────────
if not os.path.exists(CREDENTIALS_FILE):
    print("ERROR: credentials.json not found in the current directory.")
    print("Follow the ONE-TIME SETUP steps at the top of this file.")
    sys.exit(1)

if not os.path.exists("data/unicorn_companies_clean.csv"):
    print("ERROR: data/unicorn_companies_clean.csv not found.")
    print("Run notebook/clean.py first.")
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
    "Avg Years to Unicorn", "% High Value (≥$10B)",
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
    "Total Funding ($B)", "Avg Years to Unicorn", "# High Value (≥$10B)",
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

# Raw display columns
raw_cols = [
    "company", "valuation_b", "funding_b", "industry",
    "country", "continent", "year_founded", "year_joined",
    "years_to_unicorn", "funding_efficiency", "is_high_value",
]
raw_display = df[raw_cols].copy()
raw_display["date_joined"] = df["date_joined"].dt.strftime("%Y-%m-%d")
raw_display.columns = [
    "Company", "Valuation ($B)", "Funding ($B)", "Industry",
    "Country", "Continent", "Year Founded", "Year Joined",
    "Years to Unicorn", "Funding Efficiency", "High Value?", "Date Joined",
]
# Replace NaN with empty string for Sheets compatibility
raw_display = raw_display.fillna("")

# ── 3. Authenticate ───────────────────────────────────────────────────────────
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
creds  = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
client = gspread.authorize(creds)

# ── Create spreadsheet via Drive API (puts it in your Drive, not service account's) ──
drive_service = build("drive", "v3", credentials=creds)

file_metadata = {
    "name":     SPREADSHEET_NAME,
    "mimeType": "application/vnd.google-apps.spreadsheet",
}
file = drive_service.files().create(body=file_metadata, fields="id").execute()
spreadsheet_id = file.get("id")

# Share it with your personal Gmail immediately
drive_service.permissions().create(
    fileId=spreadsheet_id,
    body={
        "type":         "user",
        "role":         "writer",
        "emailAddress": "your.personal@gmail.com",  # ← replace this
    },
    sendNotificationEmail=False,
).execute()

# Also make it publicly readable (for your README link)
drive_service.permissions().create(
    fileId=spreadsheet_id,
    body={"type": "anyone", "role": "reader"},
).execute()

# Open it with gspread using the ID
sh = client.open_by_key(spreadsheet_id)
sh.sheet1.update_title("Overview")
print(f"Created and shared spreadsheet: '{SPREADSHEET_NAME}'")
print(f"URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")

# ── 4. Create or open spreadsheet ────────────────────────────────────────────
try:
    sh = client.open(SPREADSHEET_NAME)
    print(f"Opened existing spreadsheet: '{SPREADSHEET_NAME}'")
    for ws in sh.worksheets():
        if ws.title != "Sheet1":
            sh.del_worksheet(ws)
    sh.sheet1.clear()
    sh.sheet1.update_title("Overview")
except gspread.SpreadsheetNotFound:
    sh = client.create(SPREADSHEET_NAME)
    sh.sheet1.update_title("Overview")
    print(f"Created new spreadsheet: '{SPREADSHEET_NAME}'")

# Add this line immediately after — before any other operations
sh.share("your.jessicaolaniyiii@gmail.com", perm_type="user", role="writer")
sh.share(None, perm_type="anyone", role="reader")  # public read link

# ── Helper: rate-limit-safe batch update ──────────────────────────────────────
def safe_update(ws, data, range_start="A1"):
    """Push data in one batch call, convert all values to Sheets-safe types."""
    def clean(v):
        if isinstance(v, (np.integer,)):   return int(v)
        if isinstance(v, (np.floating,)):  return float(v) if not np.isnan(v) else ""
        if isinstance(v, float) and np.isnan(v): return ""
        if pd.isna(v) if not isinstance(v, (list, dict)) else False: return ""
        return v

    cleaned = [[clean(cell) for cell in row] for row in data]
    ws.update(range_start, cleaned)
    time.sleep(1.2)   # stay under 60 writes/min quota

# ── Helper: style a header row ────────────────────────────────────────────────
def style_header(ws, n_cols, row=1, bg=DARK_GREEN):
    col_letter = chr(ord("A") + n_cols - 1)
    fmt = CellFormat(
        backgroundColor=bg,
        textFormat=TextFormat(bold=True, foregroundColor=WHITE, fontSize=11),
        horizontalAlignment="CENTER",
    )
    format_cell_range(ws, f"A{row}:{col_letter}{row}", fmt)
    time.sleep(0.5)

# ── Helper: style data rows (alternating) ────────────────────────────────────
def style_data_rows(ws, n_rows, n_cols, start_row=2):
    col_letter = chr(ord("A") + n_cols - 1)
    for i in range(n_rows):
        row_num = start_row + i
        bg = LIGHT_GREEN if i % 2 == 0 else WHITE
        fmt = CellFormat(backgroundColor=bg)
        format_cell_range(ws, f"A{row_num}:{col_letter}{row_num}", fmt)
    time.sleep(0.5)

# ── Helper: push a full DataFrame to a sheet ──────────────────────────────────
def push_dataframe(ws, df_in, header_color=MID_GREEN):
    headers = [df_in.columns.tolist()]
    rows    = df_in.values.tolist()
    safe_update(ws, headers + rows)
    set_frozen(ws, rows=1)
    style_header(ws, len(df_in.columns), row=1, bg=header_color)
    style_data_rows(ws, len(rows), len(df_in.columns))
    # Set reasonable column widths
    for i in range(len(df_in.columns)):
        col_letter = chr(ord("A") + i)
        max_len = max(
            len(str(df_in.columns[i])),
            df_in.iloc[:, i].astype(str).str.len().max() if len(df_in) else 0,
        )
        set_column_width(ws, col_letter, min(max(max_len * 8 + 20, 80), 280))
    time.sleep(0.8)

# ── 5. Sheet 1: Overview ──────────────────────────────────────────────────────
print("Writing Overview sheet...")
ws_overview = sh.sheet1

overview_data = [
    ["Unicorn Companies — Data Analysis Portfolio"],
    [""],
    ["Project",        "Global Unicorn Landscape: What Makes a $1B+ Company?"],
    ["Author",         ""],   # ← add your name
    ["Dataset",        "data/unicorn_companies_clean.csv — 1,074 companies, 46 countries"],
    ["Last Updated",   pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
    [""],
    ["── Key Metrics ──", ""],
    ["Total unicorns",          f"{len(df):,}"],
    ["Countries represented",   f"{df['country'].nunique()}"],
    ["Industries",              f"{df['industry'].nunique()}"],
    ["Avg valuation",           f"${df['valuation_b'].mean():.1f}B"],
    ["Median valuation",        f"${df['valuation_b'].median():.1f}B"],
    ["Largest unicorn",         f"{df.loc[df['valuation_b'].idxmax(), 'company']} (${df['valuation_b'].max():.0f}B)"],
    ["High-value (≥$10B)",      f"{df['is_high_value'].sum()} companies ({df['is_high_value'].mean()*100:.1f}%)"],
    ["Avg years to unicorn",    f"{df['years_to_unicorn'].mean():.1f} years"],
    [""],
    ["── Tools Used ──", ""],
    ["Python",     "pandas, numpy, scikit-learn, XGBoost, Prophet, gspread"],
    ["SQL",        "SQLite — aggregations, rankings, time-to-unicorn analysis"],
    ["Excel",      "openpyxl — styled pivot tables and embedded charts"],
    ["Google Sheets", "gspread API — live collaborative dashboard (this file)"],
    ["Tableau",    "Interactive dashboard — map, treemap, ML results"],
    [""],
    ["── Sheets in this workbook ──", ""],
    ["Raw Data",          "Full 1,074-row cleaned dataset"],
    ["Industry Summary",  "Aggregated stats per industry"],
    ["Country Summary",   "Top 30 countries by unicorn count"],
    ["Continent Summary", "Continental breakdown"],
    ["Yearly Trends",     "Unicorn creation over time"],
]

safe_update(ws_overview, overview_data)

# Style the title cell
format_cell_range(ws_overview, "A1", CellFormat(
    textFormat=TextFormat(bold=True, fontSize=16, foregroundColor=DARK_GREEN),
))
# Style section headers
for row_idx, row in enumerate(overview_data, start=1):
    if row and str(row[0]).startswith("──"):
        format_cell_range(ws_overview, f"A{row_idx}", CellFormat(
            textFormat=TextFormat(bold=True, foregroundColor=MID_GREEN),
        ))
# Style metric labels
format_cell_range(ws_overview, "A9:A16", CellFormat(
    textFormat=TextFormat(bold=True),
))

set_column_width(ws_overview, "A", 220)
set_column_width(ws_overview, "B", 380)
time.sleep(0.8)
print("  Overview ✓")

# ── 6. Remaining sheets ───────────────────────────────────────────────────────
sheets_to_create = [
    ("Raw Data",          raw_display,        DARK_GREEN),
    ("Industry Summary",  industry_summary,   MID_GREEN),
    ("Country Summary",   country_summary,    MID_GREEN),
    ("Continent Summary", continent_summary,  MID_GREEN),
    ("Yearly Trends",     yearly_trends,      MID_GREEN),
]

for sheet_name, data, hdr_color in sheets_to_create:
    print(f"Writing {sheet_name}...")
    ws = sh.add_worksheet(title=sheet_name, rows=max(len(data)+10, 50), cols=max(len(data.columns)+2, 15))
    push_dataframe(ws, data, header_color=hdr_color)
    print(f"  {sheet_name} ✓  ({len(data):,} rows × {len(data.columns)} cols)")

# ── 7. Add a sparkline formula to Yearly Trends ───────────────────────────────
# Sparkline in Google Sheets is a native formula — great showcase!
print("\nAdding sparkline to Yearly Trends...")
ws_trends = sh.worksheet("Yearly Trends")
n_trends = len(yearly_trends) + 1

# Add sparkline header and formula in column F
ws_trends.update("F1", [["Trend (sparkline)"]])
ws_trends.update("F2", [[f"=SPARKLINE(B2:B{n_trends},{{\"charttype\",\"line\";\"color\",\"#2D6A4F\";\"linewidth\",2}})"]])
format_cell_range(ws_trends, "F1", CellFormat(
    textFormat=TextFormat(bold=True, foregroundColor=WHITE, fontSize=11),
    backgroundColor=MID_GREEN,
    horizontalAlignment="CENTER",
))
time.sleep(0.5)
print("  Sparkline added ✓")

# ── 8. Final summary ─────────────────────────────────────────────────────────
print("\n── Done ──────────────────────────────────────────────────────────")
print(f"Spreadsheet: {sh.url}")
print(f"Sheets created: {[ws.title for ws in sh.worksheets()]}")
print("\nTip: Copy this URL into your GitHub README so reviewers can")
"""remove"""
print("open the live sheet without downloading anything.")