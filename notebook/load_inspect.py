"""
load_inspect.py
------------------
Phase 1, Step 1 — Load the raw CSV and print a full inspection report.
Run: python3 load_inspect.py | tee report/inspectionReport.txt
"""

import pandas as pd

CSV_PATH = "data/unicorn_companies.csv"

# ── Load ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
"""]reads the csv file into a pandas DataFrame, which is a tabular data structure that allows for easy manipulation and analysis of the data.
The CSV_PATH variable specifies the location of the CSV file to be loaded.
"""


print("=" * 55)
""" 
prints a line of 55 equal signs (=) to the console, which serves as a visual separator for the different sections of the inspection report.
"""
print("  SHAPE")
"""
prints the word "SHAPE". SHAPE is the number of rows and columns. ie SHAPE = (rows, columns)
"""
print("=" * 55)
print(f"  Rows    : {df.shape[0]:,}")
"""prints the number of rows in the DataFrame, formatted with commas for thousands separators. The df.shape[0] expression retrieves
the number of rows from the shape attribute of the DataFrame."""
print(f"  Columns : {df.shape[1]}")
"""prints the number of columns in the DataFrame. The df.shape[1] expression retrieves the number of columns from the shape attribute of
the DataFrame."""

print()
print("=" * 55)
print("  FIRST 5 ROWS")
print("=" * 55)
print(df.head().to_string())
"""prints the first 5 rows of the DataFrame in a string format. The df.head() method retrieves the first 5 rows, and the to_string() method
converts it to a string representation for better readability."""

print()
print("=" * 55)
print("  COLUMN TYPES")
print("=" * 55)
print(df.dtypes.to_string())
"""prints the data types of each column in the DataFrame. The df.dtypes attribute returns a Series containing the data type of each column,
and the to_string() method converts it to a string format."""

print()
print("=" * 55)
print("  MISSING VALUES")
print("=" * 55)
missing = df.isnull().sum()
"""prints the count of missing values for each column in the DataFrame. df.isnull() method returns a DataFrame of the same shapewith boolean values
indicating whether each element is null (missing) or not. The sum() method then counts the number of True values (missing values) for each column."""
missing_pct = (missing / len(df) * 100).round(2)
"""measures the percentage of missing values for each column by dividing the count of missing values by the total number of rows in the DataFrame
and multiplying by 100 to get a percentage. The round(2) method rounds the result to 2 decimal places."""

report = pd.concat([missing.rename("count"), missing_pct.rename("%")], axis=1)
print(report[report["count"] > 0].to_string())
"""prints a report of the missing values in the DataFrame. The results (missing = df.isnull().sum()) are combined into a new DataFrame called report, which is
then filtered to show only columns with missing values and printed in a string format for better readability in the console output."""

print()
print("=" * 55)
print("  NUMERIC SUMMARY")
print("=" * 55)
print(df.describe().round(2).to_string())
"""prints a summary of the numeric columns in the DataFrame. The df.describe() method generates descriptive statistics for the numeric columns,
including count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum values. The round(2) method
rounds the results to 2 decimal places, and the to_string() method converts it to a string format."""

print()
print("=" * 55)
print("  CATEGORICAL SUMMARY")
print("=" * 55)
for col in ["Industry", "Country", "Continent"]:
    print(f"\n  {col} — top 5:")
    """prints the name of the current column being analyzed, followed by "— top 5:", indicating that the following lines will show the top 5 most common values in that column."""
    print(df[col].value_counts().head(5).to_string())
    """prints a summary of the categorical columns in the DataFrame. For each specified column (Industry, Country, Continent), the code calculates the value counts using the value_counts() method, which counts the"""

print()
print("=" * 55)
print("  YEAR FOUNDED — outliers (pre-1990)")
print("=" * 55)
print(df[df["Year Founded"] < 1990][["Company", "Year Founded", "Country"]].to_string())
"""prints a subset of the DataFrame that includes only the rows where the "Year Founded" column has values less than 1990. This is done to identify potential
outliers in the data, as companies founded before 1990 may be considered outliers in the context of unicorn companies. The resulting subset includes the "Company",
"Year Founded", and "Country" columns for better context, and it is printed in a string format for better readability in the console output."""