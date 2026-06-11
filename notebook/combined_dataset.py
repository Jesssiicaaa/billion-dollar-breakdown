import pandas as pd
"""
Run:  python3 notebook/combined_dataset.py
"""
# Load all your sources
main     = pd.read_csv("tableau_exports/unicorns_main.csv")
clusters = pd.read_csv("tableau_exports/cluster_assignments.csv")
forecast = pd.read_csv("tableau_exports/forecast.csv")
models   = pd.read_csv("ML_outputs/model_results.csv")


# Merge on both Company AND Country to avoid the Bolt collision
main = main.merge(
    clusters[["Company", "Country", "Cluster ID", "Cluster Archetype"]],
    on=["Company", "Country"],
    how="left"
)

# Verify
print(f"Original rows:  1073 or 1074")
print(f"After merge:    {len(main)}")
print(f"Valuation sum:  ${main['Valuation (B USD)'].sum():.0f}B")
print(f"Bytedance rows: {main[main['Company']=='Bytedance'].shape[0]}")
print(f"Bolt rows:      {main[main['Company']=='Bolt'].shape[0]}")
print(f"Nulls in Archetype: {main['Cluster Archetype'].isna().sum()}")

main.to_csv("tableau_exports/unicorns_master.csv", index=False)
print("\nSaved → tableau_exports/unicorns_master.csv")