import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import math

# Configurable paths and columns
DATA_FOLDER = "."
FEED_COLUMNS = ["Correction %", "Adjusted Feed Amount (g)", "Date"]
OBS_COLUMNS = [
    "Digestive Tract", "Feed Leftovers", "Dead",
    "Dead Shrimps", "Foam Formation", "Shrimp Fitness", "Shells"
]

# Helper to clean numeric columns

def parse_float_column(series):
    cleaned = series.astype(str)
    cleaned = cleaned.str.extract(r'([\d.,]+)')[0]
    cleaned = cleaned.str.replace('"', '', regex=False)
    cleaned = cleaned.str.replace(',', '', regex=False)
    cleaned = cleaned.str.replace(' ', '', regex=False)
    return pd.to_numeric(cleaned, errors='coerce')

all_files = glob.glob(os.path.join(DATA_FOLDER, "masterDataT*_B*.csv"))
tank_data = {}
observation_summary = []

for file in all_files:
    df = pd.read_csv(file, dtype=str)
    tank_name = os.path.basename(file).split(".csv")[0].split("_")[0]
    batch_id = os.path.basename(file).split("_B")[-1].replace(".csv", "")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        df = df[df["Date"] >= pd.Timestamp("2024-01-01")]

    if "Correction %" in df.columns:
        df["Correction %"] = parse_float_column(df["Correction %"])

    if "Adjusted Feed Amount (g)" in df.columns:
        df["Adjusted Feed Amount (g)"] = parse_float_column(df["Adjusted Feed Amount (g)"].str.replace('%', '', regex=False))

    df["Batch"] = batch_id

    for col in OBS_COLUMNS:
        if col in df.columns:
            counts = df[col].value_counts()
            for val, count in counts.items():
                observation_summary.append({
                    "Tank": tank_name,
                    "Batch": batch_id,
                    "Parameter": col,
                    "Observation": val,
                    "Count": count
                })

    df = df.dropna(subset=["Date", "Correction %", "Adjusted Feed Amount (g)"])
    df = df[df["Adjusted Feed Amount (g)"] > 0]

    if df.empty:
        continue

    if tank_name not in tank_data:
        tank_data[tank_name] = []
    tank_data[tank_name].append(df)

# Combine all tank correction plots into one image like tank comparison
fig_count = len(tank_data)
cols = 2
rows = math.ceil(fig_count / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5), dpi=300)
axes = axes.flatten()

for i, (tank_name, dfs) in enumerate(tank_data.items()):
    all_data = pd.concat(dfs)
    avg_corr = all_data["Correction %"].mean()
    for df in dfs:
        if df.empty:
            continue
        label = f"{tank_name} - Batch {df['Batch'].iloc[0]}"
        axes[i].plot(df["Date"], df["Correction %"], label=label)
    axes[i].set_title(f"{tank_name} (Avg: {avg_corr:.1f}%)")
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Correction %")
    axes[i].legend()
    axes[i].grid(True)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("observation_plots/all_tanks_correction_summary.jpg")
plt.close()

# Feed amount per tank
for tank_name, dfs in tank_data.items():
    plt.figure(figsize=(12, 6), dpi=300)
    for df in dfs:
        if df.empty:
            continue
        label = f"Batch {df['Batch'].iloc[0]}"
        plt.plot(df["Date"], df["Adjusted Feed Amount (g)"].astype(float), label=label)
    plt.title(f"{tank_name} - Adjusted Feed Amount Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Feed Amount (g)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{tank_name}_feed_amount.jpg")
    plt.close()

# Create observation summary CSV
summary_df = pd.DataFrame(observation_summary)
os.makedirs("observation_plots", exist_ok=True)
summary_df.to_csv("observation_summary.csv", index=False)

# Global batch comparisons grouped into single image
fig_count = len(OBS_COLUMNS)
cols = 2
rows = math.ceil(fig_count / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
axes = axes.flatten()
for idx, param in enumerate(OBS_COLUMNS):
    subset = summary_df[summary_df["Parameter"] == param]
    if not subset.empty:
        pivot = subset.pivot_table(index="Observation", columns="Batch", values="Count", aggfunc="sum", fill_value=0)
        pivot.plot(kind="bar", ax=axes[idx], colormap="tab10")
        axes[idx].set_title(f"{param}")
        axes[idx].set_ylabel("Count")
        axes[idx].set_xlabel("Observation")
        axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("observation_plots/global_batch_comparisons.jpg")
plt.close()

# Per-tank batch comparisons
for tank in summary_df['Tank'].unique():
    tank_subset = summary_df[summary_df['Tank'] == tank]
    fig_count = len(OBS_COLUMNS)
    cols = 2
    rows = math.ceil(fig_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    axes = axes.flatten()
    for idx, param in enumerate(OBS_COLUMNS):
        subset = tank_subset[tank_subset["Parameter"] == param]
        if subset.empty:
            continue
        pivot = subset.pivot_table(index="Observation", columns="Batch", values="Count", aggfunc="sum", fill_value=0)
        pivot.plot(kind="bar", ax=axes[idx], colormap="tab10")
        axes[idx].set_title(f"{param}")
        axes[idx].set_ylabel("Count")
        axes[idx].set_xlabel("Observation")
        axes[idx].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(f"observation_plots/{tank}_batch_comparison_summary.jpg")
    plt.close()

# Tank comparison summary (all params in one image)
tank_agg = summary_df.groupby(["Tank", "Parameter", "Observation"])["Count"].sum().reset_index()
fig_count = len(OBS_COLUMNS)
cols = 2
rows = math.ceil(fig_count / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
axes = axes.flatten()
for idx, param in enumerate(OBS_COLUMNS):
    subset = tank_agg[tank_agg["Parameter"] == param]
    if not subset.empty:
        pivot = subset.pivot_table(index="Observation", columns="Tank", values="Count", aggfunc="sum", fill_value=0)
        pivot.plot(kind="bar", ax=axes[idx], colormap="Set2")
        axes[idx].set_title(f"{param}")
        axes[idx].set_ylabel("Count")
        axes[idx].set_xlabel("Observation")
        axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("observation_plots/tank_comparison_summary.jpg")
plt.close()
