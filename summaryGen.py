import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from fpdf import FPDF
import re

# === Shared Setup ===
OUTPUT_DIR = "./output_data"
PLOT_DIR = os.path.join(OUTPUT_DIR, "observation_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

OBS_COLUMNS = [
    "Digestive Tract", "Feed Leftovers", "Dead",
    "Dead Shrimps", "Foam Formation", "Shrimp Fitness", "Shells"
]

# === PDF Class ===
class PDFWithTitles(FPDF):
    def header(self):
        if hasattr(self, "plot_title") and self.plot_title:
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, self.plot_title, ln=True, align="C")

# === Utility Functions ===
def parse_float_column(series):
    series = series.astype(str).str.extract(r'([\d.,]+)')[0]
    series = series.str.replace(",", ".", regex=False).str.replace(" ", "")
    return pd.to_numeric(series, errors="coerce")

def normalize_nursery_correction(x):
    try:
        val = float(x)
        return val * 100 if 0 < val <= 2 else val
    except:
        return None

# === Analysis Entrypoint ===
def generate_summary(cleaned_files, category):
    tank_data = {}
    anomaly_summary = []
    cooccurrence_data = []
    plot_paths = []

    for file in cleaned_files:
        df = pd.read_csv(file, dtype=str)
        filename = os.path.basename(file)
        match = re.search(r"(T\d+|N[A-Z])B\d+", filename)
        if match:
            tank_name = match.group(1)  # e.g. "T2" or "NA"
            batch_id = re.search(r"B\d+", filename).group(0)  # e.g. "B4", "B15"
        else:
            print(f"[WARN] Filename not recognized: {filename}")
            continue

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df[df["Date"].notna()]
        df = df[df["Date"] >= pd.Timestamp("2025-07-01")]

        if "Correction %" in df.columns:
            df["Correction %"] = parse_float_column(df["Correction %"])
            df["Correction %"] = df["Correction %"].apply(normalize_nursery_correction)

        if "Adjusted Feed Amount (g)" in df.columns:
            df["Adjusted Feed Amount (g)"] = parse_float_column(df["Adjusted Feed Amount (g)"])

        df["Batch"] = batch_id

        # Drop invalid
        df = df.dropna(subset=["Date", "Correction %", "Adjusted Feed Amount (g)"])
        df = df[df["Adjusted Feed Amount (g)"] > 0]
        if df.empty:
            continue

        # Co-occurrence
        obs_subset = df[OBS_COLUMNS].dropna(how='all')
        for _, row in obs_subset.iterrows():
            tags = [str(row[col]).strip() for col in OBS_COLUMNS if pd.notna(row[col]) and row[col] != ""]
            for pair in combinations(sorted(set(tags)), 2):
                cooccurrence_data.append({"Tank": tank_name, "Pair": pair})

        # Anomalies
        for _, row in df.iterrows():
            if row["Correction %"] < 30 or row["Correction %"] > 130:
                anomaly_summary.append({
                    "Tank": tank_name,
                    "Batch": batch_id,
                    "Date": row["Date"],
                    "Correction %": row["Correction %"]
                })

        tank_data.setdefault(tank_name, []).append(df)

    if not tank_data:
        print(f"[INFO] No usable {category} data.")
        return

    # === PLOTS ===
    plot_paths += plot_cooccurrence(cooccurrence_data, category)
    plot_paths += plot_corrections(tank_data, category)
    plot_paths += plot_feed_amounts(tank_data, category)
    plot_paths += plot_batch_comparisons(tank_data, category)
    plot_paths += plot_tank_comparisons(tank_data, category)
    plot_paths += plot_anomalies(anomaly_summary, category)

    compile_pdf(plot_paths, category)

# === PLOT FUNCTIONS ===
def plot_cooccurrence(data, category):
    if not data: return []
    co_df = pd.DataFrame(data)
    co_matrix = co_df.value_counts(subset=["Pair"]).reset_index(name="Count")
    co_matrix[["Tag 1", "Tag 2"]] = pd.DataFrame(co_matrix["Pair"].tolist(), index=co_matrix.index)
    co_matrix.drop(columns="Pair", inplace=True)
    co_pivot = co_matrix.pivot(index="Tag 1", columns="Tag 2", values="Count").fillna(0)
    co_pivot = co_pivot.add(co_pivot.T, fill_value=0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title(f"{category} Observation Co-occurrence Matrix")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{category}_cooccurrence.jpg")
    plt.savefig(path); plt.close()
    return [path]

def plot_corrections(tank_data, category):
    paths = []
    cols = 2
    rows = math.ceil(len(tank_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), dpi=300)
    axes = axes.flatten()

    for i, (tank, dfs) in enumerate(tank_data.items()):
        all_data = pd.concat(dfs)
        avg_corr = all_data["Correction %"].mean()
        for df in dfs:
            label = f"Batch {df['Batch'].iloc[0]}"
            axes[i].plot(df["Date"], df["Correction %"], label=label)
        axes[i].set_title(f"{tank} (Avg: {avg_corr:.1f}%)")
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{category}_corrections.jpg")
    plt.savefig(path); plt.close()
    return [path]

def plot_feed_amounts(tank_data, category):
    paths = []
    cols = 2
    rows = math.ceil(len(tank_data) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), dpi=300)
    axes = axes.flatten()

    for i, (tank, dfs) in enumerate(tank_data.items()):
        for df in dfs:
            label = f"Batch {df['Batch'].iloc[0]}"
            axes[i].plot(df["Date"], df["Adjusted Feed Amount (g)"], label=label)
        axes[i].set_title(f"{tank} Feed Amounts")
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{category}_feed_amounts.jpg")
    plt.savefig(path); plt.close()
    return [path]

def plot_batch_comparisons(tank_data, category):
    paths = []
    for tank, dfs in tank_data.items():
        fig, axes = plt.subplots(5, 2, figsize=(16, 22), dpi=300)
        axes = axes.flatten()
        i = 0
        for col in OBS_COLUMNS:
            batch_counts = {}
            for df in dfs:
                batch = df["Batch"].iloc[0]
                counts = df[col].value_counts().to_dict() if col in df.columns else {}
                for val, count in counts.items():
                    batch_counts.setdefault(val.strip(), {})[batch] = batch_counts.get(val.strip(), {}).get(batch, 0) + count
            if not batch_counts:
                continue
            df = pd.DataFrame(batch_counts).fillna(0).T
            df.plot(kind="bar", ax=axes[i], width=0.75, title=col, legend=True)
            axes[i].tick_params(axis='x', labelrotation=45)
            axes[i].set_ylabel("Count")
            axes[i].legend(fontsize=8)
            i += 1
        fig.suptitle(f"Batch Comparisons – {tank}", fontsize=16)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"{category}_{tank}_batch_comparisons.jpg")
        plt.savefig(path); plt.close()
        paths.append(path)
    return paths

def plot_tank_comparisons(tank_data, category):
    summary = {}
    for tank, dfs in tank_data.items():
        all_df = pd.concat(dfs)
        for col in OBS_COLUMNS:
            if col in all_df.columns:
                counts = all_df[col].value_counts().to_dict()
                for val, count in counts.items():
                    summary.setdefault(col, {})[tank] = summary.get(col, {}).get(tank, 0) + count

    fig, axes = plt.subplots(5, 2, figsize=(16, 20), dpi=300)
    axes = axes.flatten()
    for i, (col, tank_counts) in enumerate(summary.items()):
        df = pd.DataFrame(tank_counts, index=[col]).T
        colors = sns.color_palette("tab20", n_colors=len(df))
        for j, (tank, row) in enumerate(df.iterrows()):
            axes[i].bar(tank, row.iloc[0], color=colors[j])
        axes[i].set_title(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{category}_tank_comparisons.jpg")
    plt.savefig(path); plt.close()
    return [path]

def plot_anomalies(anomalies, category):
    if not anomalies: return []
    df = pd.DataFrame(anomalies)
    tank_counts = df["Tank"].value_counts()
    batch_counts = df.groupby(["Tank", "Batch"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    tank_counts.plot(kind="bar", ax=ax[0], title="Anomalies per Tank")
    batch_counts.plot(kind="bar", stacked=True, ax=ax[1], title="Anomalies per Batch")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{category}_anomalies.jpg")
    plt.savefig(path); plt.close()
    return [path]

def compile_pdf(plot_paths, category):
    pdf = PDFWithTitles()
    for plot in plot_paths:
        pdf.plot_title = os.path.splitext(os.path.basename(plot))[0].replace("_", " ").title()
        pdf.add_page()
        pdf.image(plot, x=10, y=25, w=pdf.w - 20)
    pdf_path = os.path.join(OUTPUT_DIR, f"{category}_summary.pdf")
    pdf.output(pdf_path)
    print(f"[✓] PDF created: {pdf_path}")
