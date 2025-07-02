# Final refactored script with CSV extraction, cleaning, and summary generation

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from itertools import combinations
from fpdf import FPDF
import re

# === CONFIGURATION ===
INPUT_DIR = "./input_excels"
CSV_DIR = "./extracted_csvs"
OUTPUT_DIR = "./output_data"
PLOT_DIR = os.path.join(OUTPUT_DIR, "observation_plots")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

TRANSLATION_MAP = {
    "Datum": "Date",
    "Verdaungstrakt": "Digestive Tract",
    "Futterreste": "Feed Leftovers",
    "Tote": "Dead",
    "Tote Shrimps": "Dead Shrimps",
    "Schaumbildung": "Foam Formation",
    "Shrimp Fitness": "Shrimp Fitness",
    "Schalen": "Shells",
    "Korrektur": "Correction %",
    "Futtergabe (Ideal) korrigiert (g)": "Adjusted Feed Amount (g)",
    "Zucker Menge": "Sugar Amount",
    "Zucker ja/nein": "Sugar yes/no",
    "Puffer Menge": "Buffer Amount",
    "Puffer ja/nein": "Buffer yes/no",
    "Vitamin C Menge": "Vitamin C Amount",
    "Bactosafe Menge": "Bactosafe Amount",
    "Bemerkung": "Comment"
}

VALUE_TRANSLATION_MAP = {
    "Leicht Gefüllt": "Lightly filled",
    "Keine Futterreste": "No leftover feed",
    "Wenig Tote": "Few dead",
    "sehr viele Schalen": "Many shells",
    "wenige Schalen": "Few shells",
    "80% Shrimps haben vollen Verdaungstrakt": "80% of shrimps have full digestive tracts",
    "Starke Schaumbildung": "Heavy foaming",
    "Wenig starke Schaumbildung": "Little foaming",
    "Normale Schaumbildung": "Normal foaming",
    "Shrimps merheitlich Transluzent": "Shrimps are mostly translucent",
    "viele Schalen": "Many shells",
    "Tote sind merheitlich Orange": "Dead ones are mostly orange",
    "Keine Tote": "No dead ones",
    "50% Shrimps haben vollen Verdaungstrakt": "50% of shrimp have full digestive tracts",
    "Tote sind merheitlich Grau": "Dead ones are mostly gray",
    "Tote, Orang und Grau gemischt": "Dead ones, orange and gray mixed",
    "Viele Futterreste!!!": "A lot of leftover food!",
    "Wenig Futterreste": "Little leftover feed",
    "Shrimps gestresst/ weisslich, Darm kaum ernkennbar": "Shrimps stressed/whitish, intestine barely visible",
    "Shrimps haben merheitlich weiche Panzer": "Shrimps have mostly soft shells",
    "Tote haben harte Schale": "Dead ones have hard shell",
    "Tote haben weiche Schale": "Dead ones have soft shell",
    "Viele Tote": "Many dead",
    "keine Schalen": "No shells"
}

OBS_COLUMNS = [
    "Digestive Tract", "Feed Leftovers", "Dead",
    "Dead Shrimps", "Foam Formation", "Shrimp Fitness", "Shells"
]

ALL_COLUMNS = list(TRANSLATION_MAP.values())

# === PDF With Title Header ===
class PDFWithTitles(FPDF):
    def header(self):
        if hasattr(self, "plot_title") and self.plot_title:
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, self.plot_title, ln=True, align="C")

# === UTILITY ===
def parse_float_column(series):
    cleaned = series.astype(str)
    cleaned = cleaned.str.extract(r'([\d.,]+)')[0]
    cleaned = cleaned.str.replace('"', '', regex=False)
    cleaned = cleaned.str.replace(',', '', regex=False)
    cleaned = cleaned.str.replace(' ', '', regex=False)
    return pd.to_numeric(cleaned, errors='coerce')

# === STEP 1: EXTRACT CSV FROM EXCEL ===
def extract_csv_from_excel(filepath, csv_output):
    try:
        df_header = pd.read_excel(filepath, sheet_name="Fütterung Kona", header=None, nrows=12)
        df = pd.read_excel(filepath, sheet_name="Fütterung Kona", header=None, skiprows=12)
        df.columns = df_header.iloc[11].fillna("").astype(str).str.strip()
        df.to_csv(csv_output, index=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract sheet from {filepath}: {e}")
        return False

# === STEP 2: CLEAN CSV DATA ===
def clean_csv(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv, encoding='ISO-8859-1')

        translated_cols = {}
        for col in df.columns:
            for key, val in TRANSLATION_MAP.items():
                if key in col:
                    translated_cols[col] = val
                    break

        df = df[list(translated_cols.keys())]
        df.rename(columns=translated_cols, inplace=True)

        # Date filtering and formatting
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df[df["Date"] >= pd.Timestamp("2024-01-01")]
            df = df[df["Date"].notna()]
            df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

        for col in df.columns:
            if col in df.columns and isinstance(df[col], pd.Series):
                if df[col].dtype == object:
                    df[col] = df[col].replace(VALUE_TRANSLATION_MAP)

        obs_cols = [col for col in OBS_COLUMNS if col in df.columns]
        if obs_cols:
            last_valid_idx = df[obs_cols].notna().any(axis=1)[::-1].idxmax()
            df = df.loc[:last_valid_idx]

        df.to_csv(output_csv, index=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to clean {input_csv}: {e}")
        return False

# === MAIN EXECUTION ===
if __name__ == "__main__":
    files = glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))
    for file in files:
        filename = os.path.basename(file).replace(".xlsx", "")
        csv_file = os.path.join(CSV_DIR, f"{filename}.csv")
        cleaned_csv = os.path.join(OUTPUT_DIR, f"masterData_{filename}.csv")

        if extract_csv_from_excel(file, csv_file):
            clean_csv(csv_file, cleaned_csv)

    # Summary generation will be added after validating cleaning results