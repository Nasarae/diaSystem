import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from fpdf import FPDF

# === CONFIG ===
log_dir = "./logs"  # or the actual path to your logs
log_files = {
    "Frank": os.path.join(log_dir, "Frank.txt"),
    "Nursery_A": os.path.join(log_dir, "Nursery_A.txt"),
    "Nursery_B": os.path.join(log_dir, "Nursery_B.txt"),
    "Nursery_C": os.path.join(log_dir, "Nursery_C.txt"),
    "Nursery_D": os.path.join(log_dir, "Nursery_D.txt"),
}

gpio_to_tank = {
    5: "Tank1", 6: "Tank1",
    13: "Tank2", 16: "Tank2",
    19: "Tank3", 20: "Tank3",
    21: "Tank4", 26: "Tank4"
}

# === REGEX ===
log_pattern = re.compile(r"^(?P<identifier>[A-Z]+) (?P<date>\d{2}/\d{2}/\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) - (?P<message>.+)$")
index_pattern = re.compile(r"Processed JSON data: (?P<data>[\d,]*)")
command_pattern = re.compile(r"Dispensing (?P<grams>\d+) grams, GPIO (?P<gpio>\d+)")

# === STORAGE ===
missing_index = defaultdict(set)
feeding_anomalies = defaultdict(list)
malformed_entries = defaultdict(list)
feeding_totals = defaultdict(int)
index_values_by_source = {}

def safe_json_parse(s):
    return [int(x) for x in s.split(",") if x.strip().isdigit()]

def get_tank(gpio, source):
    return gpio_to_tank.get(gpio) if source == "Frank" else source

def parse_log_file(name, path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    index_values_by_date = defaultdict(list)
    malformed_start = None
    last_good_log = None
    seen_dates = set()

    for line in lines:
        line = line.strip()
        match = log_pattern.match(line)

        if match:
            identifier = match.group("identifier")
            date_str = match.group("date")
            msg = match.group("message")
            date = datetime.strptime(date_str, "%d/%m/%y").date()

            last_good_log = (date_str, match.group("time"))
            if malformed_start:
                malformed_entries[name].append((malformed_start, last_good_log))
                malformed_start = None

            if identifier == "INDEX":
                if date in seen_dates:
                    continue
                seen_dates.add(date)
                data_match = index_pattern.search(msg)
                if data_match:
                    data = safe_json_parse(data_match.group("data"))
                    index_values_by_date[date] = data

            elif identifier == "COMMAND":
                cmd_match = command_pattern.search(msg)
                if cmd_match:
                    grams = int(cmd_match.group("grams"))
                    gpio = int(cmd_match.group("gpio"))
                    if name == "Frank":
                        tank = gpio_to_tank.get(gpio)
                        if tank:
                            feeding_totals[tank] += grams
                    else:
                        # All GPIOs belong to the same nursery tank
                        feeding_totals[name] += grams

        else:
            if malformed_start is None and last_good_log:
                malformed_start = last_good_log

    index_values_by_source[name] = index_values_by_date

    # Overfeeding logic
    sorted_dates = sorted(index_values_by_date)
    for prev, curr in zip(sorted_dates, sorted_dates[1:]):
        prev_vals = index_values_by_date[prev]
        curr_vals = index_values_by_date[curr]
        if name == "Frank":
            prev_sum = defaultdict(int)
            curr_sum = defaultdict(int)
            for gpio, val in enumerate(prev_vals):
                tank = gpio_to_tank.get(gpio)
                if tank:
                    prev_sum[tank] += val
            for gpio, val in enumerate(curr_vals):
                tank = gpio_to_tank.get(gpio)
                if tank:
                    curr_sum[tank] += val
            for tank in prev_sum:
                if prev_sum[tank] > 0 and curr_sum[tank] > 5 * prev_sum[tank]:
                    feeding_anomalies[tank].append(curr)
        else:
            if sum(prev_vals) > 0 and sum(curr_vals) > 5 * sum(prev_vals):
                feeding_anomalies[name].append(curr)

# === MAIN EXECUTION ===
for name, path in log_files.items():
    parse_log_file(name, path)

# === Missing INDEX logic ===
for name, values_by_date in index_values_by_source.items():
    if not values_by_date:
        continue
    sorted_dates = sorted(values_by_date)
    full_range = set(sorted_dates[0] + timedelta(days=i) for i in range((sorted_dates[-1] - sorted_dates[0]).days + 1))
    seen_dates = set(values_by_date)
    missing_dates = sorted(full_range - seen_dates)

    for date in missing_dates:
        prev_dates = [d for d in sorted_dates if d < date]
        if not prev_dates:
            continue
        last_known = values_by_date[prev_dates[-1]]

        if name == "Frank":
            total_feed = sum(last_known)
            if total_feed > 0:
                missing_index["Frank"].add(date)
        else:
            if sum(last_known) > 0:
                missing_index[name].add(date)

# === DATAFRAMES ===
df_malformed = pd.DataFrame([(k, s[0], s[1], e[0], e[1]) for k, v in malformed_entries.items() for s, e in v],
                            columns=["Tank/Nursery", "Start Date", "Start Time", "End Date", "End Time"])
df_anomalies = pd.DataFrame([(k, d) for k, v in feeding_anomalies.items() for d in v],
                            columns=["Tank/Nursery", "Anomaly Date"])
df_missing = pd.DataFrame([(k, d) for k, v in missing_index.items() for d in sorted(v)],
                          columns=["Tank/Nursery", "Missing Index Date"])
df_totals = pd.DataFrame([(k, g, round(g/1000, 2)) for k, g in feeding_totals.items()],
                         columns=["Tank/Nursery", "Total Grams", "Total Kilos"])

# === PDF GENERATION ===
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Feeding Log Summary Report", ln=True, align="C")

def add_table(title, df, col_widths=None):
    pdf.set_font("Arial", "B", 12)
    pdf.ln(10)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Arial", size=10)

    if df.empty:
        pdf.cell(0, 10, "No data found.", ln=True)
        return

    if col_widths is None:
        col_widths = [pdf.w / len(df.columns)] * len(df.columns)

    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 10, str(col), border=1)
    pdf.ln()

    for _, row in df.iterrows():
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], 10, str(item), border=1)
        pdf.ln()

add_table("1. Malformed Logs", df_malformed, [40, 30, 25, 30, 25])
add_table("2. Overfeeding Events (500%)", df_anomalies, [60, 40])
add_table("3. Missing INDEX Events (if feed > 0)", df_missing, [60, 40])
add_table("4. Feeding Totals (g/kg)", df_totals, [60, 50, 40])

pdf.output("log_analysis_summary.pdf")
print("✅ PDF saved as 'log_analysis_summary.pdf'")