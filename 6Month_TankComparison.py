import requests
import base64
import datetime
from dateutil import parser
import statistics
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import yaml
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Load configuration
with open("parameter_diagnostics.yaml") as f:
    config = yaml.safe_load(f)["parameters"]

# InfluxDB Config
INFLUX_URL = "http://10.0.202.244:8086"
DB_NAME = "ls"
USER = "lsuser_influxdb"
PASSWORD = "w7s9T4U8EAB4BzKaqUx9"

auth = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Basic {auth}"
}

now = datetime.datetime.utcnow()
start_time = (now - datetime.timedelta(days=180)).isoformat("T") + "Z"

# Define full tank config
tanks_and_nurseries = {
    "Tank1": {
        "temp": ('LSdata0008', '10.124.101.104'),
        "PH": ('LSdata0008', '10.124.101.104'),
        "DO mg/L": ('LSdata0010', '10.124.101.101'),
        "CO2 concentration": ('LSdata0022', '10.124.101.108'),
        "Salinity": ('LSdata0027', '10.124.101.106'),
        "Turbidity": ('LSdata0018', '10.124.101.103')
    },
    "Tank2": {
        "temp": ('LSdata0008', '10.124.102.104'),
        "PH": ('LSdata0008', '10.124.102.104'),
        "DO mg/L": ('LSdata0010', '10.124.102.101'),
        "CO2 concentration": ('LSdata0022', '10.124.102.108'),
        "Salinity": ('LSdata0027', '10.124.102.106'),
        "Turbidity": ('LSdata0018', '10.124.102.103')
    },
    "Tank3": {
        "temp": ('LSdata0008', '10.124.103.104'),
        "PH": ('LSdata0008', '10.124.103.104'),
        "DO mg/L": ('LSdata0010', '10.124.103.101'),
        "CO2 concentration": ('LSdata0022', '10.124.103.107'),
        "Salinity": ('LSdata0027', '10.124.103.106'),
        "Turbidity": ('LSdata0018', '10.124.103.103')
    },
    "Tank4": {
        "temp": ('LSdata0008', '10.124.104.104'),
        "PH": ('LSdata0008', '10.124.104.104'),
        "DO mg/L": ('LSdata0010', '10.124.104.101'),
        "CO2 concentration": ('LSdata0022', '10.124.104.107'),
        "Salinity": ('LSdata0027', '10.124.104.106'),
        "Turbidity": ('LSdata0018', '10.124.104.103')
    },
    # Repeat for Tank2, Tank3, Tank4, and NurseryA, B, C, D
    # Example for NurseryA
    "NurseryA": {
        "Temperature": ('LSdata0026', '10.124.201.104'),
        "PH": ('LSdata0026', '10.124.201.104'),
        "DO mg/L": ('LSdata0010', '10.124.201.103'),
        "CO2 concentration": ('LSdata0022', '10.124.201.106'),
        "Salinity": ('LSdata0027', '10.124.201.105'),
        # "Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        # "Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    },
    "NurseryB": {
        "Temperature": ('LSdata0026', '10.124.202.104'),
        "PH": ('LSdata0026', '10.124.202.104'),
        "DO mg/L": ('LSdata0010', '10.124.202.103'),
        "CO2 concentration": ('LSdata0022', '10.124.202.107'),
        "Salinity": ('LSdata0027', '10.124.202.105'),
        # "Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        # "Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    },
    "NurseryC": {
        "Temperature": ('LSdata0026', '10.124.203.104'),
        "PH": ('LSdata0026', '10.124.203.104'),
        "DO mg/L": ('LSdata0010', '10.124.203.103'),
        "CO2 concentration": ('LSdata0022', '10.124.203.106'),
        "Salinity": ('LSdata0027', '10.124.203.105'),
        # "Flow rate (m3/h)": ('LSdata0007', '10.124.201.107'),
        # "Flow velocity m/s": ('LSdata0007', '10.124.201.107')
    },
    "NurseryD": {
        "Temperature": ('LSdata0026', '10.124.204.104'),
        "PH": ('LSdata0026', '10.124.204.104'),
        "DO mg/L": ('LSdata0010', '10.124.204.103'),
        "CO2 concentration": ('LSdata0022', '10.124.204.106'),
        "Salinity": ('LSdata0027', '10.124.204.105'),
        # "Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        # "Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    },
    "NurseryE": {
        "Temperature": ('LSdata0026', '10.124.205.101'),
        "PH": ('LSdata0026', '10.124.205.101'),
        "DO mg/L": ('LSdata0010', '10.124.205.101'),
        "CO2 concentration": ('LSdata0022', '10.124.205.101'),
        "Salinity": ('LSdata0027', '10.124.205.101'),
        # "Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        # "Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    },
    "NurseryF": {
        "Temperature": ('LSdata0026', '10.124.206.101'),
        "PH": ('LSdata0026', '10.124.206.101'),
        "DO mg/L": ('LSdata0010', '10.124.206.101'),
        "CO2 concentration": ('LSdata0022', '10.124.206.101'),
        "Salinity": ('LSdata0027', '10.124.206.101'),
        # "Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        # "Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    }
    # Add other nurseries similarly
}

# CO2 in water calculator
def compute_cwater(T, A):
    KH = (
        76.81167 - 3.348169 * T + 0.1198361 * T**2 -
        0.0031048148 * T**3 + 0.00003566666668 * T**4
    )
    return ((44.01 / 22.4) * A * 273 / (273 + T) * (101325 / 100100) /
            (1000 * 1000) / 44.01 * KH * KH * 44)

# Fetch long-term data
def fetch_data(ip, measurement, field):
    query = f"SELECT \"{field}\" FROM \"autogen\".\"{measurement}\" WHERE \"sensorIp\" = '{ip}' AND time > '{start_time}'"
    params = {"db": DB_NAME, "q": query}
    try:
        response = requests.get(f"{INFLUX_URL}/query", params=params, headers=headers, timeout=10)
        data = response.json()
        if "results" in data and "series" in data["results"][0]:
            points = data["results"][0]["series"][0]["values"]
            timestamps = [parser.parse(p[0]) for p in points]
            values = [float(p[1]) if p[1] is not None else 0 for p in points]
            return timestamps, values
        else:
            return [], []
    except Exception as e:
        print(f"Error fetching {field}: {e}")
        return [], []

# Analyze parameter stats
def analyze_param(values, param_name=None):
    values_clean = [v for v in values if v != 0 and v is not None and not np.isnan(v)]
    if param_name == "Turbidity":
        values_clean = [v for v in values_clean if v <= 500]
    if not values_clean:
        return None
    avg = np.mean(values_clean)
    mode = float(pd.Series(values_clean).mode().values[0])
    stddev = np.std(values_clean)
    variation = stddev / avg if avg != 0 else 0
    return {"average": avg, "mode": mode, "stddev": stddev, "variation": variation}

# Combine plots per tank
def plot_all_params_per_tank(tank, param_data):
    fig, axs = plt.subplots(len(param_data), 1, figsize=(12, 2.5 * len(param_data)), sharex=True)
    if len(param_data) == 1:
        axs = [axs]
    for ax, (param, (times, values)) in zip(axs, param_data.items()):
        if param == "Turbidity":
            filtered_times = [t for t, v in zip(times, values) if v <= 400]
            filtered_values = [v for v in values if v <= 400]
            ax.plot(filtered_times, filtered_values, label=param)
        else:
            ax.plot(times, values, label=param)
        ax.set_ylabel(param)
        ax.grid(True)
        ax.legend()
    plt.xlabel("Time")
    plt.suptitle(f"{tank} - 6 Month Overview")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{tank}_6mo_overview.jpg", dpi=150)
    plt.close()

# Run all
results = defaultdict(dict)

param_aliases = {
    "temp": "Temperature",
    #"CO2_water": "CO2 concentration"
}

tank_results = defaultdict(dict)
nursery_results = defaultdict(dict)

for tank, sensors in tanks_and_nurseries.items():
    param_data = {}
    param_stats = {}
    print(f"\n--- Processing {tank} ---")
    for param, (measurement, ip) in sensors.items():
        times, values = fetch_data(ip, measurement, param)
        if times and values:
            param_data[param] = (times, values)
            stats = analyze_param(values, param)
            if stats:
                param_stats[param] = stats
                group = nursery_results if tank.lower().startswith("nursery") else tank_results
                mapped_param = param_aliases.get(param, param)
                group[mapped_param][tank] = stats
    if 'CO2 concentration' in param_data and ('temp' in param_data or 'Temperature' in param_data):
        co2_times, co2_vals = param_data['CO2 concentration']
        temp_key = 'temp' if 'temp' in param_data else 'Temperature'
        temp_times, temp_vals = param_data[temp_key]
        min_len = min(len(co2_vals), len(temp_vals))
        co2_water_vals = [compute_cwater(T, A) for T, A in zip(temp_vals[:min_len], co2_vals[:min_len])]
        param_data['CO2_water'] = (co2_times[:min_len], co2_water_vals)
        stats = analyze_param(co2_water_vals, 'CO2_water')
        if stats:
            group = nursery_results if tank.lower().startswith("nursery") else tank_results
            mapped_param = param_aliases.get('CO2_water', 'CO2_water')
            group[mapped_param][tank] = stats
    if param_data:
        plot_all_params_per_tank(tank, param_data)

# Save summary tables to CSV
os.makedirs("summary_tables", exist_ok=True)

def process_summary_data(summary_dict, label):
    all_rows = []

    for param, tank_stats in summary_dict.items():
        thresholds = config.get(param, {}).get("thresholds")
        optimal = None
        if thresholds and len(thresholds) == 5:
            optimal = thresholds[2]  # middle optimal value

        df_rows = []
        for tank, stats in tank_stats.items():
            avg = stats['average']
            mode = stats['mode']
            stddev = stats['stddev']
            var = stats['variation']

            if thresholds and (avg < thresholds[0] or avg > thresholds[4]):
                status = "Average out of bounds"
            else:
                status = ""
            df_rows.append((tank, avg, mode, stddev, var, status, param))

        if optimal is not None and df_rows:
            df_temp = pd.DataFrame(df_rows, columns=["Tank", "Average", "Mode", "Stddev", "Variance", "Status", "Parameter"])
            df_temp["AvgDiff"] = (df_temp["Average"] - optimal).abs()
            df_temp = df_temp.sort_values(by=["Stddev", "AvgDiff"], ascending=[True, True]).reset_index(drop=True)

            if not df_temp.empty:
                # Best
                for idx in df_temp.index:
                    if df_temp.at[idx, "Status"] == "":
                        df_temp.at[idx, "Status"] = "Best"
                        break

                # Worst within bounds
                in_range = df_temp[(df_temp["Average"] >= thresholds[0]) & (df_temp["Average"] <= thresholds[4])]
                if not in_range.empty:
                    worst_idx = in_range["AvgDiff"].idxmax()
                    if df_temp.at[worst_idx, "Status"] == "":
                        df_temp.at[worst_idx, "Status"] = "Worst"

            df_rows = df_temp.drop(columns=["AvgDiff"]).values.tolist()

        all_rows.append(["" for _ in range(7)])  # Empty row
        all_rows.append(["===== {} =====".format(param)] + ["" for _ in range(6)])  # Header row
        all_rows.extend(df_rows)

    df_all = pd.DataFrame(all_rows, columns=["Tank", "Average", "Mode", "Stddev", "Variance", "Status", "Parameter"])
    df_all.to_csv(f"summary_tables/{label}_parameter_summary.csv", index=False)

# Save CSVs
process_summary_data(tank_results, "tanks")
process_summary_data(nursery_results, "nurseries")