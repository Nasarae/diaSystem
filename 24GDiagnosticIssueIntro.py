# Diagnostic script with enhanced YAML-based issues/solutions integration
import requests
import base64
import datetime
from dateutil import parser
import statistics
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import yaml
import os

# Load configuration
with open("parameter_diagnostics.yaml") as f:
    config = yaml.safe_load(f)["parameters"]

# InfluxDB Config
INFLUX_URL = "http://10.0.202.244:8086"
DB_NAME = "ls"
USER = "lsuser_influxdb"
PASSWORD = "w7s9T4U8EAB4BzKaqUx9"

# Auth header
auth = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Basic {auth}"
}

# Time range
now = datetime.datetime.utcnow()
start_time = (now - datetime.timedelta(hours=24)).isoformat("T") + "Z"

# Define full tank config
tanks_and_nurseries = {
    "Tank1": {
        "Temperature": ('LSdata0101', '10.124.101.115'),
        "PH": ('LSdata0101', '10.124.101.115'),
        "DO mg/L": ('LSdata0010', '10.124.101.101'),
        "CO2 concentration": ('LSdata0022', '10.124.101.108'),
        "Salinity": ('LSdata0027', '10.124.101.106'),
        "Turbidity": ('LSdata0018', '10.124.101.103')
    },
    "Tank2": {
        "Temperature": ('LSdata0008', '10.124.102.114'),
        "PH": ('LSdata0008', '10.124.102.104'),
        "DO mg/L": ('LSdata0010', '10.124.102.101'),
        "CO2 concentration": ('LSdata0022', '10.124.102.108'),
        "Salinity": ('LSdata0027', '10.124.102.106'),
        "Turbidity": ('LSdata0018', '10.124.102.103')
    },
    "Tank3": {
        "Temperature": ('LSdata0008', '10.124.103.114'),
        "PH": ('LSdata0008', '10.124.103.104'),
        "DO mg/L": ('LSdata0010', '10.124.103.101'),
        "CO2 concentration": ('LSdata0022', '10.124.103.107'),
        "Salinity": ('LSdata0027', '10.124.103.106'),
        "Turbidity": ('LSdata0018', '10.124.103.103')
    },
    "Tank4": {
        "Temperature": ('LSdata0008', '10.124.104.114'),
        "PH": ('LSdata0008', '10.124.104.104'),
        "DO mg/L": ('LSdata0010', '10.124.104.101'),
        "CO2 concentration": ('LSdata0022', '10.124.104.107'),
        "Salinity": ('LSdata0027', '10.124.104.106'),
        "Turbidity": ('LSdata0018', '10.124.104.103')
    },

    "Tank5": {
        "Temperature": ('LSdata0008', '10.124.105.100'),
        "PH": ('LSdata0008', '10.124.105.100'),
        "DO mg/L": ('LSdata0010', '10.124.105.100'),
        "CO2 concentration": ('LSdata0022', '10.124.105.100'),
        "Salinity": ('LSdata0027', '10.124.105.100'),
        "Turbidity": ('LSdata0018', '10.124.105.100')
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

# Threshold-based classifier
def classify_value(param, value):
    thresh = config.get(param, {}).get("thresholds")
    if not thresh or value is None:
        return "Unknown"
    min_v, bl_low, opt, bl_high, max_v = thresh
    if value < min_v:
        return "isLow"
    elif min_v <= value <= bl_low:
        return "isBLLow"
    elif bl_low < value < bl_high:
        return "isRange"
    elif bl_high <= value <= max_v:
        return "isBLHigh"
    elif value > max_v:
        return "isHigh"
    return "Unknown"

# Interpretation with config-based diagnostics
def interpret_results(name, latest):
    summary = [f"\nOn {name}:"]
    remarks_found = False
    for param, val in latest.items():
        param_key = "Temperature" if param.lower() == "temp" else param
        state = classify_value(param_key, val)
        if state in ("isHigh", "isLow"):
            remarks_found = True
            info = config.get(param_key, {})
            issues = info.get("issues", [])
            solutions = info.get("solutions", {})
            solution_list = solutions.get("high" if state == "isHigh" else "low", [])
            waiting = info.get("waiting_time_h")
            state2 = "high" if state == "isHigh" else "low"
            summary.append(f"- {param_key} is {state2}.")
            if issues:
                summary.append("  Possible issues:")
                for i in issues:
                    summary.append(f"    * {i}")
            if solution_list:
                summary.append("  Possible solutions:")
                for s in solution_list:
                    summary.append(f"    * {s}")
            #if waiting:
                #summary.append(f"  Wait approximately {waiting}h for changes to be visible.\n\n")
    if not remarks_found:
        summary.append("No remarks, all parameters within bounds.\n")
    return "\n".join(summary)

# Fetch latest + 24h data
def fetch_data(ip, measurement, field):
    query = f'SELECT "{field}" FROM "autogen"."{measurement}" WHERE "sensorIp" = \'{ip}\' AND time > \'{start_time}\''
    params = {"db": DB_NAME, "q": query}
    try:
        response = requests.get(f"{INFLUX_URL}/query", params=params, headers=headers, timeout=10)
        data = response.json()
        if "results" in data and "series" in data["results"][0]:
            points = data["results"][0]["series"][0]["values"]
            timestamps = [parser.parse(p[0]) for p in points]
            values = [float(p[1]) for p in points]
            return timestamps, values
        else:
            print(f"\nNo data for {field} ({ip})")
            return [], []
    except Exception as e:
        print(f"\nError fetching {field}: {e}")
        return [], []

# Main data/plot/report function
def plot_tank_data(name, config_map):
    latest_values = {}
    full_data = {}
    averages = {}

    for label, (measurement, ip) in config_map.items():
        timestamps, values = fetch_data(ip, measurement, label)
        if values:
            full_data[label] = (timestamps, values)
            latest_values[label] = values[-1]
            averages[label] = statistics.mean(values)

    if 'CO2 concentration' in full_data:
        temp_key = 'temp' if 'temp' in full_data else 'Temperature'
        if temp_key in full_data:
            co2_times, co2_vals = full_data['CO2 concentration']
            temp_times, temp_vals = full_data[temp_key]
            min_len = min(len(co2_vals), len(temp_vals))
            full_data['CO2_water'] = (co2_times[:min_len], [
                compute_cwater(T, A) for T, A in zip(temp_vals[:min_len], co2_vals[:min_len])
            ])
            averages['CO2_water'] = statistics.mean(full_data['CO2_water'][1])
            latest_values['CO2_water'] = full_data['CO2_water'][1][-1]

    if not full_data:
        print(f"No data to plot for {name}")
        return

    # Plotting
    fig, axs = plt.subplots(len(full_data), 1, figsize=(10, 2 * len(full_data)), sharex=True)
    if len(full_data) == 1:
        axs = [axs]
    for ax, (label, (times, values)) in zip(axs, full_data.items()):
        ax.plot(times, values, label=label)
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
    plt.xlabel("Time")
    plt.suptitle(f"{name} - Last 24h")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{name}.jpg", dpi=150)
    plt.close()

    # Summary print
    print(f"\n--- Latest Diagnostics for {name} ---")
    for k, v in latest_values.items():
        print(f"{k}: {v:.3f}")
    print(interpret_results(name, latest_values))

# Run all
def main():
    for tank_name, tank_config in tanks_and_nurseries.items():
        plot_tank_data(tank_name, tank_config)

main()
