import requests
import base64
import datetime
from dateutil import parser
import statistics
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# InfluxDB Config
INFLUX_URL = "http://10.0.202.244:8086"
DB_NAME = "ls"
USER = "lsuser_influxdb"
PASSWORD = "w7s9T4U8EAB4BzKaqUx9"

# Basic auth header
auth = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Basic {auth}"
}

# Time range: last 24 hours
now = datetime.datetime.utcnow()
start_time = (now - datetime.timedelta(hours=24)).isoformat("T") + "Z"

# Tank and nursery data configurations
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
        #"Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        #"Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    },
    "NurseryB": {
        "Temperature": ('LSdata0026', '10.124.202.104'),
        "PH": ('LSdata0026', '10.124.202.104'),
        "DO mg/L": ('LSdata0010', '10.124.202.103'),
        "CO2 concentration": ('LSdata0022', '10.124.202.107'),
        "Salinity": ('LSdata0027', '10.124.202.105'),
        #"Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        #"Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    },
    "NurseryC": {
        "Temperature": ('LSdata0026', '10.124.203.104'),
        "PH": ('LSdata0026', '10.124.203.104'),
        "DO mg/L": ('LSdata0010', '10.124.203.103'),
        "CO2 concentration": ('LSdata0022', '10.124.203.106'),
        "Salinity": ('LSdata0027', '10.124.203.105'),
        #"Flow rate (m3/h)": ('LSdata0007', '10.124.201.107'),
        #"Flow velocity m/s": ('LSdata0007', '10.124.201.107')
    },
    "NurseryD": {
        "Temperature": ('LSdata0026', '10.124.204.104'),
        "PH": ('LSdata0026', '10.124.204.104'),
        "DO mg/L": ('LSdata0010', '10.124.204.103'),
        "CO2 concentration": ('LSdata0022', '10.124.204.106'),
        "Salinity": ('LSdata0027', '10.124.204.105'),
        #"Flow rate (m3/h)": ('LSdata0007', '10.124.201.108'),
        #"Flow velocity m/s": ('LSdata0007', '10.124.201.108')
    }
    # Add other nurseries similarly
}

# CO2 in water calculator
def compute_cwater(T, A):
    KH = (
        76.81167
        - 3.348169 * T
        + 0.1198361 * T**2
        - 0.0031048148 * T**3
        + 0.00003566666668 * T**4
    )
    Cwater = (
        (44.01 / 22.4)
        * A
        * 273 / (273 + T)
        * (101325 / 100100)
        / (1000 * 1000)
        / 44.01
        * KH * KH
        * 44
    )
    return Cwater

def fetch_data(ip, measurement, field):
    query = f'''
        SELECT "{field}" FROM "autogen"."{measurement}"
        WHERE "sensorIp" = '{ip}' AND time > '{start_time}' ORDER BY time ASC
    '''
    params = {"db": DB_NAME, "q": query}
    try:
        response = requests.get(f"{INFLUX_URL}/query", params=params, headers=headers, timeout=10)
        data = response.json()
        if "results" in data and "series" in data["results"][0]:
            points = data["results"][0]["series"][0]["values"]
            timestamps = [parser.parse(point[0]) for point in points]
            values = [float(point[1]) for point in points]
            return timestamps, values
        else:
            print(f"\n\nNo data for {field} ({ip})")
            return [], []
    except Exception as e:
        print(f"Error fetching {field}: {e}")
        return [], []
def plot_tank_data(name, config):
    tank_data = {}
    averages = {}

    for label, (measurement, ip) in config.items():
        timestamps, values = fetch_data(ip, measurement, label)
        if values:
            tank_data[label] = (timestamps, values)
            averages[label] = statistics.mean(values)

    if 'CO2 concentration' in tank_data and ('temp' in tank_data or 'Temperature' in tank_data):
        temp_key = 'temp' if 'temp' in tank_data else 'Temperature'
        co2_times, co2_vals = tank_data['CO2 concentration']
        temp_times, temp_vals = tank_data[temp_key]

        # Align timestamps using interpolation
        temp_series = dict(zip(temp_times, temp_vals))
        matched_vals = []
        for t, co2 in zip(co2_times, co2_vals):
            # Find nearest timestamp in temperature data
            closest_temp_time = min(temp_series, key=lambda x: abs(x - t))
            temp = temp_series[closest_temp_time]
            matched_vals.append(compute_cwater(temp, co2))

        tank_data['CO2_water'] = (co2_times[:len(matched_vals)], matched_vals)
        averages["CO2_water"] = statistics.mean(matched_vals) if matched_vals else None
    else:
        print(f"Data missing for {name}, skipping CO2_water calculation.")

    if not tank_data:
        print(f"No data to plot for {name}")
        return

    # Plotting data
    fig, axs = plt.subplots(len(tank_data), 1, figsize=(10, 2 * len(tank_data)), sharex=True)
    if len(tank_data) == 1:
        axs = [axs]

    for ax, (label, (times, values)) in zip(axs, tank_data.items()):
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

    # Print average values
    print(f"\n--- Average Values for {name} ---")
    for label, avg in averages.items():
        print(f"{label}: {avg:.4f}" if avg is not None else f"{label}: No data")


# Plot for main tanks
for tank_name, config in tanks_and_nurseries.items():
    plot_tank_data(tank_name, config)