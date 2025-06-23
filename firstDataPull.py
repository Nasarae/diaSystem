import base64
import requests

# InfluxDB Config (edit these)
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

# List of InfluxQL queries to run
queries = [
    'SELECT * FROM "autogen"."LSdata0008" WHERE ("sensorIp" = \'10.124.101.104\') ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0008" WHERE ("sensorIp" = \'10.124.101.105\') ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0008" WHERE ("sensorIp" = \'10.124.101.104\') AND "PH" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0008" WHERE ("sensorIp" = \'10.124.101.105\') AND "PH" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0010" WHERE ("sensorIp" = \'10.124.101.101\') AND "DO mg/L" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0010" WHERE ("sensorIp" = \'10.124.101.102\') AND "DO mg/L" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0022" WHERE ("sensorIp" = \'10.124.101.107\') AND "CO2 concentration" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0022" WHERE ("sensorIp" = \'10.124.101.108\') AND "CO2 concentration" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0022" WHERE ("sensorIp" = \'10.124.101.109\') AND "CO2 concentration" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0027" WHERE ("sensorIp" = \'10.124.101.106\') AND "Salinity" != 0 ORDER BY time DESC LIMIT 2',
    'SELECT * FROM "autogen"."LSdata0018" WHERE ("sensorIp" = \'10.124.101.103\') AND "Turbidity" != 0 ORDER BY time DESC LIMIT 2'
]

# Execute all queries
for i, q in enumerate(queries, 1):
    params = {
        "db": DB_NAME,
        "q": q
    }
    print(f"\n--- Query {i} ---")
    try:
        response = requests.get(f"{INFLUX_URL}/query", params=params, headers=headers, timeout=5)
        data = response.json()
        if "results" in data:
            print(data["results"])
        else:
            print("No results or error in response format.")
    except Exception as e:
        print(f"Error: {e}")
