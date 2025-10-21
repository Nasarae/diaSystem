import subprocess
import platform
import socket

raspberry_pis = {
    "10.124.201.251": "Nursery-A",
    "10.124.202.251": "Nursery-B",
    "10.124.203.251": "Nursery-C",
    "10.124.204.251": "Nursery-D",
    "10.124.98.12": "Nursery-E/F",
    "10.124.98.11": "Frank (1/2/3/4)",
    "10.124.98.13": "Frank2 (Tank 5)",

}

param = "-n" if platform.system().lower() == "windows" else "-c"

def get_hostname(ip):
    try:
        output = subprocess.check_output(["avahi-resolve-address", ip], timeout=1)
        return output.decode().split()[1]
    except:
        try:
            return socket.gethostbyaddr(ip)[0]
        except:
            return None

for ip, fallback_name in raspberry_pis.items():
    try:
        result = subprocess.run(["ping", param, "1", ip],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            hostname = get_hostname(ip)
            shown_name = hostname if hostname else fallback_name
            print(f"{shown_name} ({ip}): 🟢 Online")
        else:
            print(f"{fallback_name} ({ip}): 🔴 Offline")
    except Exception as e:
        print(f"{fallback_name} ({ip}): ❌ Error - {e}")