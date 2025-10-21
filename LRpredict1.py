#!/usr/bin/env python3
"""
ml_predict_now_verbose.py — score one tank "now" and print color + rationale + suggested actions.

What it does
------------
- Pulls last N hours from InfluxDB (same query style you used)
- Rebuilds the SAME features as the training script (last, mean/std/min/max 6h, slope, flags, dev_center)
- Loads risk_model.joblib and predicts a probability
- Maps probability to a status color (green/watch/red) — configurable thresholds
- Builds a short EXPLANATION using:
    * biggest |dev_center| features (which params are far from optimal)
    * rules (threshold states) per param with optional "issues" and "solutions"
      from parameter_diagnostics.yaml

Usage
-----
export INFLUX_URL="http://<influx-host>:8086"
export INFLUX_DB="ls"
export INFLUX_USER="user"
export INFLUX_PASS="pass"

python3 ml_predict_now_verbose.py --tank Tank1 \
  --model risk_model.joblib \
  --map-yaml ./tanks_and_nurseries_config.yaml \
  --thresholds-yaml ./parameter_diagnostics.yaml \
  --hours-back 24 \
  --green 0.2 --red 0.5

Add --json to get machine-readable output instead of human text.
"""

import os, sys, json, math, argparse, base64
import datetime as dt
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import requests
import joblib
import yaml


# --------------------------- ENV ----------------------------------------------
INFLUX_URL  = os.environ.get("INFLUX_URL", "http://10.0.202.244:8086")
INFLUX_DB   = os.environ.get("INFLUX_DB", "ls")
INFLUX_USER = os.environ.get("INFLUX_USER", "lsuser_influxdb")
INFLUX_PASS = os.environ.get("INFLUX_PASS", "w7s9T4U8EAB4BzKaqUx9")

# --------------------------- Tank map -----------------------------------------
TANKS_AND_NURSERIES: Dict[str, Dict[str, Tuple[str, str]]] = {
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

def load_tank_map(path: str):
    global TANKS_AND_NURSERIES
    if path and os.path.exists(path):
        data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        m = {}
        for tank, conf in data.items():
            params = (conf or {}).get("parameters", {})
            entry: Dict[str, Tuple[str, str]] = {}
            for label, pair in (params or {}).items():
                if isinstance(pair, list) and len(pair) >= 2:
                    entry[label] = (pair[0], pair[1])
                elif isinstance(pair, dict):
                    meas = pair.get("measurement") or pair.get("name")
                    ip   = pair.get("sensor_ip") or pair.get("ip")
                    if meas and ip:
                        entry[label] = (meas, ip)
            if entry:
                m[tank] = entry
        TANKS_AND_NURSERIES = m

# --------------------------- Thresholds / rules --------------------------------
def load_thresholds(path: str) -> Dict[str, Any]:
    try:
        y = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        return y.get("parameters", {})
    except Exception:
        return {}

def bands_for(cfg: Dict[str, Any], key: str):
    thr = (cfg.get(key) or {}).get("thresholds")
    return thr if thr and len(thr) == 5 else None

def classify(cfg: Dict[str, Any], key: str, v: float) -> str:
    thr = bands_for(cfg, key)
    if not thr or v is None or (isinstance(v, float) and math.isnan(v)):
        return "Unknown"
    t0,t1,_t2,t3,t4 = thr
    if v < t0: return "Low"
    if v > t4: return "High"
    if v < t1: return "BLow"
    if v > t3: return "BHigh"
    return "Range"

def dev_center(cfg: Dict[str, Any], key: str, v: float) -> float:
    thr = bands_for(cfg, key)
    if not thr or v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    _t0, t1, opt, t3, _t4 = thr
    center = (t1 + t3)/2 if opt is None else opt
    return float(v - center)

def suggestions_for(cfg: Dict[str, Any], key: str, state: str):
    info = cfg.get(key) or {}
    sols = (info.get("solutions") or {})
    if state == "High":  # prefer specific
        return sols.get("high") or sols.get("High") or sols.get("increase") or []
    if state == "Low":
        return sols.get("low") or sols.get("Low") or sols.get("decrease") or []
    return []

def issues_for(cfg: Dict[str, Any], key: str):
    info = cfg.get(key) or {}
    return info.get("issues") or []

# --------------------------- Influx helpers -----------------------------------
def _auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/x-www-form-urlencoded"}
    if INFLUX_USER or INFLUX_PASS:
        tok = base64.b64encode(f"{INFLUX_USER}:{INFLUX_PASS}".encode()).decode()
        h["Authorization"] = f"Basic {tok}"
    return h

def _field_candidates(label: str) -> List[str]:
    s = (label or "").strip()
    out = [s]
    sl = s.lower()
    if "co2" in sl:
        out += [
            "CO2 concentration", "CO2", "CO2_ppm", "CO2 ppm",
            "co2", "co2_ppm", "co2ppm", "CO2_concentration",
            "CO2Conc", "co2Conc"
        ]
    if sl in ("temp", "temperature"):
        out += ["Temperature", "temp", "Temp"]
    if sl in ("ph", "pH".lower()):
        out += ["pH", "PH", "ph"]
    if "do" in sl or "dissolved" in sl:
        out += ["DO mg/L", "DO", "dissolved_oxygen", "do_mgL", "DO_mg_L"]
    if "co2" in sl:
        out += ["CO2 concentration", "CO2", "CO2_ppm"]
    if "sal" in sl or "salinity" in sl:
        out += ["Salinity", "salinity", "sal"]
    if "turb" in sl or "ntu" in sl:
        out += ["Turbidity", "NTU", "turbidity"]
    seen=set(); dedup=[]
    for x in out:
        if x and x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def fetch_series_influx(ip: str, measurement: str, field_label: str,
                        start_iso: str, end_iso: str) -> Tuple[List[dt.datetime], List[float]]:
    if not (INFLUX_URL and INFLUX_DB):
        return [], []
    headers = _auth_headers()
    meas_ref = f'"autogen"."{measurement}"'
    for fld in _field_candidates(field_label):
        q = (f'SELECT "{fld}" FROM {meas_ref} '
             f'WHERE "sensorIp" = \'{ip}\' AND time >= \'{start_iso}\' AND time <= \'{end_iso}\'')
        try:
            r = requests.get(f"{INFLUX_URL}/query", params={"db": INFLUX_DB, "q": q},
                             headers=headers, timeout=12)
            j = r.json()
            series = j.get("results", [{}])[0].get("series", [])
            if not series or not series[0].get("values"):
                continue
            ts_raw, vs = [], []
            for t_iso, v in series[0]["values"]:
                if v is None:
                    continue
                ts_raw.append(t_iso)
                try:
                    vs.append(float(v))
                except Exception:
                    vs.append(np.nan)
            if ts_raw:
                dts = list(pd.to_datetime(ts_raw, utc=True).to_pydatetime())  # handles ns timestamps
                return dts, vs
        except Exception:
            continue
    return [], []

# --------------------------- Feature construction -----------------------------
def iso(ts: dt.datetime) -> str:
    return ts.replace(microsecond=0, tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def compute_cwater(T: float, A: float) -> float:
    try:
        KH = (76.81167 - 3.348169*T + 0.1198361*T**2 - 0.0031048148*T**3 + 0.00003566666668*T**4)
        return ((44.01/22.4)*A*273/(273+T)*(101325/100100)/(1000*1000)/44.01*KH*KH*44)
    except Exception:
        return float("nan")

def build_features_for_now(tank: str, thresholds: Dict[str, Any], hours_back: int = 24) -> Tuple[Dict[str, float], Dict[str, float]]:
    conf = TANKS_AND_NURSERIES.get(tank) or {}
    if not conf:
        raise RuntimeError(f"No mapping for tank {tank}")

    end = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(hours=hours_back)
    start_iso, end_iso = iso(start), iso(end)

    series: Dict[str, Tuple[List[dt.datetime], List[float]]] = {}
    for label, (measurement, ip) in conf.items():
        dts, vs = fetch_series_influx(ip, measurement, label, start_iso, end_iso)
        if dts:
            series[label] = (dts, vs)

    feats: Dict[str, float] = {}
    latest_raw: Dict[str, float] = {}

    for k, (dts, vals) in series.items():
        t = end.replace(minute=0, second=0, microsecond=0)
        t6 = t - dt.timedelta(hours=6)
        t2 = t - dt.timedelta(hours=2)

        idx24 = [i for i, tt in enumerate(dts) if (t - dt.timedelta(hours=24)) < tt <= t]
        idx6  = [i for i, tt in enumerate(dts) if t6 < tt <= t]
        idx2  = [i for i, tt in enumerate(dts) if t2 < tt <= t]

        if idx24:
            last_v = vals[idx24[-1]]
            latest_raw[k] = last_v
            feats[f"{k}__last"] = float(last_v) if last_v is not None else np.nan
            feats[f"{k}__dev_center"] = dev_center(thresholds, k, feats[f"{k}__last"])

        if idx6:
            arr = np.array([vals[i] for i in idx6], dtype=float)
            feats[f"{k}__mean_6h"] = float(np.nanmean(arr))
            feats[f"{k}__std_6h"]  = float(np.nanstd(arr))
            feats[f"{k}__min_6h"]  = float(np.nanmin(arr))
            feats[f"{k}__max_6h"]  = float(np.nanmax(arr))
            xs = np.array([dts[i].timestamp() for i in idx6], dtype=float)
            if xs.size >= 2:
                X = np.vstack([xs - xs.mean(), np.ones_like(xs)]).T
                m, _c = np.linalg.lstsq(X, arr, rcond=None)[0]
                feats[f"{k}__slope_6h"] = float(m)

        if idx2:
            red = yellow = 0
            for i in idx2:
                state = classify(thresholds, k, vals[i])
                if state in ("Low","High"): red = 1
                elif state in ("BLow","BHigh"): yellow = 1
            feats[f"{k}__flag_red_2h"] = red
            feats[f"{k}__flag_yellow_2h"] = yellow

    # ensure raw CO2 is exposed in notes
    if "CO2 concentration__last" in feats:
        latest_raw["CO2 concentration"] = feats["CO2 concentration__last"]

    # compute dissolved CO2 (needs Temperature + CO2)
    if "Temperature__last" in feats and "CO2 concentration__last" in feats:
        feats["CO2_water__last"] = compute_cwater(
            feats["Temperature__last"], feats["CO2 concentration__last"]
        )
        # expose to notes even if thresholds are missing
        latest_raw["CO2_water"] = feats["CO2_water__last"]
    return feats, latest_raw

# --------------------------- Pretty text builder -------------------------------
def color_for(p: float, green: float, red: float) -> str:
    if p >= red: return "RED"
    if p >= green: return "YELLOW"
    return "GREEN"

def summarize(thresholds, latest_raw):
    always_show = {"CO2 concentration", "CO2_water"}   # <— add this
    lines = []
    for key, val in sorted(latest_raw.items()):
        state = classify(thresholds, key, val)
        if key in always_show or state in ("High", "Low", "BHigh", "BLow"):
            thr = bands_for(thresholds, key)
            t0,t1,opt,t3,t4 = thr if thr else (None, None, None, None, None)
            try:
                lines.append(f"- {key}: {float(val):.3f} ({state}). Bands: [{t0}, {t1}, {opt}, {t3}, {t4}]")
            except Exception:
                lines.append(f"- {key}: {val} ({state}). Bands: [{t0}, {t1}, {opt}, {t3}, {t4}]")
            if state in ("High","Low"):
                iss = issues_for(thresholds, key);  sols = suggestions_for(thresholds, key, state)
                if iss:  lines.append("  Possible issues: " + "; ".join(iss[:3]))
                if sols: lines.append("  Try: " + "; ".join(sols[:3]))
    if not lines:
        lines.append("All measured parameters in range or near-range.")
    return lines


# --------------------------- Main ---------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tank", required=True)
    ap.add_argument("--model", default="risk_model.joblib")
    ap.add_argument("--map-yaml", dest="map_yaml", default="./tanks_and_nurseries_config.yaml")
    ap.add_argument("--thresholds-yaml", default="./parameter_diagnostics.yaml")
    ap.add_argument("--hours-back", type=int, default=24)
    ap.add_argument("--green", type=float, default=0.2, help="lower bound for YELLOW")
    ap.add_argument("--red", type=float, default=0.5, help="lower bound for RED")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args()

    if not (INFLUX_URL and INFLUX_DB):
        print("[error] set INFLUX_URL and INFLUX_DB", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.model):
        print(f"[error] model not found: {args.model}", file=sys.stderr); sys.exit(3)

    load_tank_map(args.map_yaml)
    if not TANKS_AND_NURSERIES:
        print("[error] empty tank map; provide --map-yaml", file=sys.stderr); sys.exit(4)
    thresholds = load_thresholds(args.thresholds_yaml)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feat_names = bundle["features"]

    feats, latest_raw = build_features_for_now(args.tank, thresholds, hours_back=args.hours_back)

    # Align to model features order
    x = pd.DataFrame([[feats.get(k, np.nan) for k in feat_names]], columns=feat_names)
    proba = float(model.predict_proba(x)[0, 1])
    color = color_for(proba, args.green, args.red)

    # Top drivers by absolute dev_center
    drivers = sorted(
        [(k[:-12], abs(v)) for k, v in feats.items() if k.endswith("__dev_center") and not (isinstance(v, float) and math.isnan(v))],
        key=lambda t: t[1], reverse=True
    )[:5]

    # Human text
    expl = summarize(thresholds, latest_raw)

    if args.json:
        print(json.dumps({
            "tank": args.tank,
            "probability": proba,
            "status": color,
            "drivers": drivers,
            "latest": latest_raw,
            "explanation": expl
        }, indent=2))
    else:
        print(f"\n== {args.tank} ==")
        print(f"Risk probability (next 6h breach): {proba:.3f}  ->  {color}")
        if drivers:
            print("Top drivers (|deviation from optimal|): " +
                  ", ".join([f"{k}: {v:.2f}" for k, v in drivers]))
        print("\nNotes:")
        for line in expl:
            print(line)
        print("")

if __name__ == "__main__":
    main()
