import os
import sys
import json
import math
import time
import yaml
import argparse
import datetime as dt
import re
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

import numpy as np

try:
    import pandas as pd
except Exception as e:
    print("[warn] pandas not available. Please: pip install pandas pyarrow scikit-learn joblib pyyaml requests", file=sys.stderr)
    pd = None

# --------------------------- CONFIG -------------------------------------------

INFLUX_URL  = os.environ.get("INFLUX_URL", "http://10.0.202.244:8086")
INFLUX_DB   = os.environ.get("INFLUX_DB", "ls")
INFLUX_USER = os.environ.get("INFLUX_USER", "lsuser_influxdb")
INFLUX_PASS = os.environ.get("INFLUX_PASS", "w7s9T4U8EAB4BzKaqUx9")

PARAM_YAML_PATH = os.environ.get("PARAM_YAML", "./parameter_diagnostics.yaml")

# Minimal default map; override via --map-yaml if you have a full one
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

def maybe_load_tank_map(path: str = "./tanks_and_nurseries_config.yaml"):
    """Optional external YAML mapping: {Tank: {parameters: {Label: [measurement, ip]}}}"""
    global TANKS_AND_NURSERIES
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        m: Dict[str, Dict[str, Tuple[str, str]]] = {}
        for tank, conf in (raw or {}).items():
            params = (conf or {}).get("parameters", {})
            entry = {}
            for label, pair in (params or {}).items():
                if isinstance(pair, list) and len(pair) >= 2:
                    entry[label] = (pair[0], pair[1])
                elif isinstance(pair, dict):
                    meas = pair.get("measurement") or pair.get("name") or pair.get("measurement_name")
                    ip   = pair.get("sensor_ip") or pair.get("ip")
                    if meas and ip:
                        entry[label] = (meas, ip)
            if entry:
                m[tank] = entry
        if m:
            print(f"[info] loaded tank map from {path} ({len(m)} tanks)")
            TANKS_AND_NURSERIES = m
    except Exception as e:
        print(f"[warn] could not parse {path}: {e}")

# --------------------------- Influx helpers ------------------------------------

def _auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/x-www-form-urlencoded"}
    if INFLUX_USER or INFLUX_PASS:
        import base64
        tok = base64.b64encode(f"{INFLUX_USER}:{INFLUX_PASS}".encode()).decode()
        h["Authorization"] = f"Basic {tok}"
    return h

def _field_candidates(label: str) -> List[str]:
    """Try small alias set so field name mismatches don't kill the query."""
    s = (label or "").strip()
    out = [s]
    sl = s.lower()
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
    seen = set(); dedup = []
    for x in out:
        if x and x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def fetch_series_influx(ip: str, measurement: str, field_label: str,
                        start_iso: str, end_iso: str) -> Tuple[List[str], List[float]]:
    """SELECT "<field>" FROM "autogen"."<measurement>" WHERE "sensorIp"='<ip>' AND time BETWEEN ..."""
    import requests
    if not (INFLUX_URL and INFLUX_DB):
        return [], []
    meas_ref = f'"autogen"."{measurement}"'
    headers = _auth_headers()
    for fld in _field_candidates(field_label):
        q = (
            f'SELECT "{fld}" FROM {meas_ref} '
            f'WHERE "sensorIp" = \'{ip}\' AND time >= \'{start_iso}\' AND time <= \'{end_iso}\''
        )
        try:
            r = requests.get(f"{INFLUX_URL}/query", params={"db": INFLUX_DB, "q": q},
                             headers=headers, timeout=15)
            j = r.json()
            series = j.get("results", [{}])[0].get("series", [])
            if not series or not series[0].get("values"):
                continue
            ts, vs = [], []
            for t_iso, v in series[0]["values"]:
                if v is None:
                    continue
                ts.append(t_iso)
                try:
                    vs.append(float(v))
                except Exception:
                    vs.append(np.nan)
            if ts:
                return ts, vs
        except Exception:
            continue
    return [], []

# --------------------------- Thresholds / features -----------------------------

def load_thresholds(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return d.get("parameters", {})
    except Exception:
        return {}

def bands_for(cfg: Dict[str, Any], key: str):
    thr = (cfg.get(key) or {}).get("thresholds")
    if thr and len(thr) == 5:
        return thr
    return None

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
    try:
        return float(v - center)
    except Exception:
        return float("nan")

def compute_cwater(T: float, A: float) -> float:
    """Same formula used in-app."""
    try:
        KH = (
            76.81167 - 3.348169 * T + 0.1198361 * T**2 -
            0.0031048148 * T**3 + 0.00003566666668 * T**4
        )
        return ((44.01 / 22.4) * A * 273 / (273 + T) * (101325 / 100100) /
                (1000 * 1000) / 44.01 * KH * KH * 44)
    except Exception:
        return float("nan")

# --------------------------- Snapshot builder ----------------------------------

def iso(dtobj: dt.datetime) -> str:
    return dtobj.replace(microsecond=0, tzinfo=dt.timezone.utc).isoformat().replace("+00:00","Z")

def to_dt(iso_ts: str) -> dt.datetime:
    """
    Parse Influx ISO timestamps with possible nanoseconds.
    Trims fractional seconds to 6 digits (microseconds) for Python 3.9.
    Preserves/assumes UTC if 'Z' or '+00:00' given.
    """
    s = iso_ts.strip()
    if not s:
        raise ValueError("empty timestamp")

    # Normalize Z to explicit UTC offset so fromisoformat can parse
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    # If there is a fractional part, trim it to 6 digits
    # Pattern: YYYY-mm-ddTHH:MM:SS(.fraction)?(+/-HH:MM)?
    m = re.match(r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<frac>\d+))?(?P<tz>[+-]\d{2}:\d{2})?$", s)
    if not m:
        # last resort: try pandas if available
        try:
            import pandas as pd
            return pd.to_datetime(iso_ts, utc=True).to_pydatetime()
        except Exception:
            pass
        raise ValueError(f"Unsupported ISO format: {iso_ts}")

    head = m.group("head")
    frac = m.group("frac") or ""
    tz   = m.group("tz") or ""

    if frac:
        frac6 = (frac + "000000")[:6]  # pad then trim to 6
        s2 = f"{head}.{frac6}{tz}"
    else:
        s2 = f"{head}{tz}"

    return dt.datetime.fromisoformat(s2)

def make_snapshots(hours_back: int,
                   tanks: Optional[List[str]],
                   thresholds: Dict[str, Any]):
    """Pull last <hours_back> hours and build hourly snapshots. Returns pandas.DataFrame."""
    if pd is None:
        raise RuntimeError("pandas required. pip install pandas pyarrow scikit-learn joblib")

    end = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(hours=hours_back)
    start_iso = iso(start); end_iso = iso(end)

    chosen = list(tanks) if tanks else list(TANKS_AND_NURSERIES.keys())
    rows = []

    for tank in chosen:
        conf = TANKS_AND_NURSERIES.get(tank) or {}
        if not conf:
            print(f"[warn] no config for tank {tank}; skipping")
            continue

        # Fetch series for all params up-front
        series: Dict[str, Tuple[List[str], List[float]]] = {}
        for label, (measurement, ip) in conf.items():
            ts, vs = fetch_series_influx(ip, measurement, label, start_iso, end_iso)
            if ts:
                series[label] = (ts, vs)

        if not series:
            print(f"[warn] no series for {tank} in last {hours_back}h")
            continue

        # Build hourly grid
        tmins = [to_dt(tslist[0]) for (tslist, _vs) in series.values() if tslist]
        tmaxs = [to_dt(tslist[-1]) for (tslist, _vs) in series.values() if tslist]
        if not tmins or not tmaxs:
            continue
        gstart = max(min(tmins), start).replace(minute=0, second=0, microsecond=0)
        gend   = min(max(tmaxs), end).replace(minute=0, second=0, microsecond=0)
        grid = []
        cur = gstart
        while cur <= gend:
            grid.append(cur)
            cur += dt.timedelta(hours=1)

        p_dt = {k: list(pd.to_datetime(v[0], utc=True).to_pydatetime()) for k, v in series.items()}
        p_v  = {k: list(v[1]) for k, v in series.items()}

        for t in grid:
            t6  = t - dt.timedelta(hours=6)
            t24 = t - dt.timedelta(hours=24)
            snap = {"tank_name": tank, "ts": t}

            # features per param
            for k in series.keys():
                dts = p_dt[k]; vals = p_v[k]
                idx24 = [i for i, tt in enumerate(dts) if t24 < tt <= t]
                idx6  = [i for i, tt in enumerate(dts) if t6  < tt <= t]
                if not idx24:
                    continue

                last_i = idx24[-1]
                last_v = vals[last_i]
                snap[f"{k}__last"] = last_v
                snap[f"{k}__dev_center"] = dev_center(thresholds, k, last_v)

                if idx6:
                    arr = np.array([vals[i] for i in idx6], dtype=float)
                    snap[f"{k}__mean_6h"] = float(np.nanmean(arr))
                    snap[f"{k}__std_6h"]  = float(np.nanstd(arr))
                    snap[f"{k}__min_6h"]  = float(np.nanmin(arr))
                    snap[f"{k}__max_6h"]  = float(np.nanmax(arr))
                    xs = np.array([dts[i].timestamp() for i in idx6], dtype=float)
                    if xs.size >= 2:
                        X = np.vstack([xs - xs.mean(), np.ones_like(xs)]).T
                        m, c = np.linalg.lstsq(X, arr, rcond=None)[0]
                        snap[f"{k}__slope_6h"] = float(m)

                # flags last 2h
                t2 = t - dt.timedelta(hours=2)
                idx2 = [i for i, tt in enumerate(dts) if t2 < tt <= t]
                if idx2:
                    red = yellow = 0
                    for i in idx2:
                        state = classify(thresholds, k, vals[i])
                        if state in ("Low","High"):
                            red = 1
                        elif state in ("BLow","BHigh"):
                            yellow = 1
                    snap[f"{k}__flag_red_2h"] = red
                    snap[f"{k}__flag_yellow_2h"] = yellow

            # derived: CO2_water_last if both present
            if "Temperature__last" in snap and "CO2 concentration__last" in snap:
                T = snap["Temperature__last"]; A = snap["CO2 concentration__last"]
                snap["CO2_water__last"] = compute_cwater(T, A)

            # Weak label y: any hard breach in next 6h?
            y = 0
            t_next = t + dt.timedelta(hours=6)
            for k in series.keys():
                thr = bands_for(thresholds, k)
                if not thr:
                    continue
                t0, _t1, _t2, _t3, t4 = thr
                dts = p_dt[k]; vals = p_v[k]
                idxF = [i for i, tt in enumerate(dts) if t < tt <= t_next]
                for i in idxF:
                    v = vals[i]
                    if v < t0 or v > t4:
                        y = 1
                        break
                if y: break
            snap["y"] = int(y)
            rows.append(snap)

    if not rows:
        print("[warn] no snapshots built; check INFLUX_* env, mapping, and hours-back.")
        return pd.DataFrame()
    return pd.DataFrame(rows)

# --------------------------- Train LR + calibrate ------------------------------

def train_lr_calibrated(df, out_model="risk_model.joblib", report=True):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.impute import SimpleImputer
    import joblib
    import numpy as np
    import pandas as pd

    # Sort by time and keep label
    df = df.sort_values("ts").reset_index(drop=True)
    df = df.dropna(subset=["y"])
    y = df["y"].astype(int)

    drop_cols = {"y", "tank_name", "ts"}
    # Drop columns that are >95% missing (pervasive NaNs)
    nan_frac = df.isna().mean()
    candidate_feats = [c for c in df.columns if c not in drop_cols and nan_frac[c] <= 0.95]

    X = df[candidate_feats].copy()

    # Optional: drop rows with too few observed features (avoid garbage snapshots)
    min_present = 5
    keep_mask = X.notna().sum(axis=1) >= min_present
    X = X.loc[keep_mask]
    y = y.loc[keep_mask]

    # Time-split: first 80% train, last 20% test
    split = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    # Imputer -> Scaler -> LR
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=400, class_weight="balanced", n_jobs=1, solver="lbfgs"))
    ])

    # Safe number of CV splits for calibration (avoid empty-class folds)
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    n_splits = max(2, min(5, n_pos, n_neg))  # at least 2, at most 5, bounded by class counts
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    model.fit(Xtr, ytr)

    # Evaluate
    probs = model.predict_proba(Xte)[:, 1] if len(Xte) else np.array([])
    if len(np.unique(yte)) > 1 and probs.size:
        roc = roc_auc_score(yte, probs)
        pr  = average_precision_score(yte, probs)
    else:
        roc = float("nan"); pr = float("nan")

    if report:
        print(f"[eval] ROC AUC={roc:.3f}  PR AUC={pr:.3f}  (test set {len(yte)} rows)")
        print(f"[eval] prevalence train={float(np.mean(ytr)):.3f}, test={float(np.mean(yte)):.3f}")

    joblib.dump({"model": model, "features": candidate_feats}, out_model)
    return {"roc_auc": roc, "pr_auc": pr, "n_test": int(len(yte)), "features": candidate_feats}


# --------------------------- Main CLI -----------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours-back", type=int, default=72, help="How many hours back to pull")
    parser.add_argument("--tanks", nargs="*", default=None, help="Tank names; default: all in map")
    parser.add_argument("--out-ds", default="snapshots.parquet", help="Output dataset file (parquet)")
    parser.add_argument("--out-model", default="risk_model.joblib", help="Output model file")
    parser.add_argument("--map-yaml", default="./tanks_and_nurseries_config.yaml", help="Optional YAML with tank mapping")
    args = parser.parse_args()

    if not INFLUX_URL or not INFLUX_DB:
        print("[error] INFLUX_URL and INFLUX_DB env vars are required.", file=sys.stderr)
        sys.exit(2)

    maybe_load_tank_map(args.map_yaml)

    thresholds = load_thresholds(PARAM_YAML_PATH)
    if not thresholds:
        print(f"[warn] no thresholds loaded from {PARAM_YAML_PATH} — features will be limited.")

    if pd is None:
        print("[error] pandas/scikit-learn required. Try: pip install pandas pyarrow scikit-learn joblib pyyaml requests", file=sys.stderr)
        sys.exit(3)

    print(f"[info] building snapshots for {args.tanks or 'ALL tanks'} over last {args.hours_back}h ...")
    df = make_snapshots(args.hours_back, args.tanks, thresholds)

    if df.empty:
        print("[error] empty dataset; aborting.")
        sys.exit(4)

    pos = int(df["y"].fillna(0).sum())
    neg = int((1 - df["y"].fillna(0)).sum())
    print(f"[info] label counts: positives={pos}, negatives={neg}")

    # save dataset
    try:
        df.to_parquet(args.out_ds)
        print(f"[ok] wrote dataset: {args.out_ds}  rows={len(df)} cols={len(df.columns)}")
    except Exception as e:
        print(f"[warn] could not write parquet ({e}); writing CSV instead")
        df.to_csv(os.path.splitext(args.out_ds)[0] + ".csv", index=False)


    print("[info] training LogisticRegression (calibrated) ...")
    rpt = train_lr_calibrated(df, out_model=args.out_model, report=True)
    print(f"[ok] saved model: {args.out_model}")
    print(f"[info] features used: {len(rpt['features'])}")

if __name__ == "__main__":
    main()