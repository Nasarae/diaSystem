#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RFCtrain1.py — Train a calibrated RandomForestClassifier using:
  - sensor features (last/mean/std/min/max/slope/flags, with turbidity handling, CO2_water)
  - daily observations from ./output_data/*.csv (tokens + combo rules)
  - correction_per_combination.csv (Kombination -> % correction)
  - actions_to_observations.csv (Situation -> Bemerkungen)

Label: "red breach in next --label-hours-ahead hours" (any param High/Low).

Example:
  python RFCtrain1.py \
    --map-yaml ./tanks_and_nurseries_config.yaml \
    --thresholds-yaml ./parameter_diagnostics.yaml \
    --combos-csv ./correction_per_combination.csv \
    --actions-csv ./actions_to_observations.csv \
    --sensor-hours-back 24 --label-hours-ahead 6 \
    --anchor-hour-local 12 --tz Europe/Zurich \
    --out-model rf_model.joblib \
    --verbose --subset-tanks "Tank1,NurseryA" --max-rows 5000
"""

from __future__ import annotations

import os, re, csv, glob, math, json, time, sys
import argparse
import datetime as dt
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import joblib

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

# --------------------------- Tiny logger & timer -------------------------------

from datetime import datetime
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

class Timer:
    def __init__(self, label=""):
        self.label = label
        self.t0 = time.perf_counter()
    def done(self, msg=""):
        dtm = time.perf_counter() - self.t0
        if msg:
            print(f"[{_ts()}] {self.label} {msg} ({dtm:.1f}s)")
        else:
            print(f"[{_ts()}] {self.label} done ({dtm:.1f}s)")

# --------------------------- Influx connection --------------------------------

INFLUX_URL  = os.environ.get("INFLUX_URL", "http://10.0.202.244:8086")
INFLUX_DB   = os.environ.get("INFLUX_DB", "ls")
INFLUX_USER = os.environ.get("INFLUX_USER", "lsuser_influxdb")
INFLUX_PASS = os.environ.get("INFLUX_PASS", "w7s9T4U8EAB4BzKaqUx9")
INFLUX_RP   = os.environ.get("INFLUX_RP", "autogen")
TAG_KEY     = os.environ.get("INFLUX_TAG_KEY", "sensorIp")   # tag key for sensor IP

# --------------------------- Helpers: time & auth -----------------------------

def iso_z(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def parse_iso_any(s: str) -> Optional[dt.datetime]:
    if not s: return None
    try:
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        try:
            if s.endswith("Z"):
                return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
            return dt.datetime.fromisoformat(s)
        except Exception:
            return None

def auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/x-www-form-urlencoded"}
    if INFLUX_USER or INFLUX_PASS:
        import base64
        tok = base64.b64encode(f"{INFLUX_USER}:{INFLUX_PASS}".encode()).decode()
        h["Authorization"] = f"Basic {tok}"
    return h

# --------------------------- Normalization maps -------------------------------

# Best-case defaults for daily blanks (your rule)
BEST_DEFAULTS = {
    "Digestive Tract": "80% Shrimps haben vollen Verdaungstrakt",
    "Feed Leftovers": "Keine Futterreste",
    "Dead": "Keine Tote",
    "Foam Formation": "Normale Schaumbildung",
    "Shrimp Fitness": "Gut",
    "Shells": "keine Schalen",
    # dosing binaries -> No/0 if blank
    "Sugar yes/no": "nein",
    "Buffer yes/no": "nein",
    "Vitamin C Amount": 0,
    "Bactosafe Amount": 0,
}

# Canonicalizer (German phrases kept as canonical keys)
CANON_EQUIV = {
    # Digestive tract
    r"^80%.*(voll|verdaung)": "80% Shrimps haben vollen Verdaungstrakt",
    r"^50%.*(voll|verdaung)": "50% Shrimps haben vollen Verdaungstrakt",
    r"^leicht.*gef(ü|u)llt$": "Leicht Gefüllt",
    r"^shrimp.*translu(z|s)ent": "Shrimps merheitlich Transluzent",

    # Leftovers
    r"^(keine|none).*futter(rest|reste)": "Keine Futterreste",
    r"^(wenig|little).*futter(rest|reste)": "Wenig Futterreste",
    r"^(viel|a lot|viele).*futter(rest|reste)": "Viele Futterreste",
    r"^viele futterreste.*": "Viele Futterreste",

    # Dead
    r"^keine tote$": "Keine Tote",
    r"^(wenig|few) tote$": "Wenig Tote",
    r"^(viele|many) tote$": "Viele Tote",
    r"tote.*orange": "Tote sind merheitlich Orange",
    r"tote.*grau": "Tote sind merheitlich Grau",

    # Foam
    r"^normale schaumbildung$": "Normale Schaumbildung",
    r"^wenig.*schaum": "Wenig starke Schaumbildung",
    r"^(starke|heavy).*schaum": "Starke Schaumbildung",

    # Shells
    r"^keine schalen$": "keine Schalen",
    r"^(wenige|few) schalen$": "wenige Schalen",
    r"^(viele|many) schalen$": "viele Schalen",
    r"^sehr viele schalen$": "sehr viele Schalen",

    # Fitness
    r"^gut$": "Gut",
    r"^ok$": "OK",
    r"^schlecht$": "Schlecht",

    # Binary yes/no (DE/EN)
    r"^(ja|yes)$": "ja",
    r"^(nein|no)$": "nein",
}

def canon_token(s: str) -> str:
    if s is None:
        return ""
    s0 = str(s).strip()
    if not s0:
        return ""
    s1 = (s0
          .replace("GefÃ¼llt","Gefüllt")
          .replace("merheitlich","mehrheitlich"))
    s_low = s1.lower().replace("viel futter", "viele futterreste")
    for pat, target in CANON_EQUIV.items():
        if re.search(pat, s_low):
            return target
    return s0  # keep as-is if already canonical

# --------------------------- Thresholds ---------------------------------------

def load_thresholds(path: str) -> Dict[str, Any]:
    import yaml
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    except Exception:
        y = {}
    return y.get("parameters", {})

def bands_for(cfg: Dict[str, Any], key: str):
    thr = (cfg.get(key) or {}).get("thresholds")
    return thr if thr and len(thr) == 5 else None

def classify(cfg: Dict[str, Any], key: str, v: float) -> str:
    thr = bands_for(cfg, key)
    if not thr or v is None or (isinstance(v, float) and math.isnan(v)):
        return "Unknown"
    t0,t1,_t2,t3,t4 = thr
    if key.lower().startswith("turbidity"):
        # Special rule: 250-400 is HIGH; >400 already dropped upstream
        if v >= 250: return "High"
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

# --------------------------- Influx pulling -----------------------------------

def field_candidates(label: str) -> List[str]:
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
    if "sal" in sl or "salinity" in sl or "salinit" in sl:
        out += ["Salinity", "salinity", "Salinität"]
    if "turb" in sl or "ntu" in sl:
        out += ["Turbidity", "NTU", "turbidity"]
    # dedup
    seen=set(); dedup=[]
    for x in out:
        if x and x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def fetch_series_influx(ip: str, measurement: str, field_label: str,
                        start_iso: str, end_iso: str) -> Tuple[List[dt.datetime], List[float]]:
    if not (INFLUX_URL and INFLUX_DB):
        return [], []
    headers = auth_headers()
    meas_ref = f'"{INFLUX_RP}"."{measurement}"'
    for fld in field_candidates(field_label):
        q = (f'SELECT "{fld}" FROM {meas_ref} '
             f'WHERE "{TAG_KEY}" = \'{ip}\' AND time >= \'{start_iso}\' AND time <= \'{end_iso}\'')
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
                # Turbidity > 400 -> drop entirely
                if str(field_label).lower().startswith("turbidity"):
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if fv > 400:
                        continue
                    v = fv
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

# --------------------------- Tank map -----------------------------------------

TANKS_AND_NURSERIES: Dict[str, Dict[str, Tuple[str,str]]] = {}

def load_tank_map(path: str):
    import yaml
    global TANKS_AND_NURSERIES
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

# --------------------------- CO2 water ----------------------------------------

def compute_cwater(T: float, A: float) -> float:
    try:
        KH = (76.81167 - 3.348169*T + 0.1198361*T**2 - 0.0031048148*T**3 + 0.00003566666668*T**4)
        return ((44.01/22.4)*A*273/(273+T)*(101325/100100)/(1000*1000)/44.01*KH*KH*44)
    except Exception:
        return float("nan")

# --------------------------- Sensor feature builder ---------------------------

def slice_idx(dts: List[dt.datetime], end: dt.datetime, hours: int) -> List[int]:
    start = end - dt.timedelta(hours=hours)
    return [i for i, t in enumerate(dts) if start < t <= end]

def recent_flags(cfg: Dict[str, Any], key: str, dts: List[dt.datetime], vals: List[float], end: dt.datetime) -> Tuple[int,int]:
    idx = slice_idx(dts, end, 2)
    red = yellow = 0
    for i in idx:
        state = classify(cfg, key, vals[i])
        if state in ("High","Low"): red = 1
        elif state in ("BHigh","BLow"): yellow = 1
    return red, yellow

def build_sensor_feats_for_window(tank: str, thresholds: Dict[str, Any],
                                  end: dt.datetime, hours_back: int) -> Tuple[Dict[str,float], Dict[str,float]]:
    conf = TANKS_AND_NURSERIES.get(tank) or {}
    if not conf:
        return {}, {}
    start = end - dt.timedelta(hours=hours_back)
    start_iso, end_iso = iso_z(start), iso_z(end)

    feats: Dict[str,float] = {}
    latest_raw: Dict[str,float] = {}
    series: Dict[str, Tuple[List[dt.datetime], List[float]]] = {}

    for label, (measurement, ip) in conf.items():
        dts, vs = fetch_series_influx(ip, measurement, label, start_iso, end_iso)
        if dts:
            series[label] = (dts, vs)

    for k, (dts, vals) in series.items():
        last_idx = slice_idx(dts, end, hours_back)
        idx6     = slice_idx(dts, end, 6)
        if last_idx:
            last_v = vals[last_idx[-1]]
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
        # 2h flags handled in labeler (at lookahead point)

    # Derived CO2_water
    if "Temperature__last" in feats and "CO2 concentration__last" in feats:
        feats["CO2_water__last"] = compute_cwater(feats["Temperature__last"], feats["CO2 concentration__last"])
        latest_raw["CO2_water"]  = feats["CO2_water__last"]

    return feats, latest_raw

# --------------------------- CSV ingestion ------------------------------------

# --- keep these earlier in the file ---
OBS_COLS = [
    "Digestive Tract", "Feed Leftovers", "Dead", "Foam Formation",
    "Shrimp Fitness", "Shells", "Correction %", "Adjusted Feed Amount (g)",
    "Sugar Amount", "Sugar yes/no", "Buffer Amount", "Buffer yes/no",
    "Vitamin C Amount", "Bactosafe Amount"
]

def best_default(name: str, val):
    if (val is None or (isinstance(val, str) and not val.strip())) and name in BEST_DEFAULTS:
        return BEST_DEFAULTS[name]
    return val

def normalize_obs_row(row: pd.Series) -> Dict[str, Any]:
    out = {}
    for k in OBS_COLS:
        v = best_default(k, row.get(k))
        if isinstance(v, str):
            v = v.strip()
        out[k] = v
    return out

def tokens_from_obs(obs: Dict[str, Any]) -> List[str]:
    """Turn one normalized observation row into a set of canonical tokens."""
    toks: List[str] = []

    # Digestive tract
    dtv = canon_token(obs.get("Digestive Tract"))
    if dtv:
        toks.append(dtv)

    # Leftover feed
    lv = canon_token(obs.get("Feed Leftovers"))
    if lv:
        toks.append(lv)

    # Dead
    dv = canon_token(obs.get("Dead"))
    if dv:
        toks.append(dv)

    # Foam
    fv = canon_token(obs.get("Foam Formation"))
    if fv:
        toks.append(fv)

    # Shells
    sv = canon_token(obs.get("Shells"))
    if sv:
        toks.append(sv)

    # Fitness (optional)
    fit = canon_token(obs.get("Shrimp Fitness"))
    if fit:
        toks.append(fit)

    # Binary dosing flags
    sugar_yes = canon_token(obs.get("Sugar yes/no"))
    if sugar_yes == "ja":
        toks.append("Zuckerdosierung")

    buffer_yes = canon_token(obs.get("Buffer yes/no"))
    if buffer_yes == "ja":
        toks.append("Buffer angewendet")

    # Numeric dosing
    try:
        if float(obs.get("Vitamin C Amount") or 0) > 0:
            toks.append("Vitamin C angewendet")
    except Exception:
        pass

    try:
        if float(obs.get("Bactosafe Amount") or 0) > 0:
            toks.append("Bactosafe angewendet")
    except Exception:
        pass

    # Deduplicate & sort for stable downstream joins
    return sorted(set(toks))

# --------------------------- Flexible CSV readers -----------------------------

def _sniff_delim(path: str, default=";") -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
            return dialect.delimiter or default
        except Exception:
            return default
    except Exception:
        return default

def _read_relaxed_csv(path: str, prefer: str = ";"):
    tried = []
    for sep in [prefer, _sniff_delim(path, prefer), ";", ",", "\t", "|"]:
        if sep in tried: continue
        tried.append(sep)
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine="python",
                encoding="utf-8-sig",
                na_filter=False,
                on_bad_lines="skip"
            )
            if not df.empty and df.shape[1] >= 1:
                return df, sep
        except Exception:
            continue
    # last resort: first two semicolon fields
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) >= 2:
                rows.append({"_col0": parts[0], "_col1": parts[1]})
    if not rows:
        raise RuntimeError(f"Could not read CSV: {path}")
    df = pd.DataFrame(rows)
    return df, ";"

def _find_col(df, candidates_lower: List[str]):
    cols = list(df.columns)
    for c in cols:
        cl = str(c).strip().lower()
        for want in candidates_lower:
            if cl == want or cl.startswith(want):
                return c
    return None

def _to_float_corr(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s).strip()
    s = s.replace("\u2212", "-").replace("%", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        v = pd.to_numeric(s, errors="coerce")
        return float(v) if v == v else 0.0

# --------------------------- Kombination CSV ----------------------------------

def load_combos(path: str) -> List[Tuple[List[str], float]]:
    """
    correction_per_combination.csv columns:
      - 'Kombination' (comma-separated phrases)
      - 'Korrektur (Futter)' (numeric %, e.g. -50)
    Returns [(tokens, correction_float), ...]
    """
    df, used_sep = _read_relaxed_csv(path, prefer=";")
    kcol = _find_col(df, ["kombination", "_col0"])
    ccol = _find_col(df, ["korrektur (futter)","korrektur","korrekturfutter","korrektur%","_col1"])
    if kcol is None or ccol is None:
        raise RuntimeError(f"Missing required columns in {path}. "
                           f"Found columns: {list(df.columns)}; need 'Kombination' and 'Korrektur (Futter)'.")
    combos: List[Tuple[List[str], float]] = []
    for _, r in df.iterrows():
        raw = str(r.get(kcol, "")).strip()
        if not raw or raw.lower().startswith("kombination"):
            continue
        corr = _to_float_corr(r.get(ccol, "0"))
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        toks = []
        for p in parts:
            tok = canon_token(p)
            if tok:
                toks.append(tok)
        toks = sorted(set(toks))
        if toks:
            combos.append((toks, corr))
    if not combos:
        raise RuntimeError(f"No valid combinations parsed from {path}. Check 'Kombination' cells.")
    return combos

# --------------------------- Actions CSV --------------------------------------

def load_actions(path: str) -> Dict[str, str]:
    """
    actions_to_observations.csv columns:
      - 'Situation'
      - 'Bemerkungen'
    Returns: { canonical_situation: remark_text }
    """
    df, _sep = _read_relaxed_csv(path, prefer=";")
    scol = _find_col(df, ["situation", "obs", "observation"])
    bcol = _find_col(df, ["bemerkungen", "bemerkung", "comment", "hinweis"])
    if scol is None or bcol is None:
        raise RuntimeError(f"Missing columns in {path}. Got {list(df.columns)}; need 'Situation' and 'Bemerkungen'.")
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        sraw = str(r.get(scol, "")).strip()
        if not sraw or sraw.lower().startswith("situation"):
            continue
        key = canon_token(sraw)
        if not key:
            continue
        remark = str(r.get(bcol, "")).strip()
        if key in out:
            if remark and remark not in out[key]:
                out[key] = (out[key] + " | " + remark) if out[key] else remark
        else:
            out[key] = remark
    return out

# --------------------------- Row & label builders -----------------------------

def build_row(tank: str,
              day_local: dt.date,
              anchor_hour_local: int,
              tz: str,
              thresholds: Dict[str, Any],
              sensor_hours_back: int,
              obs_tokens: List[str],
              combos_spec: List[Tuple[List[str], float]],
              actions_notes: Dict[str, str]) -> Dict[str, Any]:

    tzinfo = pd.Timestamp(day_local.isoformat()).tz_localize(tz).tzinfo
    end_local = dt.datetime.combine(day_local, dt.time(hour=anchor_hour_local)).replace(tzinfo=tzinfo)
    end = end_local.astimezone(dt.timezone.utc)

    feats, latest_raw = build_sensor_feats_for_window(tank, thresholds, end=end, hours_back=sensor_hours_back)
    row = {"tank": tank, "day": day_local.isoformat(), "anchor_ts_utc": iso_z(end)}

    for k, v in feats.items():
        row[k] = v

    # combo features
    combo_sum = 0.0
    combo_hits = 0
    for toks, corr in combos_spec:
        if set(toks).issubset(set(obs_tokens)):
            combo_hits += 1
            combo_sum += corr
            row["combo::" + " & ".join(toks)] = 1.0
    row["combo_hit_any"]   = 1.0 if combo_hits > 0 else 0.0
    row["combo_hit_count"] = float(combo_hits)
    row["combo_corr_reco"] = float(combo_sum)

    for t in obs_tokens:
        row["tok__" + t] = 1.0

    matched_adv = [actions_notes[t] for t in obs_tokens if t in actions_notes and actions_notes[t]]
    row["advice_notes"] = "; ".join(matched_adv) if matched_adv else ""

    return row

def label_from_future(thresholds: Dict[str, Any],
                      tank: str,
                      after_ts_utc: dt.datetime,
                      hours_ahead: int) -> int:
    end = after_ts_utc + dt.timedelta(hours=hours_ahead)
    feats, _ = build_sensor_feats_for_window(tank, thresholds, end=end, hours_back=2)
    for k, v in feats.items():
        if k.endswith("__flag_red_2h") and float(v) >= 1.0:
            return 1
    return 0

# --------------------------- Training -----------------------------------------

def train_rf_calibrated(X: pd.DataFrame, y: np.ndarray, out_model: str, verbose: bool = True):
    # GroupKFold by tank_group is handled outside; here do a final fit+calibration
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42
        ))
    ])
    clf = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    if verbose:
        print(f"[{_ts()}] fitting calibrated RF on full data: n={len(X)}, d={X.shape[1]} …")
    clf.fit(X, y)
    joblib.dump({"model": clf, "features": list(X.columns)}, out_model)
    if verbose:
        print(f"[{_ts()}] saved model: {out_model}")
    return clf

# --------------------------- Main ---------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-data-dir", default="./output_data")
    ap.add_argument("--map-yaml", required=True)
    ap.add_argument("--thresholds-yaml", required=True)
    ap.add_argument("--combos-csv", required=True)
    ap.add_argument("--actions-csv", required=True)

    ap.add_argument("--sensor-hours-back", type=int, default=24)
    ap.add_argument("--label-hours-ahead", type=int, default=6)
    ap.add_argument("--anchor-hour-local", type=int, default=12)
    ap.add_argument("--tz", default="Europe/Zurich")

    ap.add_argument("--out-model", default="rf_model.joblib")

    # Debug/verbosity helpers
    ap.add_argument("--verbose", action="store_true", help="print progress")
    ap.add_argument("--subset-tanks", default="", help="comma-separated tank names")
    ap.add_argument("--max-rows", type=int, default=0, help="limit rows for smoke test")

    args = ap.parse_args()

    if not (INFLUX_URL and INFLUX_DB):
        print("[error] set INFLUX_URL and INFLUX_DB", file=sys.stderr); raise SystemExit(2)

    if args.verbose:
        print(f"[{_ts()}] starting RFCtrain1")
        print(f"[{_ts()}] hours_back={args.sensor_hours_back}, lookahead={args.label_hours_ahead}h, anchor={args.anchor_hour_local} {args.tz}")

    t = Timer("load maps/thresholds")
    load_tank_map(args.map_yaml)
    thresholds = load_thresholds(args.thresholds_yaml)
    t.done()

    t = Timer("load combos")
    combos_spec = load_combos(args.combos_csv)
    if args.verbose:
        print(f"[{_ts()}] combos: {len(combos_spec)} (ex: {combos_spec[:3]})")
    t.done()

    t = Timer("load actions")
    actions_notes = load_actions(args.actions_csv)
    if args.verbose:
        print(f"[{_ts()}] actions: {len(actions_notes)}")
    t.done()

    # Ingest daily CSVs
    paths = sorted(glob.glob(os.path.join(args.output_data_dir, "*.csv")))
    if not paths:
        print("[error] no CSVs in", args.output_data_dir, file=sys.stderr); raise SystemExit(3)

    subset = None
    if args.subset_tanks.strip():
        subset = [s.strip() for s in args.subset_tanks.split(",") if s.strip()]
        if args.verbose:
            print(f"[{_ts()}] limiting to tanks: {subset}")

    rows = []
    total_points = 0
    print(f"[{_ts()}] building dataset from {len(paths)} CSVs …")
    for p in paths:
        try:
            df = pd.read_csv(p, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(p, encoding="latin-1")
        fname = os.path.basename(p)

        # Heuristic tank from filename like masterData_T1B1516.csv or NAB2
        m = re.search(r'(N[A-F]|T\d)B(\d+)', fname, re.IGNORECASE)
        tank_name = None
        if m:
            code = m.group(1).upper()
            if code.startswith("N"):
                tank_name = {
                    "NA":"NurseryA","NB":"NurseryB","NC":"NurseryC",
                    "ND":"NurseryD","NE":"NurseryE","NF":"NurseryF"
                }.get(code[:2], None)
            elif code.startswith("T"):
                tank_name = f"Tank{code[1:]}"
        if "Tank" in df.columns and not tank_name:
            tank_name = str(df["Tank"].iloc[0]).strip()

        if not tank_name or tank_name not in TANKS_AND_NURSERIES:
            print(f"[warn] skip {fname}: cannot resolve tank name", file=sys.stderr)
            continue
        if subset and tank_name not in subset:
            continue

        # date column
        dcol = None
        for c in df.columns:
            if str(c).strip().lower() in ("date","datum","day"):
                dcol = c; break
        if not dcol:
            print(f"[warn] skip {fname}: no Date column", file=sys.stderr)
            continue

        for _, r in df.iterrows():
            raw_date = r.get(dcol)
            if not pd.notna(raw_date):
                continue
            try:
                d_local = pd.to_datetime(str(raw_date), dayfirst=True).date()
            except Exception:
                continue

            obs = normalize_obs_row(r)
            toks = tokens_from_obs(obs)

            row = build_row(
                tank=tank_name,
                day_local=d_local,
                anchor_hour_local=args.anchor_hour_local,
                tz=args.tz,
                thresholds=thresholds,
                sensor_hours_back=args.sensor_hours_back,
                obs_tokens=toks,
                combos_spec=combos_spec,
                actions_notes=actions_notes
            )
            anchor_ts = parse_iso_any(row["anchor_ts_utc"])
            y = label_from_future(thresholds, tank_name, anchor_ts, args.label_hours_ahead)
            row["label_red_6h"] = int(y)
            row["tank_group"] = tank_name
            rows.append(row)
        if args.verbose:
            print(f"[{_ts()}] {fname}: +{len(df)} days -> rows={len(rows)}")

    if not rows:
        print("[error] no rows built; check inputs", file=sys.stderr); raise SystemExit(4)

    df = pd.DataFrame(rows)
    if args.max_rows and len(df) > args.max_rows:
        df = df.iloc[:args.max_rows].copy()

    # Split features / label
    y = df["label_red_6h"].values.astype(int)
    groups = df["tank_group"].values

    drop_cols = ["tank","day","anchor_ts_utc","advice_notes","label_red_6h","tank_group"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # ensure numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # GroupKFold CV (by tank)
    n_splits = min(5, len(np.unique(groups))) if len(np.unique(groups)) >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []; praucs = []
    print(f"[{_ts()}] CV (GroupKFold={n_splits}) on {len(y)} rows, {X.shape[1]} features …")
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xt, Xv = X.iloc[tr], X.iloc[te]
        yt, yv = y[tr], y[te]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=42
            ))
        ])
        clf = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        t = Timer(f"fold {fold}")
        clf.fit(Xt, yt)
        p = clf.predict_proba(Xv)[:,1]
        auc = roc_auc_score(yv, p)
        pr  = average_precision_score(yv, p)
        aucs.append(auc); praucs.append(pr)
        t.done(f"AUC={auc:.3f} PR={pr:.3f}")

    print(f"[cv] ROC AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}   PR AUC={np.mean(praucs):.3f} ± {np.std(praucs):.3f}   (n={len(y)})")

    # Final fit + save
    clf = train_rf_calibrated(X, y, out_model=args.out_model, verbose=True)

    # Save training bundle extras (combos/actions/thresholds) for use by predictor
    bundle = joblib.load(args.out_model)
    bundle.update({
        "combos_spec": combos_spec,
        "actions_notes": actions_notes,
        "thresholds": thresholds,
        "built_at": dt.datetime.utcnow().isoformat()+"Z",
        "sensor_hours_back": args.sensor_hours_back,
        "label_hours_ahead": args.label_hours_ahead,
        "anchor_hour_local": args.anchor_hour_local,
        "tz": args.tz,
    })
    joblib.dump(bundle, args.out_model)
    print(f"[ok] saved model bundle: {args.out_model}")
    print(f"[info] features used: {len(bundle['features'])}")

    # Optional: small debug sample
    try:
        df.head(50).to_csv("debug_sample.csv", index=False)
        print(f"[{_ts()}] wrote debug_sample.csv")
    except Exception:
        pass

if __name__ == "__main__":
    main()
