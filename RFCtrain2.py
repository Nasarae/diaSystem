#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RFCtrain2.py — LuckyShrimp model trainer

Features:
- Influx preloading + caching
- Robust CSV parsing (daily logs, combos, actions)
- German observation canonicalization
- Threshold-based feature engineering + flags
- Optional occupancy filtering (active_only / weighted / all)
- Coverage guard (minimum samples in last 6h)
- StratifiedGroupKFold when available; safe handling of one-class folds
- Calibrated RandomForest probabilities (fallback to uncalibrated if single class)
"""

import os, re, sys, csv, glob, math, time, json
import datetime as dt
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import requests
import joblib
import unicodedata

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

# --------------------------- Globals / ENV ------------------------------------

INFLUX_URL  = os.environ.get("INFLUX_URL", "http://10.0.202.244:8086")
INFLUX_DB   = os.environ.get("INFLUX_DB", "ls")
INFLUX_USER = os.environ.get("INFLUX_USER", "lsuser_influxdb")
INFLUX_PASS = os.environ.get("INFLUX_PASS", "w7s9T4U8EAB4BzKaqUx9")
INFLUX_RP   = os.environ.get("INFLUX_RP", "autogen")
TAG_KEY     = os.environ.get("INFLUX_TAG_KEY", "sensorIp")   # tag key for sensor IP

# minimum sample count in last 6h for coverage
MIN_SAMPLES_6H = int(os.environ.get("MIN_SAMPLES_6H", "3"))
UNMAPPED_OBS = set()

# --------------------------- Helpers: time & auth -----------------------------

def iso_z(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def parse_iso_any(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        try:
            if s.endswith("Z"):
                return dt.datetime.fromisoformat(s.replace("Z","+00:00"))
            return dt.datetime.fromisoformat(s)
        except Exception:
            return None

def base64_b64encode(s: str) -> str:
    import base64
    return base64.b64encode(s.encode()).decode()

def auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/x-www-form-urlencoded"}
    if INFLUX_USER or INFLUX_PASS:
        tok = base64_b64encode(f"{INFLUX_USER}:{INFLUX_PASS}")
        h["Authorization"] = f"Basic {tok}"
    return h

# --------------------------- Normalization maps -------------------------------

BEST_DEFAULTS = {
    "Digestive Tract": "80% Shrimps haben vollen Verdaungstrakt",
    "Feed Leftovers": "Keine Futterreste",
    "Dead": "Keine Tote",
    "Foam Formation": "Normale Schaumbildung",
    "Shrimp Fitness": "Gut",
    "Shells": "keine Schalen",
    "Sugar yes/no": "nein",
    "Buffer yes/no": "nein",
    "Vitamin C Amount": 0,
    "Bactosafe Amount": 0,
}

# ---------------- Canonical observation tokens ----------------
# Choose canonical GERMAN tokens (keys below). Map all variants → these keys.

CANON_TOKENS = {
    # Digestive tract
    "80% Shrimps haben vollen Verdaungstrakt",
    "50% Shrimps haben vollen Verdaungstrakt",
    "Leicht Gefüllt",
    "Shrimps merheitlich Transluzent",

    # Feed leftovers
    "Keine Futterreste",
    "Wenig Futterreste",
    "Viele Futterreste",

    # Dead
    "Keine Tote",
    "Wenig Tote",
    "Viele Tote",
    "Tote sind merheitlich Orange",
    "Tote sind merheitlich Grau",
    "Tote, Orang und Grau gemischt",

    # Foam
    "Normale Schaumbildung",
    "Wenig starke Schaumbildung",
    "Starke Schaumbildung",

    # Shells
    "keine Schalen",
    "wenige Schalen",
    "viele Schalen",
    "sehr viele Schalen",

    # Fitness
    "Gut",
    "OK",
    "Schlecht",

    # Treatments / actions toggles
    "Zuckerdosierung",
    "Buffer angewendet",
    "Vitamin C angewendet",
    "Bactosafe angewendet",

    "Shrimps gestresst/ weißlich, Darm kaum erkennbar",
    "Shrimps haben mehrheitlich weiche Panzer",
    "Shrimps lethargisch / 'Korking'",
}

# 2a) Exact/alias mapping: EN & DE, casing, common typos → canonical
ALIASES = {
    # Digestive tract (EN)
    "80% of shrimps have full digestive tracts": "80% Shrimps haben vollen Verdaungstrakt",
    "50% of shrimp have full digestive tracts": "50% Shrimps haben vollen Verdaungstrakt",
    "lightly filled": "Leicht Gefüllt",
    "shrimps are mostly translucent": "Shrimps merheitlich Transluzent",
    # common DE typos
    "leicht gefüllt": "Leicht Gefüllt",
    "leicht gefuellt": "Leicht Gefüllt",
    "shrimps mehrheitlich transluzent": "Shrimps merheitlich Transluzent",
    "shrimp merheitlich transluzent": "Shrimps merheitlich Transluzent",

    # Feed leftovers
    "no leftover feed": "Keine Futterreste",
    "little leftover feed": "Wenig Futterreste",
    "a lot of leftover food!": "Viele Futterreste",

    # Dead
    "no dead": "Keine Tote",
    "no dead ones": "Keine Tote",
    "few dead": "Wenig Tote",
    "many dead": "Viele Tote",
    "dead ones are mostly orange": "Tote sind merheitlich Orange",
    "dead ones are mostly gray": "Tote sind merheitlich Grau",
    "dead ones, orange and gray mixed": "Tote, Orang und Grau gemischt",

    # Foam
    "normal foaming": "Normale Schaumbildung",
    "little foaming": "Wenig starke Schaumbildung",
    "heavy foaming": "Starke Schaumbildung",

    # Shells
    "no shells": "keine Schalen",
    "few shells": "wenige Schalen",
    "many shells": "viele Schalen",
    # your EN source had “sehr viele Schalen”→“Many shells”; keep explicit:
    "very many shells": "sehr viele Schalen",

    # Fitness
    "good": "Gut",
    "ok": "OK",
    "schlecht": "Schlecht",  # case-folded DE

    # Treatments (binary)
    "yes": "ja",
    "no": "nein",

    # Soft shells (EN → DE)
    "shrimps have mostly soft shells": "Shrimps haben mehrheitlich weiche Panzer",
    "shrimp have mostly soft shells": "Shrimps haben mehrheitlich weiche Panzer",
    "mostly soft shells": "Shrimps haben mehrheitlich weiche Panzer",

    # Stressed / whitish, intestine barely visible (EN → DE)
    "shrimps stressed/whitish, intestine barely visible": "Shrimps gestresst/ weißlich, Darm kaum erkennbar",
    "shrimps stressed, whitish, intestine barely visible": "Shrimps gestresst/ weißlich, Darm kaum erkennbar",
    "stressed whitish intestine barely visible": "Shrimps gestresst/ weißlich, Darm kaum erkennbar",

    # Lethargic / corking (DE/EN variants → DE)
    'letargisch, "korking"': "Shrimps lethargisch / 'Corking'",
    "letargisch, 'korking'": "Shrimps lethargisch / 'Corking'",
    "lethargic, corking": "Shrimps lethargisch / 'Corking'",
    "corking": "Shrimps lethargisch / 'Corking'",
    "lethargic": "Shrimps lethargisch / 'Corking'",
}

# 2b) Light regex fallbacks (case-insensitive) for noisy phrasings
CANON_REGEX = [
    (r"^\s*80%\b.*(voll|verdaung)", "80% Shrimps haben vollen Verdaungstrakt"),
    (r"^\s*50%\b.*(voll|verdaung)", "50% Shrimps haben vollen Verdaungstrakt"),
    (r"^leicht.*gef(ü|u)llt", "Leicht Gefüllt"),
    (r"shrimp[s]?\s*(are|merheitlich|mehrheitlich)?.*transluzent", "Shrimps merheitlich Transluzent"),

    (r"(keine|no).*futter(rest|reste|leftover)", "Keine Futterreste"),
    (r"(wenig|little).*futter(rest|reste|leftover)", "Wenig Futterreste"),
    (r"(viele|many|a lot).*futter(rest|reste|leftover)", "Viele Futterreste"),

    (r"(keine|no).*tote|no dead( ones)?\b", "Keine Tote"),
    (r"(wenig|few).*tote|few dead\b", "Wenig Tote"),
    (r"(viele|many).*tote|many dead\b", "Viele Tote"),
    (r"tote.*orange|dead.*orange", "Tote sind merheitlich Orange"),
    (r"tote.*grau|dead.*gray|dead.*grey", "Tote sind merheitlich Grau"),

    (r"normale.*schaum|normal.*foam", "Normale Schaumbildung"),
    (r"(wenig|little).*schaum|slight.*foam", "Wenig starke Schaumbildung"),
    (r"(starke|heavy|strong).*schaum|heavy.*foam", "Starke Schaumbildung"),

    (r"keine schalen|no shells", "keine Schalen"),
    (r"(wenige|few) schalen|few shells", "wenige Schalen"),
    (r"(viele|many) schalen|many shells", "viele Schalen"),
    (r"sehr viele schalen|very many shells", "sehr viele Schalen"),

    (r"^gut$", "Gut"),
    (r"^ok$", "OK"),
    (r"^schlecht$", "Schlecht"),

    # Soft shells
    (r"\bsoft\s+shells?\b", "Shrimps haben mehrheitlich weiche Panzer"),
    (r"weiche\s+panzer",    "Shrimps haben mehrheitlich weiche Panzer"),

    # Stressed/whitish + intestine hardly visible
    (r"(stressed|stress|gestresst).*(whitish|weißlich|weisslich).*(intestin|darm).*(barely|kaum)",
     "Shrimps gestresst/ weißlich, Darm kaum erkennbar"),

    # Lethargic / corking (spelling: korking/corking)
    (r"(lethargic|letargisch|corking|korking)", "Shrimps lethargisch / 'Corking'"),
]

def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip()
    s = s.replace("GefÃ¼llt", "Gefüllt")  # legacy decode
    s = s.replace("merheitlich", "mehrheitlich")  # fix known typo
    s = s.replace("ernkennbar", "erkennbar")  # typo seen in earlier data
    s = s.replace("weisslich", "weißlich")
    return s

def _casefold(s: str) -> str:
    return _norm_text(s).casefold()

def canon_token(s: str) -> str:
    if s is None:
        return ""
    s0 = _norm_text(str(s))
    if not s0:
        return ""
    # exact alias (case-insensitive)
    key = _casefold(s0)
    if key in ALIASES:
        return ALIASES[key]
    # regex fallback
    low = key
    for pat, target in CANON_REGEX:
        if re.search(pat, low, flags=re.I):
            return target
    # if the original is already one of the canonical tokens, keep it
    if s0 in CANON_TOKENS:
        return s0
    # last resort: return the original untouched (so we can log and learn)
    return s0
# --------------------------- Thresholds ---------------------------------------

def load_thresholds(path: str) -> Dict[str, Any]:
    try:
        import yaml
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
        if v >= 250:
            return "High"
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
    out = [s]; sl = s.lower()
    if sl in ("temp", "temperature"): out += ["Temperature", "temp", "Temp"]
    if sl in ("ph", "pH".lower()):   out += ["pH", "PH", "ph"]
    if "do" in sl or "dissolved" in sl:
        out += ["DO mg/L", "DO", "dissolved_oxygen", "do_mgL", "DO_mg_L"]
    if "co2" in sl: out += ["CO2 concentration", "CO2", "CO2_ppm"]
    if "sal" in sl: out += ["Salinity", "salinity", "Salinität"]
    if "turb" in sl or "ntu" in sl: out += ["Turbidity", "NTU", "turbidity"]
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
            r = requests.get(f"{INFLUX_URL}/query",
                             params={"db": INFLUX_DB, "q": q},
                             headers=headers, timeout=15)
            j = r.json()
            series = j.get("results", [{}])[0].get("series", [])
            if not series or not series[0].get("values"):
                continue
            ts_raw, vs = [], []
            for t_iso, v in series[0]["values"]:
                if v is None:
                    continue
                # Turbidity: drop > 400 entirely
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
                dts = list(pd.to_datetime(ts_raw, utc=True).to_pydatetime())
                return dts, vs
        except Exception:
            continue
    return [], []

# --------------------------- Tank map -----------------------------------------

TANKS_AND_NURSERIES: Dict[str, Dict[str, Tuple[str,str]]] = {}

def load_tank_map(path: str):
    global TANKS_AND_NURSERIES
    import yaml
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

# --------------------------- Feature builder ----------------------------------

def slice_idx(dts: List[dt.datetime], end: dt.datetime, hours: int) -> List[int]:
    start = end - dt.timedelta(hours=hours)
    return [i for i, t in enumerate(dts) if start < t <= end]

def recent_flags(cfg: Dict[str, Any], key: str, dts: List[dt.datetime], vals: List[float], end: dt.datetime) -> Tuple[int,int]:
    idx = slice_idx(dts, end, 2)
    red = yellow = 0
    for i in idx:
        state = classify(cfg, key, vals[i])
        if state in ("High","Low"):
            red = 1
        elif state in ("BHigh","BLow"):
            yellow = 1
    return red, yellow

# global cache: (tank, label) -> (dts, vals)
SERIES_CACHE: Dict[Tuple[str,str], Tuple[List[dt.datetime], List[float]]] = {}

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
        if SERIES_CACHE:
            dts, vs = SERIES_CACHE.get((tank, label), ([], []))
        else:
            dts, vs = fetch_series_influx(ip, measurement, label, start_iso, end_iso)
        if dts:
            series[label] = (dts, vs)

    # compute features
    coverage_ok = False
    for k, (dts, vals) in series.items():
        idx_all = slice_idx(dts, end, hours_back)
        idx6    = slice_idx(dts, end, 6)
        if idx_all:
            last_v = vals[idx_all[-1]]
            latest_raw[k] = last_v
            feats[f"{k}__last"] = float(last_v) if last_v is not None else np.nan
            feats[f"{k}__dev_center"] = dev_center(thresholds, k, feats[f"{k}__last"])
        if idx6:
            arr = np.array([vals[i] for i in idx6], dtype=float)
            xs  = np.array([dts[i].timestamp() for i in idx6], dtype=float)
            if arr.size >= MIN_SAMPLES_6H:
                coverage_ok = True
            feats[f"{k}__mean_6h"] = float(np.nanmean(arr)) if arr.size else np.nan
            feats[f"{k}__std_6h"]  = float(np.nanstd(arr))  if arr.size else np.nan
            feats[f"{k}__min_6h"]  = float(np.nanmin(arr))  if arr.size else np.nan
            feats[f"{k}__max_6h"]  = float(np.nanmax(arr))  if arr.size else np.nan
            if xs.size >= 2:
                X = np.vstack([xs - xs.mean(), np.ones_like(xs)]).T
                m, _c = np.linalg.lstsq(X, arr, rcond=None)[0]
                feats[f"{k}__slope_6h"] = float(m)
        red, yellow = recent_flags(thresholds, k, dts, vals, end)
        feats[f"{k}__flag_red_2h"] = red
        feats[f"{k}__flag_yellow_2h"] = yellow

    # Derived CO2_water
    if "Temperature__last" in feats and "CO2 concentration__last" in feats:
        feats["CO2_water__last"] = compute_cwater(feats["Temperature__last"], feats["CO2 concentration__last"])
        latest_raw["CO2_water"]  = feats["CO2_water__last"]

    # mark coverage flag
    feats["_coverage_6h_ok"] = 1 if coverage_ok else 0
    return feats, latest_raw

# --------------------------- CSV ingestion ------------------------------------

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
    toks = []

    for key in ["Digestive Tract", "Feed Leftovers", "Dead", "Foam Formation", "Shells", "Shrimp Fitness"]:
        raw = obs.get(key)
        tok = canon_token(raw)
        if raw and tok == raw and tok not in CANON_TOKENS:
            UNMAPPED_OBS.add(str(raw))
        if tok:
            toks.append(tok)

    if canon_token(obs.get("Sugar yes/no")) == "ja":
        toks.append("Zuckerdosierung")
    if canon_token(obs.get("Buffer yes/no")) == "ja":
        toks.append("Buffer angewendet")

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

    return sorted(set(toks))

# --------------------------- combos/actions CSVs ------------------------------

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
        if sep in tried:
            continue
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
    # final manual fallback: first two semicolon fields
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
    s = str(s).strip().replace("\u2212", "-").replace("%", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        v = pd.to_numeric(s, errors="coerce")
        return float(v) if v == v else 0.0

def combos_match(tokens_day: List[str], combo_tokens: List[str]) -> bool:
    return all(t in set(tokens_day) for t in combo_tokens)

def load_combos(path: str) -> List[Tuple[List[str], float]]:
    df, _sep = _read_relaxed_csv(path, prefer=";")
    kcol = _find_col(df, ["kombination"])
    ccol = _find_col(df, ["korrektur (futter)", "korrektur"])
    if kcol is None or ccol is None:
        raise RuntimeError(f"Missing columns in {path}. Have {list(df.columns)}; need 'Kombination' & 'Korrektur (Futter)'.")
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
    return combos

def load_actions(path: str) -> Dict[str, str]:
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
        if key not in out:
            out[key] = remark
        else:
            if remark and remark not in out[key]:
                out[key] = (out[key] + " | " + remark) if out[key] else remark
    return out

# --------------------------- Occupancy calendar -------------------------------

def parse_tank_and_batch_from_fname(fname: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r'(N[A-F]|T\d)B(\d+)', fname, re.IGNORECASE)
    if not m:
        return None, None
    code = m.group(1).upper()
    bid  = m.group(2)
    if code.startswith("N"):
        tank = {
            "NA":"NurseryA","NB":"NurseryB","NC":"NurseryC",
            "ND":"NurseryD","NE":"NurseryE","NF":"NurseryF"
        }.get(code[:2], None)
    else:
        tank = f"Tank{code[1:]}"
    batch_id = f"B{bid}"
    return tank, batch_id

def build_occupancy_from_csvs(paths: List[str], gap_close_days=2, pad_days=0):
    from collections import defaultdict
    by_tank_batch_dates = defaultdict(list)
    for p in paths:
        fname = os.path.basename(p)
        tank, batch = parse_tank_and_batch_from_fname(fname)
        if not tank or not batch:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        dcol = None
        for c in df.columns:
            if str(c).strip().lower() in ("date","datum","day"):
                dcol = c; break
        if not dcol:
            continue
        dd = pd.to_datetime(df[dcol], dayfirst=True, errors="coerce").dropna().dt.date.tolist()
        by_tank_batch_dates[(tank,batch)].extend(dd)

    occ = {}
    for (tank,batch), dates in by_tank_batch_dates.items():
        if not dates:
            continue
        uniq = sorted(set(dates))
        runs = []
        run_start = uniq[0]; prev = uniq[0]
        for d in uniq[1:]:
            if (d - prev).days <= gap_close_days + 1:
                prev = d; continue
            runs.append((run_start, prev)); run_start = d; prev = d
        runs.append((run_start, prev))
        spans = []
        for s,e in runs:
            s2 = s - dt.timedelta(days=pad_days)
            e2 = e + dt.timedelta(days=pad_days)
            spans.append((s2, e2, batch))
        occ.setdefault(tank, []).extend(spans)
    return occ  # tank -> [(start,end,batch), ...]

def is_occupied(occ, tank: str, day: dt.date) -> Tuple[bool, Optional[str]]:
    for (s,e,b) in occ.get(tank, []):
        if s <= day <= e:
            return True, b
    return False, None

# --------------------------- Labeling (target) --------------------------------

def label_from_future(thresholds: Dict[str, Any],
                      tank: str,
                      after_ts_utc: dt.datetime,
                      hours_ahead: int) -> int:
    """
    Binary target: 1 if any parameter is red (High/Low via classify) at any time
    within the 2h window ending at (after_ts_utc + hours_ahead); else 0.
    Implementation reuses feature window builder centered at end time.
    """
    if after_ts_utc is None:
        return 0
    end = after_ts_utc + dt.timedelta(hours=hours_ahead)
    feats, _ = build_sensor_feats_for_window(tank, thresholds, end=end, hours_back=2)
    red_any = 0
    for k, v in feats.items():
        if k.endswith("__flag_red_2h") and float(v) >= 1.0:
            red_any = 1
            break
    return red_any

# --------------------------- Preloader ----------------------------------------

def preload_series_for_span(tanks: List[str], start_iso: str, end_iso: str):
    global SERIES_CACHE
    SERIES_CACHE.clear()
    print(f"[preload] fetching series for {len(tanks)} tanks over {start_iso} → {end_iso}", flush=True)
    t0 = time.time()
    total_nonempty = 0
    for ti, tank in enumerate(tanks, 1):
        conf = TANKS_AND_NURSERIES.get(tank) or {}
        nonempty = 0
        for label, (measurement, ip) in conf.items():
            dts, vs = fetch_series_influx(ip, measurement, label, start_iso, end_iso)
            SERIES_CACHE[(tank, label)] = (dts, vs)
            if dts:
                nonempty += 1
        total = len(conf)
        total_nonempty += nonempty
        print(f"[preload] {ti}/{len(tanks)} {tank} done ({nonempty}/{total} series with data)", flush=True)
    print(f"[preload] total cached series: {len(SERIES_CACHE)} (non-empty={total_nonempty}) in {time.time()-t0:.1f}s", flush=True)

# --------------------------- Row builder --------------------------------------

def safe_float(x) -> float:
    try:
        s = str(x).strip().replace("%","").replace(",", ".")
        return float(s)
    except Exception:
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else np.nan

def build_row(tank: str,
              day_local: dt.date,
              anchor_hour_local: int,
              tz: str,
              thresholds: Dict[str, Any],
              sensor_hours_back: int,
              obs_tokens: List[str],
              combos_spec: List[Tuple[List[str], float]],
              actions_notes: Dict[str, str]) -> Dict[str, Any]:
    # local anchor → UTC
    tzinfo = pd.Timestamp(day_local.isoformat()).tz_localize(tz).tzinfo
    end_local = dt.datetime.combine(day_local, dt.time(hour=anchor_hour_local)).replace(tzinfo=tzinfo)
    end = end_local.astimezone(dt.timezone.utc)

    # sensor features
    feats, latest_raw = build_sensor_feats_for_window(
        tank, thresholds, end=end, hours_back=sensor_hours_back
    )
    row = {"tank": tank, "day": day_local.isoformat(), "anchor_ts_utc": iso_z(end)}
    for k, v in feats.items():
        row[k] = v

    # combos
    combo_sum = 0.0
    combo_hits = 0
    stoks = set(obs_tokens)
    for toks, corr in combos_spec:
        if all(t in stoks for t in toks):
            combo_hits += 1
            combo_sum += corr
            row["combo::" + " & ".join(toks)] = 1.0
    row["combo_hit_any"] = 1.0 if combo_hits > 0 else 0.0
    row["combo_hit_count"] = float(combo_hits)
    row["combo_corr_reco"] = float(combo_sum)

    # obs tokens → one-hots
    for t in obs_tokens:
        row["tok__" + t] = 1.0

    # advice notes aggregation
    matched_adv = [actions_notes[t] for t in obs_tokens if t in actions_notes and actions_notes[t]]
    row["advice_notes"] = "; ".join(matched_adv) if matched_adv else ""

    return row

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

    ap.add_argument("--occupancy-mode", choices=["all","active_only","weighted"], default="active_only")
    ap.add_argument("--occ-gap-close-days", type=int, default=2)
    ap.add_argument("--occ-pad-days", type=int, default=0)
    ap.add_argument("--batch-calendar-csv", default="")  # optional external calendar (Tank,Batch,Start,End)

    ap.add_argument("--out-model", default="rf_model.joblib")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--subset-tanks", default="")
    ap.add_argument("--max-rows", type=int, default=0)

    args = ap.parse_args()

    if not (INFLUX_URL and INFLUX_DB):
        print("[error] set INFLUX_URL and INFLUX_DB (and optionally INFLUX_USER/INFLUX_PASS/INFLUX_RP/TAG_KEY)", file=sys.stderr)
        sys.exit(2)

    tstart = time.time()
    print("[start] RFCtrain2", flush=True)
    print(f"[cfg] hours_back={args.sensor_hours_back}, lookahead={args.label_hours_ahead}h, "
          f"anchor={args.anchor_hour_local} {args.tz}, occupancy_mode={args.occupancy_mode}", flush=True)

    load_tank_map(args.map_yaml)
    thresholds = load_thresholds(args.thresholds_yaml)
    if args.verbose:
        print(f"[maps] tanks loaded: {len(TANKS_AND_NURSERIES)}", flush=True)

    t0 = time.time()
    combos_spec = load_combos(args.combos_csv)
    t1 = time.time()
    actions_notes = load_actions(args.actions_csv)
    t2 = time.time()
    print(f"[load] maps/thresholds/combos/actions ready "
          f"({t1-t0:.1f}s combos, {t2-t1:.1f}s actions)", flush=True)
    print(f"[info] combos: {len(combos_spec)} (ex: {combos_spec[:3]})", flush=True)
    print(f"[info] actions: {len(actions_notes)}", flush=True)

    # subset tanks
    limit_tanks = []
    if args.subset_tanks.strip():
        limit_tanks = [s.strip() for s in args.subset_tanks.split(",") if s.strip()]
        print(f"[limit] subset tanks: {limit_tanks}", flush=True)

    paths = sorted(glob.glob(os.path.join(args.output_data_dir, "*.csv")))
    if not paths:
        print("[error] no CSVs in", args.output_data_dir, file=sys.stderr); sys.exit(3)
    print(f"[data] building dataset from {len(paths)} CSVs …", flush=True)

    # --- Occupancy calendar ---
    occ = None
    if args.occupancy_mode != "all":
        if args.batch_calendar_csv and os.path.exists(args.batch_calendar_csv):
            # Load explicit calendar
            cal = pd.read_csv(args.batch_calendar_csv)
            occ = {}
            for _, r in cal.iterrows():
                tank = str(r.get("Tank","")).strip()
                batch = str(r.get("Batch","")).strip() or None
                try:
                    s = pd.to_datetime(r.get("Start")).date()
                    e = pd.to_datetime(r.get("End")).date()
                except Exception:
                    continue
                occ.setdefault(tank, []).append((s, e, batch))
            print(f"[occ] loaded explicit calendar for {len(occ)} tanks", flush=True)
        else:
            occ = build_occupancy_from_csvs(paths, args.occ_gap_close_days, args.occ_pad_days)
            print(f"[occ] inferred calendar for {len(occ)} tanks", flush=True)

    # --- Discover global date span for preloading ---
    all_days = []
    for p in paths:
        try:
            df_dates = pd.read_csv(p, usecols=lambda c: str(c).strip().lower() in ("date","datum","day"))
        except Exception:
            df_dates = pd.read_csv(p)
        dcol = None
        for c in df_dates.columns:
            if str(c).strip().lower() in ("date","datum","day"):
                dcol = c; break
        if dcol is None:
            continue
        dd = pd.to_datetime(df_dates[dcol], dayfirst=True, errors="coerce").dropna().dt.date.tolist()
        all_days.extend(dd)

    if not all_days:
        print("[error] no dates found in CSVs", file=sys.stderr); sys.exit(4)

    min_day = min(all_days)
    max_day = max(all_days)

    # Convert local days to UTC span
    tanks_for_preload = list(TANKS_AND_NURSERIES.keys())
    if limit_tanks:
        tanks_for_preload = [t for t in tanks_for_preload if t in limit_tanks]
    if not tanks_for_preload:
        print("[error] no tanks matched for preload", file=sys.stderr); sys.exit(5)

    if args.occupancy_mode != "all" and occ:
        spans = []
        for t in tanks_for_preload:
            for (s,e,_b) in occ.get(t, []):
                spans.append((s,e))
        if not spans:
            print("[error] no occupancy spans found for selected tanks.", file=sys.stderr); sys.exit(7)
        base_start = min(s for s,_ in spans)
        base_end   = max(e for _,e in spans)
    else:
        base_start = min_day
        base_end   = max_day

    tzinfo = pd.Timestamp(base_start.isoformat()).tz_localize(args.tz).tzinfo
    global_start_local = dt.datetime.combine(base_start, dt.time(hour=max(args.anchor_hour_local - args.sensor_hours_back, 0))).replace(tzinfo=tzinfo)
    global_end_local   = dt.datetime.combine(base_end,   dt.time(hour=min(args.anchor_hour_local + args.label_hours_ahead, 23))).replace(tzinfo=tzinfo)
    global_start_utc = global_start_local.astimezone(dt.timezone.utc)
    global_end_utc   = global_end_local.astimezone(dt.timezone.utc)
    start_iso, end_iso = iso_z(global_start_utc), iso_z(global_end_utc)

    preload_series_for_span(tanks_for_preload, start_iso, end_iso)

    # --- Build rows ---
    rows = []
    built = 0
    skipped_no_tank = 0
    skipped_cov = 0
    skipped_unocc = 0

    for pi, p in enumerate(paths, 1):
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] cannot read {p}: {e}", file=sys.stderr)
            continue

        fname = os.path.basename(p)
        m = re.search(r'(N[A-F]|T\d)B(\d+)', fname, re.IGNORECASE)
        tank_name = None
        if m:
            code = m.group(1).upper()
            if code.startswith("N"):
                tank_name = {
                    "NA":"NurseryA","NB":"NurseryB","NC":"NurseryC",
                    "ND":"NurseryD","NE":"NurseryE","NF":"NurseryF"
                }.get(code[:2], None)
            else:
                tank_name = f"Tank{code[1:]}"
        if "Tank" in df.columns and not tank_name:
            tank_name = str(df["Tank"].iloc[0]).strip()

        if not tank_name or tank_name not in TANKS_AND_NURSERIES:
            skipped_no_tank += 1
            continue
        if limit_tanks and tank_name not in limit_tanks:
            continue

        dcol = None
        for c in df.columns:
            if str(c).strip().lower() in ("date","datum","day"):
                dcol = c; break
        if not dcol:
            continue

        for _, r in df.iterrows():
            raw_date = r.get(dcol)
            if not pd.notna(raw_date):
                continue
            try:
                d_local = pd.to_datetime(str(raw_date), dayfirst=True).date()
            except Exception:
                continue

            occupied = True
            batch_id = None
            if args.occupancy_mode != "all" and occ:
                occupied, batch_id = is_occupied(occ, tank_name, d_local)
                if args.occupancy_mode == "active_only" and not occupied:
                    skipped_unocc += 1
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

            # sensor coverage guard
            if row.get("_coverage_6h_ok", 0) < 1:
                skipped_cov += 1
                continue

            # add numeric obs features (optional but useful)
            row["Correction_pct"] = safe_float(obs.get("Correction %"))
            row["AdjustedFeed_g"] = safe_float(obs.get("Adjusted Feed Amount (g)"))

            anchor_ts = parse_iso_any(row["anchor_ts_utc"])
            ylab = label_from_future(thresholds, tank_name, anchor_ts, args.label_hours_ahead)
            row["label_red_6h"] = int(ylab)
            row["tank_group"] = tank_name
            row["occupied"] = 1 if occupied else 0
            if batch_id:
                row["batch_id"] = batch_id

            rows.append(row)
            built += 1
            if built % 200 == 0:
                print(f"[build] rows={built}  last_file={fname}", flush=True)
            if args.max_rows and built >= args.max_rows:
                print(f"[build] reached max-rows={args.max_rows}, stopping row build.", flush=True)
                break
        if args.max_rows and built >= args.max_rows:
            break

    if not rows:
        print("[error] no rows built; check inputs / occupancy / coverage filters", file=sys.stderr)
        sys.exit(6)

    print(f"[build] done. rows={len(rows)}  skipped_no_tank={skipped_no_tank}  "
          f"skipped_unoccupied={skipped_unocc}  skipped_low_coverage={skipped_cov}", flush=True)

    df = pd.DataFrame(rows)
    y = df["label_red_6h"].values.astype(int)
    groups = df["tank_group"].values

    drop_cols = ["tank","day","anchor_ts_utc","advice_notes","label_red_6h","tank_group"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    print(f"[shape] X={X.shape} y={y.shape} numeric_features={len(X.columns)}", flush=True)

    # diagnostics: prevalence
    print(f"[label] global positives: {y.sum()}/{len(y)} = {y.mean():.3%}", flush=True)
    for t in np.unique(groups):
        yt = y[groups == t]
        print(f"[label] {t}: positives {yt.sum()}/{len(yt)} = {yt.mean():.3%}", flush=True)

    # ---- Cross-validation ----
    # Try StratifiedGroupKFold if available
    splitter = None
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        n_splits = min(5, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y, groups)
        print("[cv] using StratifiedGroupKFold", flush=True)
    except Exception:
        n_splits = min(5, len(np.unique(groups))) if len(np.unique(groups)) >= 2 else 2
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
        print("[cv] using GroupKFold", flush=True)

    def has_two_classes(y_):
        return np.unique(y_).size >= 2

    aucs = []; praucs = []; fold = 0
    for tr, te in splitter:
        yt, yv = y[tr], y[te]
        if not (has_two_classes(yt) and has_two_classes(yv)):
            ct_train = np.bincount(yt, minlength=2)
            ct_val   = np.bincount(yv, minlength=2)
            print(f"[cv] skip fold (one-class) train={ct_train.tolist()} val={ct_val.tolist()}", flush=True)
            continue

        Xt, Xv = X.iloc[tr], X.iloc[te]
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(
                n_estimators=400, max_depth=None, n_jobs=-1,
                class_weight="balanced_subsample", random_state=42
            ))
        ])
        clf = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
        tfit = time.time()
        clf.fit(Xt, yt)
        p = clf.predict_proba(Xv)[:,1]
        auc = roc_auc_score(yv, p); pr  = average_precision_score(yv, p)
        aucs.append(auc); praucs.append(pr)
        fold += 1
        print(f"[cv] fold {fold}  AUC={auc:.3f}  PR-AUC={pr:.3f}  (fit {time.time()-tfit:.1f}s, n={len(yv)})", flush=True)

    if aucs:
        print(f"[cv] mean AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}   "
              f"mean PR-AUC={np.mean(praucs):.3f} ± {np.std(praucs):.3f}   (n={len(y)})", flush=True)
    else:
        print("[cv] no valid folds (all one-class). Consider adjusting label window or occupancy.", flush=True)

    # ---- Final fit on all data ----
    print("[fit] training final model …", flush=True)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=600, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        ))
    ])

    if np.unique(y).size >= 2:
        clf = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    else:
        print("[fit] dataset has a single class → using uncalibrated RF pipeline", flush=True)
        clf = pipe

    tfit = time.time()
    clf.fit(X, y)
    print(f"[fit] done in {time.time()-tfit:.1f}s", flush=True)

    out = {
        "model": clf,
        "features": list(X.columns),
        "thresholds": thresholds,
        "built_at": dt.datetime.utcnow().isoformat()+"Z",
        "sensor_hours_back": args.sensor_hours_back,
        "label_hours_ahead": args.label_hours_ahead,
        "anchor_hour_local": args.anchor_hour_local,
        "tz": args.tz,
        "occupancy_mode": args.occupancy_mode,
        "min_samples_6h": MIN_SAMPLES_6H,
    }
    joblib.dump(out, args.out_model)
    if UNMAPPED_OBS:
        print(f"[warn] Unmapped observation phrases ({len(UNMAPPED_OBS)}):")
        for s in sorted(UNMAPPED_OBS):
            print("   -", s)
    print(f"[ok] saved model: {args.out_model}", flush=True)
    print(f"[done] total time {time.time()-tstart:.1f}s", flush=True)

if __name__ == "__main__":
    main()
