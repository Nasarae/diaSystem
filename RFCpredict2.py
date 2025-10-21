#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, datetime as dt
import numpy as np
import pandas as pd
import joblib

# IMPORTANT: import training code as a module
import RFCtrain2 as rfc

# ---------- Local helpers (predict-time only) ----------

_TRANSLATE_DE_EN = {
  "50%": "50%",
  "80%": "80%",

  # Core observation phrases (DE → EN)
  "50% Shrimps haben vollen Verdaungstrakt": "50% of shrimps have a full digestive tract",
  "80% Shrimps haben vollen Verdaungstrakt": "80% of shrimps have a full digestive tract",
  "Leicht Gefüllt": "Slightly filled",
  "Leicht Gefüllt bis leer": "Slightly filled to empty",
  "Shrimps merheitlich Transluzent": "Shrimps mostly translucent",
  "Shrimps gestresst/ weisslich, Darm kaum ernkennbar": "Shrimps stressed/whitish, intestine barely visible",
  "Shrimps haben merheitlich weiche Panzer": "Shrimps mostly have soft shells",
  "Keine Futterreste": "No feed leftovers",
  "Wenig Futterreste": "Few feed leftovers",
  #"Viele Futterreste!!!": "A lot of leftover feed!",
  "Viele Futterreste": "A lot of leftover feed!",
  "kein Futter": "No feed leftover",
  "wenig Futter": "Little feed leftover",
  "keine Schalen": "No shells",
  "wenige Schalen": "Few shells",
  "viele Schalen": "Many shells",
  "sehr viele Schalen": "Very many shells",
  "Keine Tote": "No dead shrimp",
  "Wenig Tote": "Few dead",
  "Wenig tote": "Few dead",
  "Viele Tote": "Many dead",
  "Normale Schaumbildung": "Normal foam formation",
  "Wenig starke Schaumbildung": "Slight foaming",
  "Starke Schaumbildung": "Strong foaming",
  "Tote haben harte Schale": "Dead shrimp have hard shells",
  "Tote haben weiche Schale": "Dead shrimp have soft shells",
  "Tote sind merheitlich Grau": "Dead are mostly gray",
  "Tote sind merheitlich Orange": "Dead are mostly orange",
  "Tote, Orang und Grau gemischt": "Dead (orange and gray mixed)",
  "Gut": "Good fitness",

  # Advice lines (DE → EN)
  "Abschäumer einstellen Zuckerreduzieren": "Adjust skimmer; reduce sugar",
  "Ammonium, O2, CO2 prüfen": "Check ammonium, O₂, CO₂",
  "Probleme mit der Wasserqualität": "Problems with water quality",
  "Prüfen von pH, Alkalinität, Calzium, Magnesium, Kalium": "Check pH, alkalinity, calcium, magnesium, potassium",
  "Res&Alex Kontaktieren": "Contact Res & Alex",
  "Res&Alex kontaktieren": "Contact Res & Alex",
  "Res&Alex kontaktieren, eintretender Mortalitätspeak": "Contact Res & Alex, impending mortality peak",
  "Schälzyklus": "Molting cycle",
  "Turbidität prüfen bei gelegenheit": "Check turbidity when possible",
  "Verdaungstrakt genauer prüfen": "Check digestive tract more closely",
  "Vermerken": "Make a note",
  "Mortalitätspeak": "Mortality peak",
  "Vor Mortalitätspeak": "Before mortality peak",
  "Mortalitätspeak überschritten": "Mortality peak passed",

  # Fallbacks
  "...": "..."
}

def maybe_translate(s: str, lang: str) -> str:
    """Translate a semicolon/pipe-aggregated German string to English if needed."""
    if lang == "de" or not s:
        return s
    parts = [p.strip() for p in str(s).replace(";", " | ").split(" | ") if p.strip()]
    out = []
    for p in parts:
        out.append(_TRANSLATE_DE_EN.get(p, p))
    return " | ".join(out)

def translate_token_list(tokens, lang: str):
    if lang == "de":
        return tokens
    return [_TRANSLATE_DE_EN.get(t, t) for t in tokens]

def display_state(state: str, lang: str) -> str:
    """Human-friendly state names."""
    map_en = {
        "BLow": "Borderline low",
        "BHigh": "Borderline high",
        "Low": "Low",
        "High": "High",
        "Range": "In range",
        "Unknown": "Unknown",
    }
    map_de = {
        "BLow": "Grenzwertig niedrig",
        "BHigh": "Grenzwertig hoch",
        "Low": "Niedrig",
        "High": "Hoch",
        "Range": "Im Zielbereich",
        "Unknown": "Unbekannt",
    }
    table = map_en if lang == "en" else map_de
    return table.get(state, state)

def multi_param_advice(thresholds: dict, latest_raw: dict, lang: str) -> list:
    """Return extra advice lines based on combos of parameter states, localized."""
    def st(name):
        v = latest_raw.get(name) if name in latest_raw else latest_raw.get(f"{name}__last", latest_raw.get(name))
        try:
            return rfc.classify(thresholds, name, float(v) if v is not None else np.nan)
        except Exception:
            return "Unknown"

    S = {k: st(k) for k in ["Temperature","PH","DO mg/L","CO2 concentration","CO2_water","Salinity","Turbidity"]}

    adv_en = []
    adv_de = []

    # DO low & CO2_water high
    if S["DO mg/L"] in ("Low","BLow") and S["CO2_water"] in ("High","BHigh"):
        adv_en.append("Low DO with high CO₂ (water) → check aeration/circulation and reduce CO₂ source.")
        adv_de.append("Niedriger DO bei hohem CO₂ (Wasser) → Belüftung/Zirkulation prüfen und CO₂-Quelle reduzieren.")

    # High pH & high CO2_water
    if S["PH"] in ("High","BHigh") and S["CO2_water"] in ("High","BHigh"):
        adv_en.append("Elevated pH with high CO₂ (water) → review buffering/alkalinity and CO₂ dosing.")
        adv_de.append("Erhöhter pH bei hohem CO₂ (Wasser) → Puffer/Alkalinität und CO₂-Dosierung überprüfen.")

    # Turbidity very high
    if S["Turbidity"] == "High":
        adv_en.append("High turbidity → check filtration, biofouling/inputs; reduce leftover feed.")
        adv_de.append("Hohe Trübung → Filtration, Biofouling/Eintrag prüfen; Futterreste reduzieren.")

    # Temperature out of range
    if S["Temperature"] in ("Low","High"):
        adv_en.append("Temperature out of range → verify heating/cooling and sensor drift.")
        adv_de.append("Temperatur außerhalb Zielbereich → Heizung/Kühlung und Sensor-Drift prüfen.")

    # Salinity out of range
    if S["Salinity"] in ("Low","High"):
        adv_en.append("Salinity out of band → adjust gradually; check probe calibration.")
        adv_de.append("Salinität außerhalb Band → schrittweise anpassen; Messsonde kalibrieren.")

    # DO low + Temperature high
    if S["DO mg/L"] == "Low" and S["Temperature"] == "High":
        adv_en.append("Low DO at high temperature → oxygen demand ↑; consider extra aeration.")
        adv_de.append("Niedriger DO bei hoher Temperatur → Sauerstoffbedarf ↑; zusätzliche Belüftung erwägen.")

    # De-dup
    adv_en = list(dict.fromkeys(adv_en))
    adv_de = list(dict.fromkeys(adv_de))
    return adv_en if lang == "en" else adv_de

def safe_float(x) -> float:
    try:
        s = str(x).strip().replace("%","").replace(",", ".")
        return float(s)
    except Exception:
        v = pd.to_numeric(x, errors="coerce")
        return float(v) if pd.notna(v) else np.nan

def parse_when_local(s: str, tz: str) -> dt.datetime:
    if str(s).lower() in ("now", "today"):
        return pd.Timestamp("now", tz=tz).to_pydatetime()
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts.to_pydatetime()

def obs_from_args(args) -> dict:
    obs = {
        "Digestive Tract": args.digestive_tract or "",
        "Feed Leftovers":  args.feed_leftovers or "",
        "Dead":            args.dead or "",
        "Foam Formation":  args.foam or "",
        "Shrimp Fitness":  args.fitness or "",
        "Shells":          args.shells or "",
        "Sugar yes/no":    args.sugar_yesno or "",
        "Buffer yes/no":   args.buffer_yesno or "",
        "Vitamin C Amount": args.vitamin_c_amount if args.vitamin_c_amount is not None else 0,
        "Bactosafe Amount": args.bactosafe_amount if args.bactosafe_amount is not None else 0,
        "Adjusted Feed Amount (g)": args.adjusted_feed_g if args.adjusted_feed_g is not None else np.nan,
        "Correction %": args.correction_pct if args.correction_pct is not None else np.nan,
        "Sugar Amount": args.sugar_amount,
        "Buffer Amount": args.buffer_amount,
    }
    for k in ["Sugar yes/no", "Buffer yes/no"]:
        v = str(obs.get(k) or "").strip().lower()
        if v in ("y","yes","ja"):   obs[k] = "ja"
        elif v in ("n","no","nein"): obs[k] = "nein"
    return obs

def build_prediction_row(tank: str,
                         when_local: dt.datetime,
                         tz: str,
                         thresholds: dict,
                         sensor_hours_back: int,
                         obs_tokens: list,
                         combos_spec: list,
                         actions_notes: dict):
    end_utc = pd.Timestamp(when_local).tz_convert("UTC").to_pydatetime()
    feats, latest_raw = rfc.build_sensor_feats_for_window(
        tank, thresholds, end=end_utc, hours_back=sensor_hours_back
    )
    row = {
        "tank": tank,
        "day": when_local.date().isoformat(),
        "anchor_ts_utc": rfc.iso_z(end_utc),
        **feats,
    }
    stoks = set(obs_tokens); combo_sum = 0.0; combo_hits = 0
    for toks, corr in combos_spec:
        if all(t in stoks for t in toks):
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
    return row, latest_raw

def pretty_param_diagnostics(thresholds: dict, latest_raw: dict, lang: str) -> pd.DataFrame:
    rows = []
    for k, v in latest_raw.items():
        key = k.replace("__last", "")
        try:
            state = rfc.classify(thresholds, key, v)
        except Exception:
            state = "Unknown"
        rows.append({"parameter": key, "value": v, "state": display_state(state, lang)})
    return pd.DataFrame(rows).sort_values("parameter") if rows else pd.DataFrame(columns=["parameter","value","state"])

def align_to_features(row: dict, features: list) -> pd.DataFrame:
    X1 = pd.DataFrame([{k: pd.to_numeric(v, errors="coerce") for k, v in row.items()}])
    drop_cols = ["tank","day","anchor_ts_utc","advice_notes","label_red_6h","tank_group","occupied","batch_id"]
    X1 = X1.drop(columns=[c for c in drop_cols if c in X1.columns], errors="ignore")
    for f in features:
        if f not in X1.columns:
            X1[f] = np.nan
    X1 = X1[features]
    return X1

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--map-yaml", required=True)
    ap.add_argument("--thresholds-yaml", required=True)
    ap.add_argument("--combos-csv", required=True)
    ap.add_argument("--actions-csv", required=True)
    ap.add_argument("--tank", required=True)
    ap.add_argument("--tz", default="Europe/Zurich")
    ap.add_argument("--when-local", default="now", help='e.g. "now" or "2025-03-05 12:00"')
    ap.add_argument("--sensor-hours-back", type=int, default=24)
    # observations
    ap.add_argument("--digestive-tract", default="")
    ap.add_argument("--feed-leftovers", default="")
    ap.add_argument("--dead", default="")
    ap.add_argument("--foam", default="")
    ap.add_argument("--fitness", default="")
    ap.add_argument("--shells", default="")
    ap.add_argument("--sugar-yesno", default="")
    ap.add_argument("--buffer-yesno", default="")
    ap.add_argument("--vitamin-c-amount", type=float)
    ap.add_argument("--bactosafe-amount", type=float)
    ap.add_argument("--adjusted-feed-g", type=float)
    ap.add_argument("--correction-pct", type=float)
    ap.add_argument("--sugar-amount", type=float)
    ap.add_argument("--buffer-amount", type=float)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--lang", choices=["de", "en"], default="en")  # default to English
    args = ap.parse_args()

    # Load config artifacts
    rfc.load_tank_map(args.map_yaml)
    thresholds = rfc.load_thresholds(args.thresholds_yaml)
    combos     = rfc.load_combos(args.combos_csv)
    actions    = rfc.load_actions(args.actions_csv)

    print(f"[map] tanks loaded: {len(rfc.TANKS_AND_NURSERIES)}")
    print(f"[map] sample keys: {list(rfc.TANKS_AND_NURSERIES.keys())[:8]}")

    if args.tank not in rfc.TANKS_AND_NURSERIES:
        print(f"[error] tank {args.tank} not found in map. "
              f"Available: {list(rfc.TANKS_AND_NURSERIES)[:8]}...", file=sys.stderr)
        sys.exit(2)

    # Load model bundle
    bundle   = joblib.load(args.model)
    clf      = bundle["model"]
    features = bundle["features"]

    # Build observations → tokens
    obs  = obs_from_args(args)
    toks = rfc.tokens_from_obs(obs)

    # Build prediction row anchored at --when-local
    when_local = parse_when_local(args.when_local, args.tz)
    row, latest_raw = build_prediction_row(
        tank=args.tank,
        when_local=when_local,
        tz=args.tz,
        thresholds=thresholds,
        sensor_hours_back=args.sensor_hours_back,
        obs_tokens=toks,
        combos_spec=combos,
        actions_notes=actions,
    )

    # coverage warning like training
    if row.get("_coverage_6h_ok", 0) < 1:
        print("[warn] Insufficient sensor coverage in the last 6h for a confident diagnostic.", file=sys.stderr)

    # numeric obs fields (names consistent with training)
    row["Correction_pct"] = safe_float(obs.get("Correction %"))
    row["AdjustedFeed_g"] = safe_float(obs.get("Adjusted Feed Amount (g)"))

    # Align columns and predict
    X1    = align_to_features(row, features)
    proba = clf.predict_proba(X1)[:, 1][0]

    # Pretty diagnostics (localized states)
    diag_df = pretty_param_diagnostics(thresholds, latest_raw, args.lang)

    # Console output (localized headings)
    print("\n=== LuckyShrimp Diagnostic ===")
    print(f"Tank: {args.tank}")
    print(f"When (local {args.tz}): {when_local.isoformat()}")
    print(f"Anchor (UTC): {row['anchor_ts_utc']}")
    print(f"Sensor lookback: {args.sensor_hours_back}h")
    print(f"{'Risk P(red within +6h):' if args.lang=='en' else 'Risiko P(rot innerhalb +6h):'} {proba:.3f}")
    print(f"{'Coverage OK (6h):' if args.lang=='en' else 'Abdeckung OK (6h):'} {'YES' if row.get('_coverage_6h_ok',0)==1 else 'NO'}")

    if not diag_df.empty:
        print("\nParameter diagnostics (latest values):" if args.lang == "en" else "\nParameterdiagnostik (aktuelle Werte):")
        w_param = max(9, diag_df["parameter"].map(len).max())
        print(f"{'parameter'.ljust(w_param)}  {'value'.rjust(10)}  {'state' if args.lang=='en' else 'Zustand'}")
        for _, r in diag_df.iterrows():
            pv = f"{r['value']:.3f}" if isinstance(r['value'], (int,float,np.floating)) else str(r['value'])
            print(f"{str(r['parameter']).ljust(w_param)}  {pv.rjust(10)}  {r['state']}")

    if toks:
        shown = translate_token_list(toks, args.lang)
        print("\nObservation tokens:" if args.lang == "en" else "\nBeobachtungstokens:")
        for t in shown:
            print(f" - {t}")

    if row.get("combo_hit_any", 0) >= 1:
        print("\nMatched observation combos:" if args.lang == "en" else "\nPassende Beobachtungskombinationen:")
        for k, v in row.items():
            if k.startswith("combo::") and float(v) == 1.0:
                # keep raw combo key; or translate parts if you prefer
                print(f" - {k[7:]}")
        line = f"→ Recommended feed correction (sum): {row['combo_corr_reco']:+.1f}%"
        print(line if args.lang == "en" else line.replace("Recommended feed correction", "Empfohlene Futterkorrektur"))
    else:
        print("\nNo observation combo matched; recommendation stays at 0%." if args.lang=="en"
              else "\nKeine Beobachtungskombination erkannt; Empfehlung bleibt bei 0%.")

    if isinstance(row.get("combo_corr_reco"), (int,float)) and not np.isnan(row.get("combo_corr_reco")):
        line = f"Combo-based correction recommendation (sum): {row['combo_corr_reco']:+.1f}%"
        print(line if args.lang == "en" else line.replace("Combo-based correction recommendation", "Kombinationsbasierte Korrekturempfehlung"))

    if row.get("advice_notes"):
        notes = maybe_translate(row["advice_notes"], args.lang)
        print("\nAdvice notes:" if args.lang == "en" else "\nHinweise:")
        for part in str(notes).split(" | "):
            print(f" - {part}")

    extra = multi_param_advice(thresholds, latest_raw, args.lang)
    if extra:
        print("\nSystem notes:" if args.lang == "en" else "\nSystemische Hinweise:")
        for line in extra:
            print(f" - {line}")

    # Optional JSON (always English keys; values are whatever lang is printed)
    if args.out_json:
        out = {
            "tank": args.tank,
            "when_local": when_local.isoformat(),
            "anchor_ts_utc": row["anchor_ts_utc"],
            "probability_red_plus_6h": float(proba),
            "coverage_ok_6h": bool(row.get("_coverage_6h_ok",0) == 1),
            "latest_raw": latest_raw,
            "parameter_diagnostics": diag_df.to_dict(orient="records"),
            "observation_tokens": translate_token_list(toks, args.lang),
            "combo_corr_reco": row.get("combo_corr_reco"),
            "advice_notes": maybe_translate(row.get("advice_notes",""), args.lang),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n[wrote] {args.out_json}")

if __name__ == "__main__":
    main()
