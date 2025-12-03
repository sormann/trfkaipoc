#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_mindre_streng_holdningsklasse.py

To-stegs modell (Steg 1: Avslag vs Tillatelse, Steg 2: tillatelsestyper)
med 'confidence-gating' OG en regel: auto-beslutninger tillates KUN når
'Avkjørsel, holdningsklasse' ∈ {'Lite streng','Mindre streng'}.

VIKTIG: Denne versjonen beregner og logger alltid metrikker etter at
holdnings-gate er brukt (dette gjelder både utskrift og rapport).

Kjør:
    python predict_mindre_streng_holdningsklasse.py \
        --data data_2022-2025.csv \
        --target-precision 0.7 \
        --export-auto-csv

Artefakter:
    - egsv_stage1_avslag_vs_tillatelse_holdningsklasse.pkl
    - egsv_stage2_tillatelsestyper_holdningsklasse.pkl
    - confidence_thresholds_holdningsklasse.json
    - confidence_policy_report_holdningsklasse.txt
    - auto_approved_holdningsklasse.csv

Eksponerer også:
    - load_two_stage_with_thresholds()
    - predict_with_manual_review()   # med holdnings-gating
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# Konstanter og konfig
# -----------------------------
TARGET_COL = "EGS.VEDTAK.10670"
FEATURE_COLS = [
    "ÅDT, total",
    "ÅDT, andel lange kjøretøy",
    "Fartsgrense",
    "Avkjørsler",
    "Trafikkulykker",
    "EGS.BRUKSOMRÅDE.1256",
    "LOK.KOMMUNE",
    "Avkjørsel, holdningsklasse",
    "Funksjonsklasse",
]

NUM_COLS_BASE = [
    "ÅDT, total",
    "ÅDT, andel lange kjøretøy",
    "Fartsgrense",
    "Avkjørsler",
    "Trafikkulykker",
    "Avkjørsler_innen_60m",
]

HOLDNINGS_COL = "Avkjørsel, holdningsklasse"
ALLOWED_HOLDNING = {"Lite streng", "Mindre streng"}  # <-- policy filter


# -----------------------------
# Lasting og rensing
# -----------------------------
def load_and_clean(path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df0 = pd.read_csv(path, sep=";", dtype=str)
    df0.columns = [c.strip() for c in df0.columns]
    for c in df0.columns:
        df0[c] = df0[c].astype(str).str.strip()

    available_features = [c for c in FEATURE_COLS if c in df0.columns]
    use_cols = [TARGET_COL] + available_features
    df = df0[use_cols].copy()

    # Behold bare rader med målverdi
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] != "")]

    # Numeriske
    num_cols = [c for c in available_features if c in NUM_COLS_BASE]
    for c in num_cols:
        df[c] = pd.to_numeric(
            df[c]
            .str.replace("\u00A0", "", regex=True)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # Kategorier: normaliser litt spacing/hyphen
    cat_cols = [c for c in available_features if c not in num_cols]
    for c in cat_cols:
        df[c] = df[c].str.replace(" +", " ", regex=True)
        df[c] = df[c].str.replace(" \- ", " - ", regex=True)

    # Fjern helt like duplikater på features+target
    if available_features:
        df = df.drop_duplicates(subset=available_features + [TARGET_COL])

    return df, available_features, num_cols


# -----------------------------
# Preprosessering
# -----------------------------
def build_preprocess(available_features: List[str], num_cols: List[str]) -> ColumnTransformer:
    numeric_features = [c for c in num_cols if c in available_features]
    categorical_features = [c for c in available_features if c not in numeric_features]

    num_pre = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # NB: hvis du har eldre sklearn, bytt 'sparse_output=False' til 'sparse=False'
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocess = ColumnTransformer(
        [("num", num_pre, numeric_features), ("cat", cat_pre, categorical_features)]
    )
    return preprocess


# -----------------------------
# Trening + terskler
# -----------------------------
def train_stage1_with_threshold(
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y1_train: pd.Series,
    target_precision: float,
    val_size: float = 0.2,
    random_state: int = 123,
) -> Tuple[Pipeline, Optional[float], Optional[int]]:
    # Indre validering
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y1_train, test_size=val_size, random_state=random_state, stratify=y1_train
    )
    stage1_val = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced_subsample")),
        ]
    )
    stage1_val.fit(X_tr, y_tr)

    proba_val = stage1_val.predict_proba(X_val)
    classes1 = list(stage1_val.named_steps["model"].classes_)
    idx_avslag = classes1.index("Avslag")
    p_avslag_val = proba_val[:, idx_avslag]
    pred_val = stage1_val.predict(X_val)
    mask_pred_avslag = (pred_val == "Avslag")

    best_t1, best_cov, best_prec = None, 0.0, 0.0
    true_bin = (y_val == "Avslag").astype(int).to_numpy()
    for t in np.unique(np.round(p_avslag_val[mask_pred_avslag], 6)):
        m = mask_pred_avslag & (p_avslag_val >= t)
        if m.sum() == 0:
            continue
        prec = true_bin[m].mean()
        cov = m.mean()
        if prec >= target_precision and cov > best_cov:
            best_t1, best_cov, best_prec = float(t), float(cov), float(prec)

    stage1 = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced_subsample")),
        ]
    )
    stage1.fit(X_train, y1_train)
    return stage1, best_t1, idx_avslag


def prepare_stage2_pools(
    X_all: pd.DataFrame, y_all: pd.Series, y_stage1: np.ndarray, idx_train: np.ndarray
) -> Tuple[pd.DataFrame, pd.Series]:
    X_all_r = X_all.reset_index(drop=True)
    y_all_r = y_all.reset_index(drop=True)
    y_stage1_r = pd.Series(y_stage1).reset_index(drop=True)
    mask_true_t = (y_stage1_r.iloc[idx_train] == "Tillatelse")
    X2_pool = X_all_r.iloc[idx_train].loc[mask_true_t]
    y2_pool = y_all_r.iloc[idx_train].loc[mask_true_t]
    return X2_pool, y2_pool


def train_stage2_with_thresholds(
    preprocess: ColumnTransformer,
    X2_pool: pd.DataFrame,
    y2_pool: pd.Series,
    target_precision: float,
    val_size: float = 0.2,
    random_state: int = 123,
) -> Tuple[Pipeline, List[str], Dict[str, Optional[float]]]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X2_pool, y2_pool, test_size=val_size, random_state=random_state, stratify=y2_pool
    )
    stage2_val = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=600, random_state=42, class_weight="balanced_subsample")),
        ]
    )
    stage2_val.fit(X_tr, y_tr)

    proba_val = stage2_val.predict_proba(X_val)
    classes2 = list(stage2_val.named_steps["model"].classes_)
    pred_val = stage2_val.predict(X_val)

    thresholds2: Dict[str, Optional[float]] = {}
    proba_df = pd.DataFrame(proba_val, columns=classes2, index=X_val.index)

    for cls in classes2:
        idx_pred_cls = proba_df.index[pred_val == cls]
        if len(idx_pred_cls) == 0:
            thresholds2[cls] = None
            continue
        p_cls = proba_df.loc[idx_pred_cls, cls].values
        y_true_cls = (y_val.loc[idx_pred_cls] == cls).astype(int).values
        best_t, best_cov = None, 0.0
        for t in np.unique(np.round(p_cls, 6)):
            m = p_cls >= t
            if m.sum() == 0:
                continue
            prec = y_true_cls[m].mean()
            cov = m.sum() / len(y_val)
            if prec >= target_precision and cov > best_cov:
                best_t, best_cov = float(t), float(cov)
        thresholds2[cls] = best_t

    stage2 = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=600, random_state=42, class_weight="balanced_subsample")),
        ]
    )
    stage2.fit(X2_pool, y2_pool)
    return stage2, classes2, thresholds2


# -----------------------------
# Policy (med holdnings-gating)
# -----------------------------
def normalize_holdningsverdi(s: str) -> str:
    """Liten normalisering for robust sammenligning."""
    if s is None:
        return ""
    s2 = str(s).strip()
    s2 = " ".join(s2.split())
    s2 = s2.replace(" \t", " ")
    return s2


def evaluate_policy_on_test(
    stage1: Pipeline,
    stage2: Pipeline,
    idx_avslag: int,
    classes2: List[str],
    thresholds2: Dict[str, Optional[float]],
    t1: Optional[float],
    X_test: pd.DataFrame,
    y_all: pd.Series,
    idx_test: np.ndarray,
) -> Tuple[int, float, float, pd.DataFrame, int]:
    """
    Policy:
      - Holdnings-gate: auto beslutninger KUN dersom HOLDNINGS_COL i ALLOWED_HOLDNING
      - Steg 1: hvis pred == 'Avslag' og p_avslag >= t1 -> auto Avslag
      - Steg 2: hvis pred == 'Tillatelse', topp-klasse k og p_top >= t2[k] -> auto k
      - Ellers: manual_review

    Returnerer:
      - auto_n: antall auto-besluttede (etter gate)
      - auto_acc: accuracy på auto-beslutninger (etter gate)
      - coverage: auto_n / len(test)
      - df_auto: DataFrame med KUN auto-besluttede rader (etter gate)
      - blocked_by_holdnings: antall test-rader som IKKE oppfylte holdningskravet
    """
    assert HOLDNINGS_COL in X_test.columns, (
        f"{HOLDNINGS_COL} mangler i features – må være med for policy."
    )

    proba1 = stage1.predict_proba(X_test)
    pred1 = stage1.predict(X_test)
    p_avslag = proba1[:, idx_avslag]

    N = len(X_test)
    auto_decision = np.array(["manual_review"] * N, dtype=object)
    decision_conf = np.array([np.nan] * N, dtype=float)

    # Holdnings‑gate (bool per rad)
    hold_ok = X_test[HOLDNINGS_COL].map(normalize_holdningsverdi).isin(ALLOWED_HOLDNING).to_numpy()
    blocked_by_holdnings = int((~hold_ok).sum())

    # Steg 1: Avslag
    if t1 is not None:
        mask_auto1 = (pred1 == "Avslag") & (p_avslag >= t1) & hold_ok
        auto_decision[mask_auto1] = "Avslag"
        decision_conf[mask_auto1] = p_avslag[mask_auto1]

    # Steg 2: Tillatelse
    need_stage2_pos = np.where((auto_decision == "manual_review") & (pred1 == "Tillatelse"))[0]
    if len(need_stage2_pos):
        sub = X_test.iloc[need_stage2_pos]
        proba2 = stage2.predict_proba(sub)
        classes2_arr = np.array(classes2)
        top_idx = proba2.argmax(axis=1)
        top_cls = classes2_arr[top_idx]
        top_p = proba2.max(axis=1)
        # policy + holdnings‑gate
        for j, pos in enumerate(need_stage2_pos):
            if not hold_ok[pos]:
                continue  # ikke i lovlig holdningsklasse => manual_review
            thr = thresholds2.get(top_cls[j])
            if thr is not None and top_p[j] >= thr:
                auto_decision[pos] = top_cls[j]
                decision_conf[pos] = float(top_p[j])

    final_true = y_all.reset_index(drop=True).iloc[idx_test].values
    auto_mask = (auto_decision != "manual_review")
    auto_n = int(auto_mask.sum())
    auto_acc = float((auto_decision[auto_mask] == final_true[auto_mask]).mean()) if auto_n > 0 else float("nan")
    coverage = auto_n / N

    # Bygg DataFrame for eksport
    df_test = pd.DataFrame(X_test).copy()
    df_test[TARGET_COL] = final_true
    df_test["stage1_pred"] = pred1
    df_test["p_avslag"] = p_avslag
    df_test["holdningsklasse_ok"] = hold_ok
    df_test["auto_decision"] = auto_decision
    df_test["auto_confidence"] = decision_conf

    df_auto = df_test[df_test["auto_decision"] != "manual_review"].copy()
    return auto_n, auto_acc, coverage, df_auto, blocked_by_holdnings
def save_artifacts(
    stage1: Pipeline,
    stage2: Pipeline,
    t1: Optional[float],
    thresholds2: Dict[str, Optional[float]],
    auto_n: int,
    auto_acc: float,
    coverage: float,
    target_precision: float,
    blocked_by_holdnings: int,
) -> None:
    """Lagrer modeller + terskler + policyrapport som bruker GATEDE tall."""
    import joblib

    joblib.dump(stage1, "egsv_stage1_avslag_vs_tillatelse_holdningsklasse.pkl")
    joblib.dump(stage2, "egsv_stage2_tillatelsestyper_holdningsklasse.pkl")

    Path("confidence_thresholds_holdningsklasse.json").write_text(
        json.dumps(
            {
                "target_precision": target_precision,
                "stage1": {"Avslag_threshold": t1},
                "stage2": {"class_thresholds": thresholds2},
                "holdnings_gate": {
                    "column": HOLDNINGS_COL,
                    "allowed_values": sorted(ALLOWED_HOLDNING),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # VIKTIG: rapporten bruker tallene ETTER gate (auto_n/coverage/auto_acc er allerede gatede)
    Path("confidence_policy_report_holdningsklasse.txt").write_text(
        (
            f"Target precision (auto): {target_precision}\n"
            f"Holdnings-gate: tillater kun {sorted(ALLOWED_HOLDNING)}\n"
            f"Auto-decisions (test): {auto_n} (coverage {coverage:.3f})\n"
            f"Accuracy på auto-beslutninger: {auto_acc:.3f}\n"
            f"Blokkert av holdnings-gate (test, alle rader): {blocked_by_holdnings}\n"
        ),
        encoding="utf-8",
    )


# -----------------------------
# Drift-funksjoner (med gate)
# -----------------------------
def load_two_stage_with_thresholds(
    m1_path: str = "egsv_stage1_avslag_vs_tillatelse_holdningsklasse.pkl",
    m2_path: str = "egsv_stage2_tillatelsestyper_holdningsklasse.pkl",
    thresholds_path: str = "confidence_thresholds_holdningsklasse.json",
):
    import joblib
    m1 = joblib.load(m1_path)
    m2 = joblib.load(m2_path)
    thr = json.load(open(thresholds_path, "r", encoding="utf-8"))
    return m1, m2, thr


def normalize_holdningsverdi_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def predict_with_manual_review(
    df_features: pd.DataFrame,
    m1: Pipeline,
    m2: Pipeline,
    thr: Dict,
) -> pd.DataFrame:
    """
    Prediksjon med holdnings-gate:
      - kun auto-beslutning hvis holdningsklasse i ALLOWED_HOLDNING
      - samme gating som i evaluate_policy_on_test
    """
    # Les policy fra terskel-fil (men bruk også de faste ALLOWED_HOLDNING)
    t1 = thr["stage1"]["Avslag_threshold"]
    class_thr = thr["stage2"]["class_thresholds"]

    # Sjekk at vi har kolonnen for policy-gate:
    assert HOLDNINGS_COL in df_features.columns, (
        f"{HOLDNINGS_COL} mangler i df_features (kreves for policy-gate)."
    )

    # Stage 1
    proba1 = m1.predict_proba(df_features)
    classes1 = list(m1.named_steps["model"].classes_)
    idx_avslag = classes1.index("Avslag")
    p_avslag = proba1[:, idx_avslag]
    pred1 = m1.predict(df_features)

    res = pd.DataFrame(index=df_features.index)
    res["pred_stage1"] = pred1
    res["p_avslag"] = p_avslag
    res["decision"] = "manual_review"

    # Holdnings-gate
    hold_ok = normalize_holdningsverdi_series(df_features[HOLDNINGS_COL]).isin(ALLOWED_HOLDNING)

    # Steg 1 auto
    if t1 is not None:
        mask = (pred1 == "Avslag") & (p_avslag >= t1) & hold_ok
        res.loc[mask, "decision"] = "Avslag"

    # Steg 2 auto
    idx_till = res[(res["decision"] == "manual_review") & (res["pred_stage1"] == "Tillatelse")].index
    if len(idx_till):
        sub = df_features.loc[idx_till]
        proba2 = m2.predict_proba(sub)
        classes2 = list(m2.named_steps["model"].classes_)
        top_idx = proba2.argmax(axis=1)
        top_cls = [classes2[i] for i in top_idx]
        top_p = proba2.max(axis=1)
        for j, idx in enumerate(idx_till):
            if not hold_ok.loc[idx]:
                continue  # ikke tillatt holdningsklasse => manual_review
            thr2 = class_thr.get(top_cls[j])
            if thr2 is not None and top_p[j] >= thr2:
                res.loc[idx, "decision"] = top_cls[j]
    return res


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="To-stegs modell med confidence-gating og holdnings-gate (kun Lite/Mindre streng auto)"
    )
    parser.add_argument("--data", type=str, default="data_2022-2025.csv", help="Sti til CSV-fil")
    parser.add_argument("--target-precision", type=float, default=0.7, help="Minste presisjon for auto-beslutning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Andel til test")
    parser.add_argument("--val-size", type=float, default=0.2, help="Andel til indre validering for terskler")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--export-auto-csv", action="store_true", default="auto_approved_holdningsklasse.csv", help="Eksporter auto_approved_holdningsklasse.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    target_precision = float(args.target_precision)
    test_size = float(args.test_size)
    val_size = float(args.val_size)
    rs = int(args.random_state)

    print(f"[INFO] Leser og renser data: {data_path}")
    df, available_features, num_cols = load_and_clean(data_path)
    X_all = df[available_features].copy()
    y_all = df[TARGET_COL].copy()
    y_stage1 = np.where(y_all == "Avslag", "Avslag", "Tillatelse")

    assert HOLDNINGS_COL in X_all.columns, (
        f"Kolonnen '{HOLDNINGS_COL}' må være en del av features for at policy-gate skal fungere."
    )

    print("[INFO] Tren/test-splitt (stratifisert)")
    X_train, X_test, y1_train, y1_test, idx_train, idx_test = train_test_split(
        X_all,
        y_stage1,
        np.arange(len(X_all)),
        test_size=test_size,
        random_state=rs,
        stratify=y_stage1,
    )

    preprocess = build_preprocess(available_features, num_cols)

    print("[INFO] Steg 1: trening + t1 (indre validering)")
    stage1, t1, idx_avslag = train_stage1_with_threshold(
        preprocess, X_train, pd.Series(y1_train), target_precision, val_size=val_size, random_state=123
    )
    print(f"[INFO] t1 (Avslag) = {t1}")

    print("[INFO] Steg 2: treningspool (sanne Tillatelse)")
    X2_pool, y2_pool = prepare_stage2_pools(X_all, y_all, y_stage1, idx_train)

    print("[INFO] Steg 2: trening + t2 per klasse (indre validering)")
    stage2, classes2, thresholds2 = train_stage2_with_thresholds(
        preprocess, X2_pool, y2_pool, target_precision, val_size=val_size, random_state=123
    )
    print(f"[INFO] t2 per klasse = {thresholds2}")

    print("[INFO] Evaluerer policy (med holdnings-gate) på test")
    auto_n, auto_acc, coverage, df_auto, blocked = evaluate_policy_on_test(
        stage1, stage2, idx_avslag, classes2, thresholds2, t1, X_test, y_all, idx_test
    )

    # VIKTIG: Disse tallene er ETTER holdnings-gate (riktige å logge/lagre)
    print(f"[RESULT] Auto-decisions (after holdnings-gate): {auto_n} (coverage {coverage:.3f}); accuracy {auto_acc:.3f}")
    print(f"[INFO] Rader i test med IKKE-tillatt holdningsklasse (blokkerte av policy): {blocked}")

    print("[INFO] Lagrer artefakter og policy-rapport")
    save_artifacts(stage1, stage2, t1, thresholds2, auto_n, auto_acc, coverage, target_precision, blocked)

    if args.export_auto_csv:
        # Eksporter KUN auto-besluttede rader (etter holdnings-gate)
        # df_auto har allerede features + target + beslutning/konfidens
        # Du kan tilpasse kolonnerekkefølgen her:
        base_cols = [TARGET_COL, "auto_decision", "auto_confidence", "stage1_pred", "p_avslag", HOLDNINGS_COL]
        ordered_cols = [c for c in base_cols if c in df_auto.columns]
        # legg til resten av features til slutt (uten duplikat)
        for c in available_features:
            if c not in ordered_cols and c in df_auto.columns:
                ordered_cols.append(c)

        out_csv = "auto_approved_holdningsklasse.csv"
        df_auto[ordered_cols].to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] Skrevet {len(df_auto)} auto-besluttede rader (med holdnings-gate) til {out_csv}")

    print("[DONE]")


if __name__ == "__main__":
    main()