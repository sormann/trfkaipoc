#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
two_stage_confidence.py

Trener en to-stegs modell for EGS.VEDTAK.10670 (Steg 1: Avslag vs Tillatelse,
Steg 2: type for Tillatelse), og legger på en "confidence-gating" policy som
kun auto-beslutter når modellens sikkerhet er høy nok, ellers "manual_review".

Kjør:
    python predict_with_confidence.py

Etter kjøring lagres:
    - egsv_stage1_avslag_vs_tillatelse.pkl
    - egsv_stage2_tillatelsestyper.pkl
    - confidence_thresholds.json
    - confidence_policy_report.txt

Scriptet eksporterer også:
    - load_two_stage_with_thresholds()
    - predict_with_manual_review()

for bruk i prediksjon/drift.
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
# Konfigurasjon
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


# -----------------------------
# Lasting & rensing av data
# -----------------------------
def load_and_clean(path: Path) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Les CSV, rens, og returner (df, available_features, num_cols)."""
    df0 = pd.read_csv(path, sep=";", dtype=str)
    df0.columns = [c.strip() for c in df0.columns]
    for c in df0.columns:
        df0[c] = df0[c].astype(str).str.strip()

    available_features = [c for c in FEATURE_COLS if c in df0.columns]
    use_cols = [TARGET_COL] + available_features
    df = df0[use_cols].copy()

    # Behold kun rader med målverdi
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] != "")]

    # Numerisk konvertering
    num_cols = [c for c in available_features if c in NUM_COLS_BASE]
    for c in num_cols:
        df[c] = pd.to_numeric(
            df[c]
            .str.replace("\u00A0", "", regex=True)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # Rens kategorier
    cat_cols = [c for c in available_features if c not in num_cols]
    for c in cat_cols:
        df[c] = df[c].str.replace(" +", " ", regex=True)
        df[c] = df[c].str.replace(" \- ", " - ", regex=True)

    # Fjern helt like duplikater (features + target)
    if available_features:
        df = df.drop_duplicates(subset=available_features + [TARGET_COL])

    return df, available_features, num_cols


# -----------------------------
# Preprosessering og modeller
# -----------------------------
def build_preprocess(available_features: List[str], num_cols: List[str]) -> ColumnTransformer:
    numeric_features = [c for c in num_cols if c in available_features]
    categorical_features = [c for c in available_features if c not in numeric_features]

    num_pre = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pre = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # NB: sparse_output=False krever nyere scikit-learn. Om nødvendig: bytt til sparse=False.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("num", num_pre, numeric_features), ("cat", cat_pre, categorical_features)]
    )
    return preprocess


def train_stage1_with_threshold(
    preprocess: ColumnTransformer,
    X_train: pd.DataFrame,
    y1_train: pd.Series,
    target_precision: float,
    val_size: float = 0.2,
    random_state: int = 123,
) -> Tuple[Pipeline, Optional[float], Optional[int]]:
    """
    Tren steg 1 (Avslag vs Tillatelse). Finn t1 på et indre valideringssett som gir presisjon >= target_precision
    for auto-Avslag, og maksimer dekning under dette kravet. Returnerer (pipeline, t1, idx_avslag).
    """
    # Indre valideringssplit
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y1_train, test_size=val_size, random_state=random_state, stratify=y1_train
    )

    stage1_val = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced_subsample"))
        ]
    )
    stage1_val.fit(X_tr, y_tr)

    proba_val = stage1_val.predict_proba(X_val)
    classes1 = list(stage1_val.named_steps["model"].classes_)
    idx_avslag = classes1.index("Avslag")
    p_avslag_val = proba_val[:, idx_avslag]
    pred_val = stage1_val.predict(X_val)

    mask_pred_avslag = (pred_val == "Avslag")
    # Finn terskelkandidater
    cand = np.unique(np.round(p_avslag_val[mask_pred_avslag], 6))
    best_t1, best_prec1, best_cov1 = None, 0.0, 0.0
    true_bin = (y_val == "Avslag").astype(int).to_numpy()
    for t in cand:
        m = mask_pred_avslag & (p_avslag_val >= t)
        if m.sum() == 0:
            continue
        prec = true_bin[m].mean()
        cov = m.mean()  # andel av valideringssettet
        if prec >= target_precision and cov > best_cov1:
            best_t1, best_prec1, best_cov1 = float(t), float(prec), float(cov)

    # Tren endelig steg 1 på hele treningssettet
    stage1 = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced_subsample"))
        ]
    )
    stage1.fit(X_train, y1_train)
    return stage1, best_t1, idx_avslag
def prepare_stage2_pools(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    y_stage1: np.ndarray,
    idx_train: np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Hent ut pool til steg 2-trening: sanne 'Tillatelse'-rader i train-delen.
    """
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
    """
    Tren steg 2 på 'Tillatelse'-poolen. Finn per-klasse terskler t2[k] på indre validering
    som gir presisjon >= target_precision og samsvarer best mulig coverage for hver klasse.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X2_pool, y2_pool, test_size=val_size, random_state=random_state, stratify=y2_pool
    )

    stage2_val = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=600, random_state=42, class_weight="balanced_subsample"))
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
        # Kandidatterskler fra p_cls
        for t in np.unique(np.round(p_cls, 6)):
            m = p_cls >= t
            if m.sum() == 0:
                continue
            prec = y_true_cls[m].mean()
            cov = m.sum() / len(y_val)  # dekning mot valideringssettet i steg 2
            if prec >= target_precision and cov > best_cov:
                best_t, best_cov = float(t), float(cov)

        thresholds2[cls] = best_t

    # Tren endelig steg 2 på hele poolen
    stage2 = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=600, random_state=42, class_weight="balanced_subsample"))
        ]
    )
    stage2.fit(X2_pool, y2_pool)
    return stage2, classes2, thresholds2


# -----------------------------
# Policy-evaluering og lagring
# -----------------------------
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
) -> Tuple[int, float, float]:
    """
    Evaluer "auto vs manual_review" policy på testsettet.
    Returnerer (auto_n, auto_accuracy, coverage).
    """
    proba1 = stage1.predict_proba(X_test)
    pred1 = stage1.predict(X_test)
    p_avslag = proba1[:, idx_avslag]

    N = len(X_test)
    auto_decision = np.array(["manual_review"] * N, dtype=object)

    # Auto Avslag
    if t1 is not None:
        mask_auto_avslag = (pred1 == "Avslag") & (p_avslag >= t1)
        auto_decision[mask_auto_avslag] = "Avslag"

    # Steg 2 for resten predikert Tillatelse
    need_stage2_pos = np.where((auto_decision == "manual_review") & (pred1 == "Tillatelse"))[0]
    if len(need_stage2_pos):
        sub = X_test.iloc[need_stage2_pos]
        proba2 = stage2.predict_proba(sub)
        classes2_arr = np.array(classes2)
        top_idx = proba2.argmax(axis=1)
        top_cls = classes2_arr[top_idx]
        top_p = proba2.max(axis=1)
        # Anvend terskler pr klasse
        for j, pos in enumerate(need_stage2_pos):
            thr = thresholds2.get(top_cls[j])
            if thr is not None and top_p[j] >= thr:
                auto_decision[pos] = top_cls[j]

    final_true = y_all.reset_index(drop=True).iloc[idx_test].values
    auto_mask = (auto_decision != "manual_review")
    auto_n = int(auto_mask.sum())
    auto_acc = float((auto_decision[auto_mask] == final_true[auto_mask]).mean()) if auto_n > 0 else float("nan")
    coverage = auto_n / N
    return auto_n, auto_acc, coverage


def save_artifacts(
    stage1: Pipeline,
    stage2: Pipeline,
    t1: Optional[float],
    thresholds2: Dict[str, Optional[float]],
    auto_n: int,
    auto_acc: float,
    coverage: float,
    target_precision: float,
) -> None:
    import joblib

    joblib.dump(stage1, "egsv_stage1_avslag_vs_tillatelse.pkl")
    joblib.dump(stage2, "egsv_stage2_tillatelsestyper.pkl")
    Path("confidence_thresholds.json").write_text(
        json.dumps(
            {
                "target_precision": target_precision,
                "stage1": {"Avslag_threshold": t1},
                "stage2": {"class_thresholds": thresholds2},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    Path("confidence_policy_report.txt").write_text(
        f"Target precision (auto): {target_precision}\n"
        f"Auto-decisions (test): {auto_n} (coverage {coverage:.3f})\n"
        f"Accuracy på auto-beslutninger: {auto_acc:.3f}\n",
        encoding="utf-8",
    )


# -----------------------------
# Hjelpefunksjoner for drift
# -----------------------------
def load_two_stage_with_thresholds(
    m1_path: str = "egsv_stage1_avslag_vs_tillatelse.pkl",
    m2_path: str = "egsv_stage2_tillatelsestyper.pkl",
    thresholds_path: str = "confidence_thresholds.json",
):
    import joblib
    m1 = joblib.load(m1_path)
    m2 = joblib.load(m2_path)
    thr = json.load(open(thresholds_path, "r", encoding="utf-8"))
    return m1, m2, thr


def predict_with_manual_review(
    df_features: pd.DataFrame,
    m1: Pipeline,
    m2: Pipeline,
    thr: Dict,
) -> pd.DataFrame:
    """
    Prediksjon i drift:

    - Hvis pred_stage1 == "Avslag" og p_avslag >= t1 -> auto: "Avslag"
    - Ellers hvis pred_stage1 == "Tillatelse" og top2_prob >= t2[pred_klasse] -> auto: klassen
    - Ellers "manual_review"

    Returnerer DataFrame med kolonner: pred_stage1, p_avslag, decision
    """
    proba1 = m1.predict_proba(df_features)
    classes1 = list(m1.named_steps["model"].classes_)
    idx_avslag = classes1.index("Avslag")
    p_avslag = proba1[:, idx_avslag]
    pred1 = m1.predict(df_features)

    res = pd.DataFrame(index=df_features.index)
    res["pred_stage1"] = pred1
    res["p_avslag"] = p_avslag
    res["decision"] = "manual_review"

    t1 = thr["stage1"]["Avslag_threshold"]
    if t1 is not None:
        mask = (pred1 == "Avslag") & (p_avslag >= t1)
        res.loc[mask, "decision"] = "Avslag"

    idx_till = res[(res["decision"] == "manual_review") & (res["pred_stage1"] == "Tillatelse")].index
    if len(idx_till):
        sub = df_features.loc[idx_till]
        proba2 = m2.predict_proba(sub)
        classes2 = list(m2.named_steps["model"].classes_)
        top_idx = proba2.argmax(axis=1)
        top_cls = [classes2[i] for i in top_idx]
        top_p = proba2.max(axis=1)
        for j, idx in enumerate(idx_till):
            thr2 = thr["stage2"]["class_thresholds"].get(top_cls[j])
            if thr2 is not None and top_p[j] >= thr2:
                res.loc[idx, "decision"] = top_cls[j]

    return res


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="To-stegs modell med confidence-gating (auto vs manual review)")
    parser.add_argument("--data", type=str, default="data_2022-2025.csv", help="Sti til CSV-fil")
    parser.add_argument("--target-precision", type=float, default=0.99, help="Minste presisjon på auto-beslutninger")
    parser.add_argument("--test-size", type=float, default=0.2, help="Andel til test")
    parser.add_argument("--val-size", type=float, default=0.2, help="Andel til indre validering for terskler")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
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

    print("[INFO] Lager tren/test-splitt (stratifisert på Avslag/Tillatelse)")
    X_train, X_test, y1_train, y1_test, idx_train, idx_test = train_test_split(
        X_all,
        y_stage1,
        np.arange(len(X_all)),
        test_size=test_size,
        random_state=rs,
        stratify=y_stage1,
    )

    print("[INFO] Setter opp preprosessering")
    preprocess = build_preprocess(available_features, num_cols)

    print("[INFO] Trener Steg 1 + finner terskel t1 på indre validering")
    stage1, t1, idx_avslag = train_stage1_with_threshold(
        preprocess, X_train, pd.Series(y1_train), target_precision, val_size=val_size, random_state=123
    )
    print(f"[INFO] t1 (Avslag) = {t1}")

    print("[INFO] Forbereder Steg 2 treningspool (sanne Tillatelse i train)")
    X2_pool, y2_pool = prepare_stage2_pools(X_all, y_all, y_stage1, idx_train)

    print("[INFO] Trener Steg 2 + finner klasse-spesifikke terskler")
    stage2, classes2, thresholds2 = train_stage2_with_thresholds(
        preprocess, X2_pool, y2_pool, target_precision, val_size=val_size, random_state=123
    )
    print(f"[INFO] t2 per klasse = {thresholds2}")

    print("[INFO] Evaluerer 'auto vs manual_review' på testsett")
    auto_n, auto_acc, coverage = evaluate_policy_on_test(
        stage1, stage2, idx_avslag, classes2, thresholds2, t1, X_test, y_all, idx_test
    )
    print(f"[RESULT] Auto-decisions: {auto_n} (coverage {coverage:.3f}), accuracy på auto: {auto_acc:.3f}")

    # ------------------------------------------------------------
    # (NYTT) Lag en CSV med alle auto-besluttede rader
    # ------------------------------------------------------------
    # 1) Reproduser auto-beslutningene + confidence (sannsynligheter)
    proba1_test = stage1.predict_proba(X_test)
    pred1_test = stage1.predict(X_test)
    p_avslag_test = proba1_test[:, idx_avslag]

    # Start med "manual_review" for alle
    N = len(X_test)
    auto_decision = np.array(['manual_review'] * N, dtype=object)
    decision_conf = np.array([np.nan] * N, dtype=float)  # lagrer confidence

    # Auto Avslag i steg 1 dersom p_avslag >= t1
    if t1 is not None:
        mask_auto_avslag = (pred1_test == 'Avslag') & (p_avslag_test >= t1)
        auto_decision[mask_auto_avslag] = 'Avslag'
        decision_conf[mask_auto_avslag] = p_avslag_test[mask_auto_avslag]

    # Til steg 2: rader som fortsatt er manual_review men predikert "Tillatelse" i steg 1
    need_stage2_pos = np.where((auto_decision == 'manual_review') & (pred1_test == 'Tillatelse'))[0]
    if len(need_stage2_pos):
        sub = X_test.iloc[need_stage2_pos]
        proba2_test = stage2.predict_proba(sub)
        classes2_arr = np.array(classes2)
        top_idx = proba2_test.argmax(axis=1)
        top_cls = classes2_arr[top_idx]
        top_p = proba2_test.max(axis=1)

        # Bruk klasse-spesifikke terskler
        for j, pos in enumerate(need_stage2_pos):
            thr = thresholds2.get(top_cls[j])
            if (thr is not None) and (top_p[j] >= thr):
                auto_decision[pos] = top_cls[j]
                decision_conf[pos] = float(top_p[j])

    # 2) Bygg ut en tabell med original-data (inkl. ID hvis kolonne finnes)
    # Finn ut om det finnes en ID/objekt-kolonne i original-data (bruk gjerne 'OBJ.VEGOBJEKT-ID' hvis den finnes)
    possible_id_cols = [c for c in df.columns if c.lower() in ['obj.vegobjekt-id', 'obj.vegobjekt\\-id', 'obj.vegobjekt_id']]
    id_col = possible_id_cols[0] if len(possible_id_cols) else None
    # NB: idx_test refererer til rad-indekser i df (etter rensing og deduplisering)
    df_test = df.reset_index(drop=True).iloc[idx_test].copy()

    # Ta med ønskede kolonner i eksporten:
    base_cols = []
    if id_col is not None:
        base_cols.append(id_col)
    base_cols += [TARGET_COL]  # fasit
    # Ta også med feature-kolonnene for kontekst
    export_cols = base_cols + [c for c in available_features if c in df_test.columns]

    # Sett på prediksjon, steg1-resultater og confidence
    df_test['stage1_pred'] = pred1_test
    df_test['p_avslag'] = p_avslag_test
    df_test['auto_decision'] = auto_decision
    df_test['auto_confidence'] = decision_conf

    # Filtrer kun auto-besluttede rader
    df_auto = df_test[df_test['auto_decision'] != 'manual_review'].copy()

    # Rekkefølge på kolonner i eksport: ID (hvis finnes), fasit, auto_decision, auto_confidence, p_avslag, stage1_pred, features...
    ordered_cols = []
    if id_col is not None and id_col in df_auto.columns:
        ordered_cols.append(id_col)
    ordered_cols += [TARGET_COL, 'auto_decision', 'auto_confidence', 'stage1_pred', 'p_avslag']
    # Ta med features til slutt (unngå duplikater)
    for c in available_features:
        if c in df_auto.columns and c not in ordered_cols:
            ordered_cols.append(c)

    # Skriv CSV
    out_csv = 'auto_approved.csv'
    df_auto[ordered_cols].to_csv(out_csv, index=False, encoding='utf-8')
    print(f"[INFO] Skrevet {len(df_auto)} auto-besluttede rader til {out_csv}")

    print("[INFO] Lagrer artefakter og policy-rapport")
    save_artifacts(stage1, stage2, t1, thresholds2, auto_n, auto_acc, coverage, target_precision)
    print("[DONE] Artefakter skrevet til disk.")


if __name__ == "__main__":
    main()