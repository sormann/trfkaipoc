
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import numpy as np
import pickle
import gzip
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from rf_prediction import get_rf_prediction

def train_model(df):
    target_col = 'Vurdering'
    numeric_cols = [
        'Fartsgrense', 'Avkjørsler', 'Trafikkulykker', 'ÅDT, total',
        'ÅDT, andel lange kjøretøy', 'Kurvatur, horisontal', 'Kurvatur, stigning'
    ]

    # Rens data
    data = df.copy()
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data.loc[:, col] = data[col].fillna(data[col].median())
    data = data.dropna(subset=[target_col])

    # Encode target
    le = LabelEncoder()
    data[target_col] = le.fit_transform(data[target_col])  # Enkel=0, Vanskelig=1

    # Feature engineering
    data['ÅDT_per_avkjørsel'] = data['ÅDT, total'] / (data['Avkjørsler'] + 1)
    data['ÅDT_per_avkjørsel'] = data['ÅDT_per_avkjørsel'].fillna(0)
    data['log_ÅDT'] = np.log1p(data['ÅDT, total'])
    numeric_cols += ['ÅDT_per_avkjørsel', 'log_ÅDT']

    prob_avslag_liste = [] 
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Predikerer")):
        ml_prediction = get_rf_prediction(
            avkjorsel=row.get('Avkjørsel', 0),
            bakke=row.get('Kurvatur, stigning', 0),
            adt_total=row.get('ÅDT, total', 0),
            andel_lange=row.get('ÅDT, andel lange kjøretøy', 0),
            fartsgrense=row.get('Fartsgrense', 0),
            sving=row.get('Kurvatur, horisontal', 0),
            return_exp=True
            )
        prob_avslag = ml_prediction["probability_percent"]
        prob_avslag_liste.append(prob_avslag)

    data["prob_avslag"] = prob_avslag_liste  

    if 'prob_avslag' in data.columns:
        numeric_cols.append('prob_avslag')

    if 'Prediksjon_vedtak' in data.columns:
        data['Prediksjon_vedtak_bin'] = data['Prediksjon_vedtak'].apply(lambda x: 1 if x == 'Avslag' else 0)
        numeric_cols.append('Prediksjon_vedtak_bin')

    X = data[numeric_cols]
    y = data[target_col]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=42, stratify=y)

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # Compute scale_pos_weight
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

    # Train XGBoost
    model = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, random_state=42,
                          scale_pos_weight=scale_pos_weight, reg_lambda=1.0,
                          objective='binary:logistic', eval_metric='logloss')
    model.fit(X_train_res, y_train_res)

    # Threshold tuning
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 17)
    best_score = -1
    best_tuple = None
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thr).ravel()
        prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = (rec_pos + prec_neg) / 2
        if prec_neg >= 0.6 and rec_pos >= 0.6:
            if score > best_score:
                best_score = score
                best_tuple = (thr, score)

    best_threshold = best_tuple[0] if best_tuple else 0.5
    print(f"\nValgt threshold: {best_threshold}, score: {best_score}")

    # Lagre modellen
    with gzip.open("models/xgb_model.pkl.gz", "wb") as f:
        pickle.dump((model, numeric_cols, best_threshold), f)

    return model, numeric_cols, best_threshold, X_test, y_test


def predict_category(row: pd.Series):
    with gzip.open("models/xgb_model.pkl.gz", "rb") as f:
        model, numeric_cols, threshold = pickle.load(f)

    num = {c: row.get(c, 0.0) for c in numeric_cols}
    aadt_total = num.get('ÅDT, total', 0.0)
    avkj = num.get('Avkjørsler', 0.0)
    num['ÅDT_per_avkjørsel'] = aadt_total / (avkj + 1)
    num['log_ÅDT'] = np.log1p(aadt_total)
    X_row = pd.DataFrame([num])

    proba_vanskelig = model.predict_proba(X_row)[0, 1]  # FIX for DeprecationWarning
    kategori = 'Vanskelig' if proba_vanskelig >= threshold else 'Enkel'
    return kategori, proba_vanskelig




def main():
    # 1. Les datasettet
    df = pd.read_csv("data_annotert.csv", sep=";")
    df = df.dropna(subset=["Vurdering"])  # Kun rader med fasit

    # 2. Tren modellen og få testsettet tilbake
    model, numeric_cols, best_threshold, X_test, y_test = train_model(df)

    # 3. Prediker på testsettet med valgt threshold
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = np.where(y_proba_test >= best_threshold, "Vanskelig", "Enkel")

    # 4. Evaluer ytelse på testsettet
    print("\n# 13. Evaluer (TESTSETT etter threshold tuning)")
    conf_matrix = confusion_matrix(y_test.map({0: "Enkel", 1: "Vanskelig"}), y_pred_test, labels=["Enkel", "Vanskelig"])
    report = classification_report(y_test.map({0: "Enkel", 1: "Vanskelig"}), y_pred_test, labels=["Enkel", "Vanskelig"])
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)
    print("\nAntall testprediksjoner:", len(y_test))


    # Etter trening:
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': numeric_cols,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print(feature_importance_df)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title('Feature Importance (XGBoost)')
    plt.show()




if __name__ == "__main__":
    main()