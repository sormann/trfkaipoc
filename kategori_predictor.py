
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from scipy.sparse import hstack
import numpy as np

def train_model(df):
    # 1. Target og base features
    target_col = 'Vurdering'
    numeric_cols = [
        'Fartsgrense', 'Avkjorsler', 'Trafikkulykker', 'ÅDT, total',
        'ÅDT, andel lange kjøretøy', 'Kurvatur, horisontal', 'Kurvatur, stigning'
    ]
    text_col = 'Kommentar'

    # 2. Rens data
    data = df.copy()
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data.loc[:, col] = data[col].fillna(data[col].median())
    data[text_col] = data[text_col].fillna("")
    data = data.dropna(subset=[target_col])

    # 3. Encode target
    le = LabelEncoder()
    data[target_col] = le.fit_transform(data[target_col])  # Enkel=0, Vanskelig=1

    # 4. Feature engineering
    data['ÅDT_per_avkjørsel'] = data['ÅDT, total'] / (data['Avkjorsler'] + 1)
    data['ÅDT_per_avkjørsel'] = data['ÅDT_per_avkjørsel'].fillna(0)
    data['log_ÅDT'] = np.log1p(data['ÅDT, total'])
    numeric_cols += ['ÅDT_per_avkjørsel', 'log_ÅDT']

    # 5. Legg til prob_avslag og Prediksjon_vedtak_bin
    if 'prob_avslag' not in data.columns:
        data['prob_avslag'] = 0.0
    numeric_cols.append('prob_avslag')

    if 'Prediksjon_vedtak' in data.columns:
        data['Prediksjon_vedtak_bin'] = data['Prediksjon_vedtak'].apply(lambda x: 1 if x == 'Avslag' else 0)
        numeric_cols.append('Prediksjon_vedtak_bin')

    # 6. Split features
    X_num = data[numeric_cols]
    y = data[target_col]

    # 7. TF-IDF for tekst
    vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(data[text_col])

    # 8. Kombiner numerisk + tekst
    X = hstack([X_num, X_text])

    # 9. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    # 10. Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # 11. Compute scale_pos_weight
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]

    # 12. Train XGBoost
    model = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, random_state=42,
                          scale_pos_weight=scale_pos_weight, reg_lambda=1.0,
                          objective='binary:logistic', eval_metric='logloss')
    model.fit(X_train_res, y_train_res)

    # 13. Threshold tuning
    y_proba = model.predict_proba(X_test)[:, 1]

    def evaluate(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec_pos = precision_score(y_true, y_pred, average='binary', pos_label=1)
        rec_pos = recall_score(y_true, y_pred, average='binary', pos_label=1)
        f1_pos = f1_score(y_true, y_pred, average='binary', pos_label=1)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return acc, prec_pos, rec_pos, f1_pos, prec_neg

    thresholds = np.linspace(0.1, 0.9, 17)
    best_score = -1
    best_tuple = None
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        acc, prec_pos, rec_pos, f1_pos, prec_neg = evaluate(y_test, y_pred_thr)
        # Kombiner recall for Vanskelig og presisjon for Enkel
        score = (rec_pos + prec_neg) / 2  # balansert score
        if prec_neg >= 0.6 and rec_pos >= 0.6:  # krav for begge klasser
            if score > best_score:
                best_score = score
                best_tuple = (thr, acc, prec_pos, rec_pos, f1_pos, prec_neg)

    best_threshold = best_tuple[0] if best_tuple else 0.5
    print(f"Valgt threshold: {best_threshold}, score: {best_score}")


    return model, vectorizer, numeric_cols, best_threshold


def predict_category(row: pd.Series, model, vectorizer, numeric_cols, threshold: float):
    # Forbered numeriske features
    num = {c: row.get(c, 0.0) for c in numeric_cols}
    aadt_total = num.get('ÅDT, total', 0.0)
    avkj = num.get('Avkjorsler', 0.0)
    num['ÅDT_per_avkjørsel'] = aadt_total / (avkj + 1)
    num['log_ÅDT'] = np.log1p(aadt_total)
    X_num = pd.DataFrame([num])

    # Tekstfeatures
    text = row.get('Kommentar', '')
    if pd.isna(text):
        text = ''
    X_text = vectorizer.transform([text])

    # Kombiner
    X_row = hstack([X_num, X_text])

    # Prediksjon
    proba_vanskelig = float(model.predict_proba(X_row)[:, 1])
    kategori = 'Vanskelig' if proba_vanskelig >= threshold else 'Enkel'
    forklaring = f"p(Vanskelig)={proba_vanskelig:.2f}, threshold={threshold:.2f}"
    return kategori, forklaring, proba_vanskelig
