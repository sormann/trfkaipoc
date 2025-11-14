import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report
import gzip
import pickle
from datetime import datetime


def train_model():
    # ---------------------------------------------------
    # Last data
    # ---------------------------------------------------
    file_path = "https://raw.githubusercontent.com/sormann/trfkaipoc/refs/heads/main/data_2022-2025_norge.csv"
    df = pd.read_csv(file_path, sep=";")
    df = df[df["EGS.VEDTAK.10670"].notna()]

    # Lag target-variabel
    df['Avslag_ind'] = df['EGS.VEDTAK.10670'].apply(
        lambda x: 1 if x == "Avslag" or x == "Avslag etter klage " else 0
    )

    # Fallback dersom kolonnavn varierer
    if "Kurvatur, horisontalelement" in df.columns:
        df["Kurvatur, horisontal"] = df["Kurvatur, horisontalelement"]

    # ---------------------------------------------------
    # Featurevalg
    # ---------------------------------------------------
    features = [
        'Avslag_ind',
        "ÅDT, total",
        "ÅDT, andel lange kjøretøy",
        "Fartsgrense",
        "Avkjørsel, holdningsklasse",
        "Funksjonsklasse",
        "Avkjørsler",
        "Trafikkulykker",
        "EGS.BRUKSOMRÅDE.1256",
        "Kurvatur, horisontal",
        "Kurvatur, stigning"
    ]

    df_encoded = pd.get_dummies(df[features])
    df_encoded = df_encoded.dropna()

    # Ekstra variabler
    df_encoded['sving_ind'] = np.where(df_encoded['Kurvatur, horisontal'].abs() > 99000, 0, 1)
    df_encoded['bakke'] = df_encoded['Kurvatur, stigning'].abs()
    df_encoded['bakke_ind'] = np.where(df_encoded['Kurvatur, stigning'].abs() > 0.1, 1, 0)
    df_encoded['sving_sigmoid'] = np.where(
        df_encoded['Kurvatur, horisontal'].abs() < 99000,
        1 / (1 + np.exp(-0.001 * df_encoded['Kurvatur, horisontal'].abs())),
        0
    )
    df_encoded['antall_lange_kj'] = df_encoded['ÅDT, total'] * df_encoded['ÅDT, andel lange kjøretøy'] / 100

    # Drop original kurvatur-variabler
    df_encoded = df_encoded.drop(['Kurvatur, horisontal', 'Kurvatur, stigning'], axis=1)

    # ---------------------------------------------------
    # Shuffle
    # ---------------------------------------------------
    df_encoded = df_encoded.sample(frac=1)

    y = df_encoded['Avslag_ind']
    X = df_encoded.drop(columns=['Avslag_ind'])

    # Polynomial interactions
    poly = PolynomialFeatures(3, include_bias=False, interaction_only=True)
    X = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(X.columns))

    # Foreløpig modell for å finne top 10 features
    model = BalancedRandomForestClassifier(
        sampling_strategy=0.5,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # Finn top features
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances_sorted = importances.sort_values(ascending=False)
    top_features = importances_sorted.index[:10]

    # Reduser datasettet
    X = X[top_features]

    # ---------------------------------------------------
    # Endelig trening
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)

    smote = ADASYN(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = BalancedRandomForestClassifier(
        n_estimators=5000,
        random_state=42,
        n_jobs=-1,
        sampling_strategy=1
    )
    model.fit(X_train, y_train)


    # ---------------------------------------------------
    # Evaluer på testsett og lag classification report
    # ---------------------------------------------------
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    # ---------------------------------------------------
    # Lagre modellen med timestamp
    # ---------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Lagre kun komprimert modell direkte
    gz_filename = f"balanced_rf_model_{timestamp}.pkl.gz"
    with gzip.open(gz_filename, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---------------------------------------------------
    # Skriv loggfil med top features og timestamp
    # ---------------------------------------------------
    log_filename = "model_training_log.txt"
    with open(log_filename, "a", encoding="utf-8") as log:
        log.write("\n" + "-" * 60 + "\n")
        log.write(f"Modell trent: {timestamp}\n")
        log.write(f"Fil lagret: {gz_filename}\n")
        log.write("Topp 10 features:\n")
        for feat in top_features:
            log.write(f"  - {feat}\n")

        log.write("\nClassification report (test-set):\n")
        log.write(report + "\n")

        log.write("-" * 60 + "\n")

    print("✔ Modell trent og lagret:")
    print("   ", gz_filename)
    print("✔ Top features logget til:", log_filename)


# Kjør funksjonen
train_model()
