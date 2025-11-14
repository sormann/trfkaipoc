import pickle
import pandas as pd
import numpy as np
import gzip
from treeinterpreter import treeinterpreter as ti


def get_rf_prediction(
        avkjorsel, bakke, adt_total, andel_lange, fartsgrense, sving=0,
        model_path="models/balanced_rf_model.pkl.gz",
        return_exp=False):
    
    FEATURE_MAP = {
    'ÅDT, total Avkjørsler bakke': "Total trafikkmengde per år × antall avkjørsler × kurvatur, stigning",
    'ÅDT, total Avkjørsler': "Total trafikkmengde per år × antall avkjørsler",
    'ÅDT, total antall_lange_kj': "Total trafikkmengde per år × antall tunge kjøretøy per år",
    'ÅDT, andel lange kjøretøy Avkjørsler antall_lange_kj': "Andel tunge kjøretøy per år × antall avkjørsler × antall tunge kjøretøy per år",
    'Avkjørsler bakke antall_lange_kj': "Antall avkjørsler × kurvatur, stigning × antall tunge kjøretøy per år",
    'ÅDT, total Avkjørsler antall_lange_kj': "Total trafikkmengde per år × avkjørsler × antall tunge kjøretøy per år",
    'Avkjørsler antall_lange_kj': "Antall avkjørsler × antall tunge kjøretøy per år",
    'ÅDT, total bakke': "Total trafikkmengde per år × kurvatur, stigning",
    'ÅDT, total ÅDT, andel lange kjøretøy': "Total trafikkmengde per år × andel tunge kjøretøy",
    'ÅDT, total Fartsgrense Avkjørsler': "Total trafikkmengde per år × fartsgrense × antall avkjørsler"
    }

    # Konverter til tall
    avkjorsel = float(avkjorsel)
    bakke = float(bakke)
    adt_total = float(adt_total)
    andel_lange = float(andel_lange)
    fartsgrense = float(fartsgrense)
    sving = float(sving)

    # Avledede verdier
    antall_lange = adt_total * andel_lange / 100
    bakke = np.abs(bakke)
    sving = np.abs(sving)

    # Input slik modellen forventer
    prediction_row = {
        'ÅDT, total Avkjørsler bakke': adt_total * avkjorsel * bakke,
        'ÅDT, total Avkjørsler': adt_total * avkjorsel,
        'ÅDT, total antall_lange_kj': adt_total * antall_lange,
        'ÅDT, andel lange kjøretøy Avkjørsler antall_lange_kj': andel_lange * avkjorsel * antall_lange,
        'Avkjørsler bakke antall_lange_kj': avkjorsel * bakke * antall_lange,
        'ÅDT, total Avkjørsler antall_lange_kj': adt_total * avkjorsel * antall_lange,
        'Avkjørsler antall_lange_kj': avkjorsel * antall_lange,
        'ÅDT, total bakke': adt_total * bakke,
        'ÅDT, total ÅDT, andel lange kjøretøy': adt_total * antall_lange,
        'ÅDT, total Fartsgrense Avkjørsler': adt_total * fartsgrense * avkjorsel
    }
    prediction_df = pd.DataFrame([prediction_row])

    # Last modell
    with gzip.open(model_path, "rb") as f:
        model = pickle.load(f)

    # Modellprediksjon
    pred = model.predict_proba(prediction_df)[0]
    prob_avslag = float(pred[1])

    # Klassifisering
    if prob_avslag < 0.05:
        klasse = "statistisk lav sannsynlighet for avslag"
    elif prob_avslag < 0.50:
        klasse = "statistisk medium sannsynlighet for avslag"
    else:
        klasse = "statistisk høy sannsynlighet for avslag"

    # --------------------------------------
    # Forklaring (treeinterpreter)
    # --------------------------------------
    # Forklaring med treeinterpreter

    prediction, bias, contributions = ti.predict(model, prediction_df.values)
    contrib_class1 = contributions[0][:, 1]

    explanation_df = pd.DataFrame({
        "technical_name": prediction_df.columns,
        "readable_name": [FEATURE_MAP[col] for col in prediction_df.columns],
        "value": prediction_df.iloc[0],
        "contribution": contrib_class1
    }).sort_values(by="contribution", ascending=False)

    drivers_high = explanation_df[explanation_df["contribution"] > 0].head(3)
    drivers_low = explanation_df[explanation_df["contribution"] < 0].tail(3)

    # Pen tekst
    def format_list(df):
        if df.empty:
            return "  (ingen vesentlige faktorer)"
        out = []
        for _, r in df.iterrows():
            navn = r["readable_name"]
            verdi = round(float(r["value"]), 3)
            effekt = round(float(r["contribution"]), 3)
            tegn = "+" if effekt >= 0 else ""
            out.append(f"  • {navn} (verdi: {verdi}, effekt: {tegn}{effekt})")
        return "\n".join(out)

    pretty_text = (
        f"Sannsynlighet for avslag: {round(prob_avslag * 100, 1)} %\n"
        f"Kategori: {klasse}\n\n"
        f"Faktorer som ØKER sannsynligheten for avslag:\n{format_list(drivers_high)}\n\n"
        f"Faktorer som REDUSERER sannsynligheten for avslag:\n{format_list(drivers_low)}"
    )

    if return_exp:
        return {
            "probability_percent": round(prob_avslag * 100, 1),
            "klasse": klasse,
            "drivers_higher": drivers_high.to_dict(orient="records"),
            "drivers_lower": drivers_low.to_dict(orient="records"),
            "pretty_text": pretty_text
        }
    else:
        return prob_avslag, klasse

# Test
print(get_rf_prediction(5, 0.1, 5000, 0.1, 40)["pretty_text"])
