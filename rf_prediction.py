import pickle
import pandas as pd
import numpy as np
import gzip


def get_rf_prediction(avkjorsel, bakke, adt_total, andel_lange, fartsgrense, sving=0, model_path="models/balanced_rf_model.pkl.gz"):
    """
    Beregn sannsynlighet for avdslag ved bruk av en forhåndstrent RandomForest-modell.

    Input:
        avkjorsel: "Avkjørsler"
        bakke: "Kurvatur, stigning"
        adt_total: "ÅDT, total"
        andel_lange: "ÅDT, andel lange kjøretøy"
        fartsgrense: "Fartsgrense"
        sving: "Kurvatur, horisontal' 

    Output:
        prob_avslag: predikert sannsynlighet for "Avslag", 0–100 (float)
        klasse: klassifisert sannsynlighet basert på historikk og prediksjon (vudering av model_training_txt)
    """

    # Sørg for at alle inputs er float
    try:
        avkjorsel = float(avkjorsel)
        bakke = float(bakke)
        adt_total = float(adt_total)
        andel_lange = float(andel_lange)
        fartsgrense = float(fartsgrense)
        sving = float(sving)
    except ValueError:
        return "Feil: En eller flere input-verdier kunne ikke konverteres til tall."

    # Beregn avledet variabel
    antall_lange = adt_total * andel_lange/100
    bakke=np.abs(bakke)
    sving=np.abs(sving)

    if sving>99000:
        sving_ind=0
    else:
        sving_ind=1

    if bakke>0.1:
        bakke_ind=1
    else:
        sving_ind=0

    if sving<99000:
        sving_sigmoid= 1 / (1 + np.exp(-0.001 * sving))
    else:
        sving_sigmoid=0


    # Forbered én rad med inputdata slik modellen forventer dem
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

    # Konverter dict → DataFrame (formatet scikit-learn krever)
    prediction_df = pd.DataFrame([prediction_row])

    # -------------------------------
    # Last inn modellen trygt med try/except
    # -------------------------------
    try:
        with gzip.open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return "Feil: Fant ikke modellfilen. Sjekk at models/balanced_rf_model.pkl.gz finnes."
    except Exception as e:
        return f"Feil ved lasting av modell: {e}"

    # -------------------------------
    # Kjør prediksjon med try/except
    # -------------------------------
    try:
        pred = model.predict_proba(prediction_df)
    except Exception as e:
        return f"Feil under prediksjon: {e}"

    # Hent ut sannsynlighet for 'Avslag' og rund av til én desimal (prosent)
    prob_avslag = round(pred[0][1] * 100, 1)

    if prob_avslag< 5:
        klasse = "statistisk lav sannsynlighet for avslag"
    elif prob_avslag>5 and prob_avslag <25:
        klasse= "statistisk medium sannsynlighet for avslag"
    else:
        klasse = "statistisk høy sannsynlighet for avslag"


    return prob_avslag, klasse


# Example test:
print(get_rf_prediction(5, 0.1, 5000, 0.1, 40))
