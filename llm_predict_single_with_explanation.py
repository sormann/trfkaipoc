import os
import pandas as pd
import fitz
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.preprocessing import StandardScaler
import json
import re
import uuid
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from rf_prediction import get_rf_prediction

# 1. Last inn miljøvariabler
load_dotenv()
api_version = os.getenv("api_version")
endpoint = os.getenv("endpoint")
subscription_key = os.getenv("subscription_key")
openai_model_name = "gpt-4.1"

# 2. Initialiser Azure OpenAI-klient
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# 3. Les PDF-retningslinjer
with fitz.open("HB_R701_Behandling_avkjorselssaker_2014.pdf") as pdf:
    pdf_text = "\n".join(page.get_text() for page in pdf)

# 4. Les datasettet
df = pd.read_csv("data_annotert.csv", sep=";")
df = df.dropna(subset=['Vurdering'])
df["ID"] = [str(uuid.uuid4()) for _ in range(len(df))]
df = df.drop(columns=["Kurvatur, horisontal", "EGS.VEDTAKSDATO.11444", "EGS.TILLEGGSINFORMASJON.11566", "EGS.TILLATELSE GJELDER TIL DATO.12049", "EGS.GEOMETRI, PUNKT.4753"])
df["Kurvatur, stigning"] = df["Kurvatur, stigning"].abs()

# 5. Lag binær fasitkolonne
df["VEDTAK_BINÆR"] = df["EGS.VEDTAK.10670"].apply(lambda x: "Avslag" if x == "Avslag" else "Godkjent")

# 6. Velg numeriske features for matching
features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense", "Trafikkulykker"
]

# 8. Skaler numeriske verdier
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features].fillna(0))

# 10. Format rad

def format_row(row):
    row_dict = row.drop(["EGS.VEDTAK.10670", "VEDTAK_BINÆR", "ID", "OBJ.VEGOBJEKT-ID", "EGS.SAKSNUMMER.1822", "EGS.ARKIVREFERANSE, URL.12050"]).to_dict()
    return "\n".join([f"{key}: {value}" for key, value in row_dict.items()])

def get_few_shot_examples(test_row, test_vector, df_without_test, scaler):
    filtered = df_without_test[df_without_test["Avkjørsel, holdningsklasse"] == test_row["Avkjørsel, holdningsklasse"]]
    new_filtered = filtered[filtered["EGS.BRUKSOMRÅDE.1256"] == test_row["EGS.BRUKSOMRÅDE.1256"]]
    if len(new_filtered) > 0:
        filtered = new_filtered
    if len(filtered) == 0:
        filtered = df_without_test.copy()

    filtered_scaled = scaler.transform(filtered[features].fillna(0))
    filtered["distance"] = ((filtered_scaled - test_vector) ** 2).sum(axis=1)

    def fill_examples(df_source, existing_df, label, needed, vector):
        current = existing_df
        missing = needed - len(current)
        if missing > 0:
            remaining = df_source[df_source["VEDTAK_BINÆR"] == label].drop(existing_df.index, errors='ignore')
            remaining_scaled = scaler.transform(remaining[features].fillna(0))
            remaining["distance"] = ((remaining_scaled - vector) ** 2).sum(axis=1)
            filler = remaining.nsmallest(missing, "distance")
            return pd.concat([current, filler])
        return current

    avslagte = df_without_test[df_without_test["VEDTAK_BINÆR"] == "Avslag"]
    godkjente = filtered[filtered["VEDTAK_BINÆR"] == "Godkjent"].nsmallest(1, "distance")
    godkjente = fill_examples(df_without_test, godkjente, "Godkjent", 1, test_vector)

    examples = pd.concat([godkjente, avslagte]).sort_values("distance")
    return examples

def predict_approval_detailed(row, row_index, prob_avslag, ml_prediction_explanation):
    kategori = "Vanskelig"
    begrunnelseForKategori = ""
    #if row["Avkjørsel, holdningsklasse"] != "Lite streng":
    #    begrunnelseForKategori = f"holdningsklassen er {row['Avkjørsel, holdningsklasse']}"

    #if row["Fartsgrense"] > 60 and row["Søknadstype"] not in ["Utvidet bruk", "Endret bruk"]:
    #    begrunnelseForKategori = f"fartsgrensen er {row['Fartsgrense']}km/t og søknadstypen er {row['Søknadstype']}"

    #if row["Svingkategori"] == "Krapp sving" and row["Søknadstype"] not in ["Utvidet bruk", "Endret bruk"]:
    #   begrunnelseForKategori = f"avkjørselen er i en sving og søknadstypen er {row['Søknadstype']}"

    test_vector = df_scaled[row_index]
    df_without_test = df[df["ID"] != row["ID"]]
    few_shot_df = get_few_shot_examples(row, test_vector, df_without_test, scaler)

    few_shot_examples = []
    for _, ex in few_shot_df.iterrows():
        example = f"""
        Søknadsdata:
        {format_row(ex)}
        Fasit:
        {{"approve": {"true" if ex['VEDTAK_BINÆR'] == "Godkjent" else "false"}}}
        """
        few_shot_examples.append(example)

    prompt = f"""
    Du er saksbehandler i Statens vegvesen. Gitt følgende søknadsdata, tree-explainer fra maskinlæringsmodell og retningslinjene nedenfor, skal du avgjøre om søknaden burde fått godkjent eller avslag. 

    {ml_prediction_explanation}
        
    Nå vurder denne søknaden:
    {format_row(row)}

    Retningslinjer:
    {pdf_text}

    Her er noen eksempler med fasit:
    {''.join(few_shot_examples)}

    Returner kun JSON med formatet:
    {{
        "approve": true/false,
        "reason": "tekstlig forklaring"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = json.loads(re.sub(r"^```[a-zA-Z]*\n?|```$", "", response.choices[0].message.content.strip()))
        vedtak = "Godkjent" if result.get("approve") else "Avslag"
        begrunnelse = result.get("reason", "Ingen begrunnelse oppgitt.")
        if begrunnelseForKategori == "":
            if vedtak == "Godkjent" and prob_avslag < 50:
                kategori = "Enkel"
                begrunnelseForKategori = "det ikke høy sannsynlighet for avslag og foreslått vedtak er godkjent"
            elif vedtak == "Godkjent" and prob_avslag >= 50:
                kategori = "Vanskelig"
                begrunnelseForKategori = "det er høy sannsynlighet for avslag"
            else:
                kategori = "Vanskelig"
                begrunnelseForKategori = "foreslått vedtak er avslag"
        
        return {
            "vedtak": vedtak,
            "begrunnelse": begrunnelse,
            "kategori": kategori,
            "begrunnelseForKategori": begrunnelseForKategori,
        }
    except Exception as e:
        return {
            "vedtak": "Feil",
            "begrunnelse": str(e),
            "kategori": kategori,
            "begrunnelseForKategori": begrunnelseForKategori,
        }
    


predictions = []
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
    ml_prediction_explanation = ml_prediction["pretty_text"]
    predictions.append(predict_approval_detailed(row, i, prob_avslag, ml_prediction_explanation)["kategori"])

df["Prediksjon"] = predictions

# 13. Evaluer
y_true = df["Vurdering"]
y_pred = df["Prediksjon"]
valid_mask = y_pred.isin(["Enkel", "Vanskelig"])

if valid_mask.sum() > 0:
    conf_matrix = confusion_matrix(y_true[valid_mask], y_pred[valid_mask], labels=["Enkel", "Vanskelig"])
    report = classification_report(y_true[valid_mask], y_pred[valid_mask])
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)
    print("\nAntall prediksjoner gjort:", valid_mask.sum(), "av", len(df))
else:
    print("Ingen gyldige prediksjoner ble gjort.")

# 14. Lagre resultater
df.to_csv("prediksjoner_test_dynamisk.csv", index=False)