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
from kategori_predictor import predict_category
import math

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
df = pd.read_csv("data_2022-2025.csv", sep=";")
#df = df.dropna(subset=['Vurdering'])
df["ID"] = [str(uuid.uuid4()) for _ in range(len(df))]
df = df.drop(columns=["EGS.VEDTAKSDATO.11444", "EGS.TILLEGGSINFORMASJON.11566", "EGS.TILLATELSE GJELDER TIL DATO.12049", "EGS.GEOMETRI, PUNKT.4753"])
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
    filtered = df_without_test[df_without_test["Avkjørsel, holdningsklasse"] == test_row["Avkjørsel, holdningsklasse"]].copy()
    new_filtered = filtered[filtered["EGS.BRUKSOMRÅDE.1256"] == test_row["EGS.BRUKSOMRÅDE.1256"]].copy()
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
            remaining = df_source[df_source["VEDTAK_BINÆR"] == label].drop(existing_df.index, errors='ignore').copy()
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


        return {
            "vedtak": vedtak,
            "begrunnelse": begrunnelse,
        }
    except Exception as e:
        return {
            "vedtak": "Feil",
            "begrunnelse": str(e),
        }
    
# Antall avslag i testsettet
#n_avslag = df[df["VEDTAK_BINÆR"] == "Avslag"].shape[0]

# Filtrer alle avslag
#avslag_df = df[df["VEDTAK_BINÆR"] == "Avslag"]

# Velg godkjente basert på antall avslag
#godkjent_df = df[df["VEDTAK_BINÆR"] == "Godkjent"].sample(n=math.floor(n_avslag*2), random_state=42)

# Kombiner og shuffle
#df = pd.concat([avslag_df, godkjent_df]).sample(frac=1, random_state=42).reset_index(drop=True)

#print("Fordeling i balansert testsett:")
#print(df["VEDTAK_BINÆR"].value_counts())
    
vedtak_predictions = []
prob_avslag_liste = [] 
begrunnelser = []
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
    prob_avslag_liste.append(prob_avslag)
    result = predict_approval_detailed(row, i, prob_avslag, ml_prediction_explanation)
    vedtak_predictions.append(result["vedtak"])
    begrunnelser.append(result["begrunnelse"])

df["Prediksjon_vedtak"] = vedtak_predictions
df["Begrunnelse_vedtak"] = begrunnelser
df["prob_avslag"] = prob_avslag_liste  

# Bruk den oppdaterte modellen til å predikere kategori for hver rad
kategori_resultater = []
p_vanskelig_resultater = []

for _, row in df.iterrows():
    kategori, p_vanskelig = predict_category(row)
    kategori_resultater.append(kategori)
    p_vanskelig_resultater.append(p_vanskelig)

# Legg til i datasettet
df["Prediksjon_kategori"] = kategori_resultater
df["p_vanskelig"] = p_vanskelig_resultater

# 13. Evaluer vedtak
y_true_vedtak = df["VEDTAK_BINÆR"]
y_pred_vedtak = df["Prediksjon_vedtak"]
valid_mask_vedtak = y_pred_vedtak.isin(["Godkjent", "Avslag"])

if valid_mask_vedtak.sum() > 0:
    conf_matrix = confusion_matrix(y_true_vedtak[valid_mask_vedtak], y_pred_vedtak[valid_mask_vedtak], labels=["Godkjent", "Avslag"])
    report = classification_report(y_true_vedtak[valid_mask_vedtak], y_pred_vedtak[valid_mask_vedtak])
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)
    print("\nAntall prediksjoner gjort:", valid_mask_vedtak.sum(), "av", len(df))
else:
    print("Ingen gyldige prediksjoner ble gjort.")

# 14. Lagre resultater
df.to_csv("prediksjoner_test_dynamisk.csv", index=False)