import os
import pandas as pd
import fitz
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import re
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

# 5. Lag binær fasitkolonne
df["VEDTAK_BINÆR"] = df["EGS.VEDTAK.10670"].apply(lambda x: "Avslag" if x == "Avslag" else "Godkjent")

# 6. Velg numeriske features for matching
features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense",
    "Avkjørsler", "Trafikkulykker", "Kurvatur, horisontal", "Kurvatur, stigning"
]

# 7. Train/Test-split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 8. Skaler numeriske verdier
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[features].fillna(0))
test_scaled = scaler.transform(test_df[features].fillna(0))

# 9. JSON parsing
def parse_json_response(result_text):
    if not result_text.strip():
        return False
    clean_text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", result_text).strip()
    try:
        parsed = json.loads(clean_text)
        return parsed.get("approve", False)
    except json.JSONDecodeError:
        return False

# 10. Format rad
def format_row(row):
    row_dict = row.drop(["EGS.VEDTAK.10670", "VEDTAK_BINÆR"]).to_dict()
    return "\n".join([f"{key}: {value}" for key, value in row_dict.items()])

def get_few_shot_examples(test_row, test_vector, train_df, train_scaled, scaler):
    # Start med streng filtrering
    filtered = train_df.copy()
    filtered = filtered[filtered["Avkjørsel, holdningsklasse"] == test_row["Avkjørsel, holdningsklasse"]]

    # Legg til bruksområde hvis nok eksempler
    new_filtered = filtered[filtered["EGS.BRUKSOMRÅDE.1256"] == test_row["EGS.BRUKSOMRÅDE.1256"]]
    if len(new_filtered) > 0:
        filtered = new_filtered

    # Hvis filtrering gir tomt datasett, bruk hele train_df
    if len(filtered) == 0:
        filtered = train_df.copy()

    # Beregn avstand for filtrerte
    filtered_scaled = scaler.transform(filtered[features].fillna(0))
    filtered["distance"] = ((filtered_scaled - test_vector) ** 2).sum(axis=1)

    # Fyll opp hvis for få
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
    
    # Velg alle avslåtte og godkjente
    avslagte = train_df[train_df["VEDTAK_BINÆR"] == "Avslag"]
    godkjente = filtered[filtered["VEDTAK_BINÆR"] == "Godkjent"].nsmallest(math.floor(len(avslagte)*0.5), "distance")
    print("antall godkjente som ble med fra filtrering: ", len(godkjente))
    # Fyll på godkjente eksempler hvis for få
    godkjente = fill_examples(train_df, godkjente, "Godkjent", math.floor(len(avslagte)*0.5), test_vector)

    return pd.concat([godkjente, avslagte]).sort_values("distance")

# 11. Prediksjonsfunksjon med dynamisk few-shot
def predict_approval(row, row_index):
    # Beregn avstand
    test_vector = test_scaled[row_index]
    few_shot_df = get_few_shot_examples(row, test_vector, train_df, train_scaled, scaler)
    #few_shot_df = train_df[train_df["VEDTAK_BINÆR"] == "Avslag"]

    # Lag few-shot eksempler
    few_shot_examples = []
    for _, ex in few_shot_df.iterrows():
        example = f"""
        Søknadsdata:
        {format_row(ex)}
        Fasit:
        {{"approve": {"true" if ex['VEDTAK_BINÆR'] == "Godkjent" else "false"}}}
        """
        few_shot_examples.append(example)

    # Lag prompt
    prompt = f"""
    Du er saksbehandler i Statens vegvesen. Gitt følgende søknadsdata og retningslinjene nedenfor, skal du avgjøre om søknaden burde fått godkjent eller avslag. 

    Nå vurder denne søknaden:
    {format_row(row)}

    Retningslinjer:
    {pdf_text}

    Her er noen eksempler med fasit:
    {''.join(few_shot_examples)}

    Returner kun JSON med formatet {{"approve": true}} eller {{"approve": false}}.
    """
    try:
        response = client.chat.completions.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        approve = parse_json_response(response.choices[0].message.content.strip())
        return "Godkjent" if approve else "Avslag"
    except Exception as e:
        print("Feil:", e)
        return "Feil"

# Antall avslag i testsettet
n_avslag = test_df[test_df["VEDTAK_BINÆR"] == "Avslag"].shape[0]

# Filtrer alle avslag
avslag_df = test_df[test_df["VEDTAK_BINÆR"] == "Avslag"]

# Velg like mange godkjente som avslag
godkjent_df = test_df[test_df["VEDTAK_BINÆR"] == "Godkjent"].sample(n=n_avslag*2, random_state=42)

# Kombiner og shuffle
balanced_test_df = pd.concat([avslag_df, godkjent_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("Fordeling i balansert testsett:")
print(balanced_test_df["VEDTAK_BINÆR"].value_counts())

# 12. Prediksjon med progressbar
predictions = []
for i, (_, row) in enumerate(tqdm(balanced_test_df.iterrows(), total=len(balanced_test_df), desc="Predikerer")):
    predictions.append(predict_approval(row, i))

balanced_test_df["Prediksjon"] = predictions

# 13. Evaluer
y_true = balanced_test_df["VEDTAK_BINÆR"]
y_pred = balanced_test_df["Prediksjon"]
valid_mask = y_pred.isin(["Godkjent", "Avslag"])

if valid_mask.sum() > 0:
    conf_matrix = confusion_matrix(y_true[valid_mask], y_pred[valid_mask], labels=["Godkjent", "Avslag"])
    report = classification_report(y_true[valid_mask], y_pred[valid_mask])
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", report)
else:
    print("Ingen gyldige prediksjoner ble gjort.")

# 14. Lagre resultater
balanced_test_df.to_csv("prediksjoner_test_dynamisk.csv", index=False)

# 10. Print statistikk
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", report)
print("\nAntall prediksjoner gjort:", valid_mask.sum(), "av", len(df))

