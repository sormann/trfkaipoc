import os
import pandas as pd
import fitz
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.preprocessing import StandardScaler
import json
import re
import uuid

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
df["ID"] = [str(uuid.uuid4()) for _ in range(len(df))]

# 5. Lag binær fasitkolonne
df["VEDTAK_BINÆR"] = df["EGS.VEDTAK.10670"].apply(lambda x: "Avslag" if x == "Avslag" else "Godkjent")

df.drop(columns=["Kurvatur, horisontal","Kurvatur, stigning"])

# 6. Velg numeriske features for matching
features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense",
    "Avkjørsler", "Trafikkulykker"
]

# 8. Skaler numeriske verdier
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features].fillna(0))

# 10. Format rad
def format_row(row):
    row_dict = row.drop(["EGS.VEDTAK.10670", "VEDTAK_BINÆR"]).to_dict()
    return "\n".join([f"{key}: {value}" for key, value in row_dict.items()])

def get_few_shot_examples(test_row, test_vector, df_without_test, scaler):
    # Start med streng filtrering
    filtered = df_without_test[df_without_test["Avkjørsel, holdningsklasse"] == test_row["Avkjørsel, holdningsklasse"]]

    # Legg til bruksområde hvis nok eksempler
    new_filtered = filtered[filtered["EGS.BRUKSOMRÅDE.1256"] == test_row["EGS.BRUKSOMRÅDE.1256"]]
    if len(new_filtered) > 0:
        filtered = new_filtered
    # Hvis filtrering gir tomt datasett, bruk hele df uten test-raden
    if len(filtered) == 0:
        filtered = df_without_test.copy()

    # Beregn avstand
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
    avslagte = df_without_test[df_without_test["VEDTAK_BINÆR"] == "Avslag"]
    godkjente = filtered[filtered["VEDTAK_BINÆR"] == "Godkjent"].nsmallest(1, "distance")

    # Fyll på godkjente eksempler hvis for få
    godkjente = fill_examples(df_without_test, godkjente, "Godkjent", 1, test_vector)

    examples = pd.concat([godkjente, avslagte]).sort_values("distance")

    # Verifiser at test-raden ikke er med
    if test_row["ID"] in df_without_test["ID"].values:
        print("⚠️ Test-raden er fortsatt med!")

    return examples

# 11. Prediksjonsfunksjon med dynamisk few-shot
def predict_approval_detailed(row, row_index):
    # Automatisk avslag basert på regler
    if row["Avkjørsel, holdningsklasse"] != "Lite streng":
        return {
            "vedtak": "Avslag",
            "begrunnelse": f"Automatisk vurdert som en vanskelig søknad på grunn av at holdningsklassen er {row['Avkjørsel, holdningsklasse']}."
        }

    if row["Fartsgrense"] > 60 and row["Søknadstype"] not in ["Utvidet bruk", "Endret bruk"]:
        return {
            "vedtak": "Avslag",
            "begrunnelse": f"Automatisk vurdert som en vanskelig søknad på grunn av fartsgrense {row['Fartsgrense']}km/t og søknadstypen er {row['Søknadstype']}."
        }
    
    if  row["Svingkategori"] == "Krapp sving" and row["Søknadstype"] not in ["Utvidet bruk", "Endret bruk"]:
        return {
            "vedtak": "Avslag",
            "begrunnelse": f"Automatisk vurdert som en vanskelig søknad på grunn av at avkjørselen er i en sving og søknadstypen er {row['Søknadstype']}."
        }

    # Beregn avstand
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
    Du er saksbehandler i Statens vegvesen. Gitt følgende søknadsdata og retningslinjene nedenfor, skal du avgjøre om søknaden burde fått godkjent eller avslag. 

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
            "begrunnelse": begrunnelse
        }
    except Exception as e:
        print("Feil:", e)
        return {
            "vedtak": "Feil",
            "begrunnelse": str(e)
        }

target_row = df[df["EGS.SAKSNUMMER.1822"] == "202443995-2"]
if not target_row.empty:
    index = target_row.index[0]
    result = predict_approval_detailed(target_row.iloc[0], index)
    if result["vedtak"] == "Godkjent":
        print("Kategori: Enkel søknad")
        print("Foreslått vedtak:", result["vedtak"])
    else:
        print("Kategori: Vanskelig søknad")
    print("Begrunnelse:", result["begrunnelse"])
else:
    print("Fant ikke søknad med saksnummer 202443995-2")