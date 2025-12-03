
import streamlit as st
import os
import pandas as pd
import fitz
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.preprocessing import StandardScaler
import json
import re
import uuid
from streamlit_extras.stylable_container import stylable_container
from kategori_predictor import predict_category
from rf_prediction import get_rf_prediction
from tqdm import tqdm

# Last inn miljøvariabler
load_dotenv()
api_version = os.getenv("api_version")
endpoint = os.getenv("endpoint")
subscription_key = os.getenv("subscription_key")
openai_model_name = "gpt-4.1"

# Initialiser OpenAI-klient
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Les PDF-retningslinjer
@st.cache_data
def load_pdf_text():
    with fitz.open("HB_R701_Behandling_avkjorselssaker_2014.pdf") as pdf:
        return "\n".join(page.get_text() for page in pdf)

pdf_text = load_pdf_text()

# Les datasettet
@st.cache_data
def load_data():
    df = pd.read_csv("data_2022-2025.csv", sep=";")
    df["ID"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["VEDTAK_BINÆR"] = df["EGS.VEDTAK.10670"].apply(lambda x: "Avslag" if x == "Avslag" else "Godkjent")
    df = df.drop(columns=["EGS.VEDTAKSDATO.11444", "EGS.TILLEGGSINFORMASJON.11566", "EGS.TILLATELSE GJELDER TIL DATO.12049", "EGS.GEOMETRI, PUNKT.4753"])
    df["Kurvatur, stigning"] = df["Kurvatur, stigning"].abs()
    return df

df = load_data()

features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense", "Trafikkulykker"
]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features].fillna(0))

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

# Streamlit UI
st.title("Søknadsvurdering for avkjørsler")

st.sidebar.header("Filtrer søknader")
mottatt_dato = st.sidebar.date_input("Søknad mottatt etter", value=None)
søknadstype = st.sidebar.multiselect("Hva det søkes om", sorted(df["Søknadstype"].dropna().unique()))
formål = st.sidebar.multiselect("Formål", sorted(df["EGS.BRUKSOMRÅDE.1256"].dropna().unique()))
funksjonsklasse = st.sidebar.multiselect("Funksjonsklasse", sorted(df["Funksjonsklasse"].dropna().unique()))
holdningsklasse = st.sidebar.multiselect("Holdningsklasse", sorted(df["Avkjørsel, holdningsklasse"].dropna().unique()))
kommune = st.sidebar.multiselect("Kommune", sorted(df["LOK.KOMMUNE"].dropna().unique()))

# Numeriske filtre
fartsgrense = st.sidebar.slider("Fartsgrense (km/t)", int(df["Fartsgrense"].min()), int(df["Fartsgrense"].max()), (int(df["Fartsgrense"].min()), int(df["Fartsgrense"].max())))
adt = st.sidebar.slider("ÅDT (trafikkmengde)", int(df["ÅDT, total"].min()), int(df["ÅDT, total"].max()), (int(df["ÅDT, total"].min()), int(df["ÅDT, total"].max())))

# Filtrer datasettet
filtered_df = df.copy()
if formål:
    filtered_df = filtered_df[filtered_df["EGS.BRUKSOMRÅDE.1256"].isin(formål)]
if søknadstype:
    filtered_df = filtered_df[filtered_df["Søknadstype"].isin(søknadstype)]
if funksjonsklasse:
    filtered_df = filtered_df[filtered_df["Funksjonsklasse"].isin(funksjonsklasse)]
if holdningsklasse:
    filtered_df = filtered_df[filtered_df["Avkjørsel, holdningsklasse"].isin(holdningsklasse)]
if kommune:
    filtered_df = filtered_df[filtered_df["LOK.KOMMUNE"].isin(kommune)]
if mottatt_dato:
    # Konverter kolonnen til datetime hvis ikke allerede
    filtered_df["EGS.SØKNAD MOTTATT DATO.12048"] = pd.to_datetime(filtered_df["EGS.SØKNAD MOTTATT DATO.12048"], errors="coerce")
    filtered_df = filtered_df[filtered_df["EGS.SØKNAD MOTTATT DATO.12048"] >= pd.to_datetime(mottatt_dato)]

filtered_df = filtered_df[
    (filtered_df["Fartsgrense"].between(fartsgrense[0], fartsgrense[1])) &
    (filtered_df["ÅDT, total"].between(adt[0], adt[1]))
]

def getReadableTurnCategory(turnCategory:str, radius:float):
    if turnCategory == "Krapp sving":
        return f"Sving (svingradius {abs(radius)}m)"
    return turnCategory

def getKlasse(prob_avslag):
    # Klassifisering
    if prob_avslag < 5:
        return "lav sannsynlighet for avslag"
    elif prob_avslag < 50:
        return "medium sannsynlighet for avslag"
    else:
        return "høy sannsynlighet for avslag"

if len(filtered_df) == 0:
    st.warning("Ingen søknader passer filteret. Endre filtrene for å se resultater.")
else:
    filtered_df["display_label"] = filtered_df.apply(lambda row: f"Saksnr: {row.get('EGS.SAKSNUMMER.1822', '')} | For {str(row.get('EGS.BRUKSOMRÅDE.1256', '')).lower()} på {row.get('VSR.VEGSYSTEMREFERANSE', '').split(' ')[0]} i {row.get('LOK.KOMMUNE', '')} mottatt {row.get('EGS.SØKNAD MOTTATT DATO.12048', '')}", axis=1)
    selected_id = st.selectbox("Velg søknad", options=filtered_df["ID"], format_func=lambda x: filtered_df.loc[filtered_df["ID"] == x, "display_label"].values[0])

    target_row = filtered_df[filtered_df["ID"] == selected_id]
    if not target_row.empty:
        row = target_row.iloc[0]
        index = row.name

        # Vis søknadsinformasjon
        with stylable_container(
            key="info_box",
            css_styles="""
                {
                    background-color: #e6f2ff;
                    padding: 15px;
                    padding-bottom: 30px;
                    border-radius: 8px;
                    border: 1px solid #b3d7ff;
                    margin-bottom: 10px;
                }
            """
        ):
            st.write(f"""
            ### Søknadsinformasjon
            **Saksnummer:** {row.get('EGS.SAKSNUMMER.1822', 'Ikke oppgitt')}  
            **Søknad mottatt:** {row.get('EGS.SØKNAD MOTTATT DATO.12048', 'Ikke oppgitt')}   
            **Hva det søkes om:** {row.get('Søknadstype', 'Ikke oppgitt')}  
            **Formål:** {row.get('EGS.BRUKSOMRÅDE.1256', 'Ikke oppgitt')}  
            **Vegsystemreferanse:** {row.get('VSR.VEGSYSTEMREFERANSE', 'Ikke oppgitt')}  
            **Kommune:** {row.get('LOK.KOMMUNE', 'Ikke oppgitt')}  
            **Funksjonsklasse:** {row.get('Funksjonsklasse', 'Ikke oppgitt')}  
            **Holdningsklasse:** {row.get('Avkjørsel, holdningsklasse', 'Ikke oppgitt')}  
            **Trafikkulykker:** {row.get('Trafikkulykker', 'Ikke oppgitt')}  
            **Sving eller rett veg:** {getReadableTurnCategory(row.get('Svingkategori', 'Ikke oppgitt'), row.get('Kurvatur, horisontal', 0))}   
            **Fartsgrense:** {row.get('Fartsgrense', 'Ikke oppgitt')} km/t  
            **Trafikkmengde (ÅDT):** {row.get('ÅDT, total', 'Ikke oppgitt')}  
            **Arkivreferanse (URL):** {row.get('EGS.ARKIVREFERANSE, URL.12050', 'Ikke oppgitt')}  
            """)

        with st.spinner("Vurderer søknaden..."):
            ml_prediction = get_rf_prediction(
                avkjorsel=row.get('Avkjørsler', 0),
                bakke=row.get('Kurvatur, stigning', 0),
                adt_total=row.get('ÅDT, total', 0),
                andel_lange=row.get('ÅDT, andel lange kjøretøy', 0),
                fartsgrense=row.get('Fartsgrense', 0),
                sving=row.get('Kurvatur, horisontal', 0),
                return_exp=True
            )        
            prob_avslag = ml_prediction["probability_percent"]
            ml_prediction_explanation = ml_prediction["pretty_text"]
            result = predict_approval_detailed(row, index, prob_avslag, ml_prediction_explanation)
            kategori, p_vanskelig = predict_category(row)

        # Vis lagrede prediksjoner
        st.subheader("Automatisk råd")
        st.write(f"**Kategori:** {kategori} søknad (Sannsynlighet for vanskelig: {p_vanskelig:.2f})")
        st.write(f"**Sannsynlighet for avslag:** Vår maskinlæringsmodell, som er trent på historiske søknader fra perioden januar 2022 til oktober 2025, predikerer en sannsynlighet for avslag på {prob_avslag}%. I denne perioden er ca. 5% avslått, så dette kan regnes som {getKlasse(prob_avslag)}.")
        st.write(f"**Foreslått vedtak:** {result['vedtak']} for {row['Søknadstype'].lower()}.")
        st.write(f"**Begrunnelse for foreslått vedtak:** {result['begrunnelse']}")
    else:
        st.error("Fant ikke søknad med valgt ID.")
