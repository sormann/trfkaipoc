
import streamlit as st
import pandas as pd
from streamlit_extras.stylable_container import stylable_container

# 1. Les ferdige prediksjoner fra CSV
@st.cache_data
def load_predictions():
    return pd.read_csv("data_annotert_med_prediksjoner.csv", encoding="utf-8")

df = load_predictions()

# 2. Streamlit UI
st.title("Søknadsvurdering for avkjørsler")

st.sidebar.header("Filtrer søknader")

# Filtreringsvalg basert på kolonner i CSV
mottatt_dato = st.sidebar.date_input("Søknad mottatt etter", value=None)
søknadstype = st.sidebar.multiselect("Hva det søkes om", sorted(df["Søknadstype"].dropna().unique()))
formål = st.sidebar.multiselect("Formål", sorted(df["EGS.BRUKSOMRÅDE.1256"].dropna().unique()))
funksjonsklasse = st.sidebar.multiselect("Funksjonsklasse", sorted(df["Funksjonsklasse"].dropna().unique()))
holdningsklasse = st.sidebar.multiselect("Holdningsklasse", sorted(df["Avkjørsel, holdningsklasse"].dropna().unique()))
kommune = st.sidebar.multiselect("Kommune", sorted(df["LOK.KOMMUNE"].dropna().unique()))

def getReadableTurnCategory(turnCategory: str, radius: float):
    if turnCategory == "Krapp sving":
        return f"Sving (svingradius {abs(radius)}m)"
    return turnCategory

# Numeriske filtre
fartsgrense = st.sidebar.slider(
    "Fartsgrense (km/t)",
    int(df["Fartsgrense"].min()),
    int(df["Fartsgrense"].max()),
    (int(df["Fartsgrense"].min()), int(df["Fartsgrense"].max()))
)
adt = st.sidebar.slider(
    "ÅDT (trafikkmengde)",
    int(df["ÅDT, total"].min()),
    int(df["ÅDT, total"].max()),
    (int(df["ÅDT, total"].min()), int(df["ÅDT, total"].max()))
)

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
    filtered_df["EGS.SØKNAD MOTTATT DATO.12048"] = pd.to_datetime(filtered_df["EGS.SØKNAD MOTTATT DATO.12048"], errors="coerce")
    filtered_df = filtered_df[filtered_df["EGS.SØKNAD MOTTATT DATO.12048"] >= pd.to_datetime(mottatt_dato)]

filtered_df = filtered_df[
    (filtered_df["Fartsgrense"].between(fartsgrense[0], fartsgrense[1])) &
    (filtered_df["ÅDT, total"].between(adt[0], adt[1]))
]

def getKlasse(prob_avslag):
    # Klassifisering
    if prob_avslag < 5:
        return "lav sannsynlighet for avslag"
    elif prob_avslag < 50:
        return "medium sannsynlighet for avslag"
    else:
        return "høy sannsynlighet for avslag"


# Vis resultater
if len(filtered_df) == 0:
    st.warning("Ingen søknader passer filteret. Endre filtrene for å se resultater.")
else:
    filtered_df["display_label"] = filtered_df.apply(
        lambda row: f"Saksnr: {row.get('EGS.SAKSNUMMER.1822', '')} | For {str(row.get('EGS.BRUKSOMRÅDE.1256', '')).lower()} på {row.get('VSR.VEGSYSTEMREFERANSE', '').split(' ')[0]} i {row.get('LOK.KOMMUNE', '')} mottatt {row.get('EGS.SØKNAD MOTTATT DATO.12048', '')}",
        axis=1
    )

    selected_id = st.selectbox(
        "Velg søknad",
        options=filtered_df["ID"],
        format_func=lambda x: filtered_df.loc[filtered_df["ID"] == x, "display_label"].values[0]
    )

    target_row = filtered_df[filtered_df["ID"] == selected_id]
    if not target_row.empty:
        row = target_row.iloc[0]

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

        # Vis lagrede prediksjoner
        st.subheader("Automatisk råd")
        st.write(f"**Kategori:** {row.get('Prediksjon_kategori', 'Ikke tilgjengelig')} søknad (Sannsynlighet for vanskelig: {round(row.get('p_vanskelig', 'Ikke tilgjengelig'),2)})")
        st.write(f"**Sannsynlighet for avslag:** Vår maskinlæringsmodell, som er trent på historiske søknader fra perioden januar 2022 til oktober 2025, predikerer en sannsynlighet for avslag på {row.get('prob_avslag', 'Ikke tilgjengelig')}%. I denne perioden er ca. 5% avslått, så dette kan regnes som {getKlasse(row.get('prob_avslag', 'Ikke tilgjengelig'))}.")
        st.write(f"**Foreslått vedtak:** {row.get('Prediksjon_vedtak', 'Ikke tilgjengelig')} for {row.get('Søknadstype', '').lower()}.")
        st.write(f"**Begrunnelse for vedtak:** {row.get('Begrunnelse_vedtak', 'Ikke tilgjengelig')}")
    else:
        st.error("Fant ikke søknad med valgt ID.")
