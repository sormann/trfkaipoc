# Streamlit-applikasjon for prediksjon av avkjørselssøknader

## 1. Oversikt

Denne applikasjonen gir **beslutningsstøtte** for behandling av søknader om avkjørsler basert på:

- **Historiske data (2022–2025)** fra Nasjonal vegdatabank (`data_2022-2025.csv`)
- **Retningslinjer** fra _Håndbok R701 – Behandling av avkjørselssaker_
- **Maskinlæringsmodeller** for:
  - Prediksjon av sannsynlighet for avslag (Balanced Random Forest)
  - Klassifisering av søknadskompleksitet (XGBoost)

Applikasjonen kombinerer:

- Klassisk ML-modellering
- Forklarbarhet (TreeInterpreter)
- Språkmodellbasert resonnering (Azure OpenAI GPT-4.1)

---

## 2. Arkitektur

Applikasjonen består av to hovedkomponenter:

### **Frontend (Streamlit)**

- Filtrering av søknader (dato, formål, funksjonsklasse, kommune, etc.)
- Presentasjon av søknadsdata og prediksjoner
- Interaktiv vurdering med maskinlæringsresultater og språkmodellbegrunnelse

### **Prediksjonsmodeller (maskinlæring og språkmodeller)**

- **Steg 1: Avslagssannsynlighetsprediksjon**
  - Modell: Balanced Random Forest
  - Dokumentasjon: avslagsprediksjon.md
  - Output:
    - Sannsynlighet for avslag (0–100%)
    - Risikokategori: lav / medium / høy
    - Forklaring basert på feature-bidrag
- **Steg 2: Vedtaksprediksjon**
  - Modell: Azure OpenAI GPT-4.1 til å foreslå vedtak basert på:
    - Søknadsdata
    - ML-forklaring
    - Retningslinjer fra Håndbok R701
    - Eksempler fra lignende saker (few-shot learning)
  - Output:
  - Godkjent/Avslag
  - Begrunnelse for avgjørelse
- **Steg 3: Kompleksitetsklassifisering**
  - Modell: XGBoost
  - Output:
    - Kategori: _Enkel_ eller _Vanskelig_
    - Sannsynlighet for vanskelig sak

###

---

## 3. Datagrunnlag

- **Fil:** `data_2022-2025.csv`
- **Kilde:** Nasjonal vegdatabank
- **Periode:** 2022–2025
- **Antall saker:** 464 med registrert vedtak
- **Begrensning:** Veldig skjevt datasett, kun 19 registrerte avslag
- **Kolonner (utvalg):**
  - `OBJ.VEGOBJEKT-ID`
  - `EGS.BRUKSOMRÅDE.1256` (Formål)
  - `EGS.SAKSNUMMER.1822`
  - `EGS.VEDTAK.10670` (Avslag/Godkjent)
  - `ÅDT, total` (Årsdøgntrafikk)
  - `Fartsgrense`
  - `Avkjørsel, holdningsklasse`
  - `Funksjonsklasse`
  - `Kurvatur, horisontal` og `Kurvatur, stigning`
  - `Søknadstype`
  - `Svingkategori`

---

## 4. Funksjonalitet i applikasjonen

### **Filtrering**

- Dato for mottatt søknad
- Søknadstype, formål, funksjonsklasse, holdningsklasse
- Kommune
- Numeriske filtre: fartsgrense, ÅDT

### **Prediksjoner**

- **Sannsynlighet for avslag:** Basert på Balanced Random Forest
- **Kategori (Enkel/Vanskelig):** Basert på XGBoost
- **Foreslått vedtak:** Basert på språkmodell og retningslinjer

### **Forklaringer**

- Feature-bidrag fra ML-modellen (TreeInterpreter)
- Tekstlig begrunnelse fra språkmodellen

---

## 5. Teknisk oppsett

### **Avhengigheter**

- `streamlit`
- `pandas`, `numpy`
- `scikit-learn`, `imbalanced-learn`
- `xgboost`
- `matplotlib`
- `PyMuPDF` (for PDF-tekst)
- `openai` (Azure OpenAI SDK)
- `.env` for API-nøkler

### **Kjøring**

```bash
# Installer avhengigheter
pip install -r requirements.txt

# Start applikasjonen
streamlit run app.py
```

---

## 6. Modellbeskrivelse

### **Avslagsprediksjon (Balanced Random Forest)**

- Håndterer ubalansert datasett (ca. 3% avslag)
- Bruker ADASYN-oversampling
- Forklarbarhet via TreeInterpreter
- Dokumentasjon: avslagsprediksjon.md

### **Kompleksitetsklassifisering (XGBoost)**

- Treningsdata: annotert vurdering (_Enkel_ / _Vanskelig_)
- Feature engineering:
  - `ÅDT_per_avkjørsel`
  - `log_ÅDT`
  - `prob_avslag` fra steg 1
- Threshold-tuning for balansert presisjon og recall

---

## 7. Resultater fra modellene

### **Avslagsprediksjon (Balanced Random Forest)**

- **Modell:** `balanced_rf_model_20251114_132757.pkl.gz`

- **Topp 10 features:**

  1.  ÅDT, total × bakke
  2.  ÅDT, andel lange kjøretøy × bakke × antall_lange_kj
  3.  Funksjonsklasse_E – Lokale adkomstveger × sving_sigmoid
  4.  ÅDT, total × antall_lange_kj
  5.  Avkjørsler × Funksjonsklasse_E × sving_sigmoid
  6.  ÅDT, total × Avkjørsel, holdningsklasse_Streng × bakke
  7.  ÅDT, total × bakke × antall_lange_kj
  8.  ÅDT, total × Fartsgrense × bakke
  9.  Avkjørsler × Funksjonsklasse_D × Bruksområde_Skog/skogbruk
  10. ÅDT, andel lange kjøretøy × Fartsgrense

- **Ytelse (test-sett):**
  precision recall f1-score support
  0 (ikke avslag) 0.98 0.94 0.96 334
  1 (avslag) 0.05 0.11 0.07 9
  accuracy 0.92

  - **Macro avg:** 0.51 (precision), 0.53 (recall)
  - **Weighted avg:** 0.95 (precision), 0.92 (recall)

- **Kvantil for predikert sannsynlighet:**

  - 25%: 0.0055
  - Median: 0.0292
  - 75%: 0.1103

- **Kommentar:** Modellen gir noe separasjon, men svært lav recall for avslag pga. ekstrem ubalanse (kun 3% avslag).

---

### **Vedtaksprediksjon (LLM + few-shot)**

- **Confusion Matrix:**
  [[226 215]
[  5  14]]
- **Classification Report:**
  Avslag: precision=0.06, recall=0.74, f1=0.11 (support=19)
  Godkjent: precision=0.98, recall=0.51, f1=0.67 (support=441)
  accuracy=0.52
- **Kommentar:** Språkmodellen har høy recall for avslag (74%), men svært lav presisjon (6%). Den er konservativ og foreslår avslag oftere enn fasit. Presisjon på kategorien "enkel" er 98%.

---

### **Kategoriprediksjon (XGBoost)**

- **Threshold valgt:** 0.1 (score=0.79)
- **Confusion Matrix:**
  [[10 10]
[ 2  6]]
- **Classification Report:**
  Enkel: precision=0.83, recall=0.50, f1=0.62 (support=20)
  Vanskelig: precision=0.38, recall=0.75, f1=0.50 (support=8)
  accuracy=0.57
- **Kommentar:** Modellen prioriterer recall for vanskelige saker, men presisjon er lav. Nyttig for å flagge komplekse saker.
