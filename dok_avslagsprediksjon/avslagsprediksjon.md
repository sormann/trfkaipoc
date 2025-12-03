## Modell i kontekst: Fler-stegs beslutningsstøttesystem

Denne modellen inngår som **første trinn i en fler-stegs beslutningsstøtteløsning** for behandling av veisøknader. Arkitekturen kombinerer klassisk maskinlæring, forklarbarhet og språkmodellbasert resonnering.

Hvert trinn har et tydelig avgrenset formål, og ingen enkeltkomponent fatter vedtak alene.

---

## Oversikt over beslutningskjeden

### Steg 1: Prediksjon av sannsynlighet for avslag (Random Forest)

Først kjøres Balanced Random Forest-modellen beskrevet i dette dokumentet.  
Utdata fra dette steget er:

- numerisk sannsynlighet for avslag (0–1)
- en grov klassifisering:
  - lav sannsynlighet for avslag
  - medium sannsynlighet for avslag
  - høy sannsynlighet for avslag
- en forklaring basert på treeinterpreter (se under)

Denne informasjonen:
- vises direkte til brukere i frontend
- mates videre inn i språkmodellen i steg 2

Formålet med steg 1 er å gi en **objektiv, statistisk risikovurdering** basert på historiske data.

---
## Steg 2 og 3 er dokumentert 

---

## Forklarbarhet i modellen: `predictions.py`

Prediksjonen i steg 1 utføres av `predictions.py`, som laster den trente Balanced Random Forest-modellen og produserer både prediksjon og forklaring.

### Input

Modellen forventer følgende hovedparametere:
- antall avkjørsler
- stigning (bakke)
- ÅDT, total
- andel lange kjøretøy
- fartsgrense
- ev. kurvatur (sving)

Basert på disse konstrueres de interaksjonsleddene modellen er trent på.

---

### Output

Funksjonen `get_rf_prediction()` returnerer enten:
- kun sannsynlighet og klasse, eller
- et forklaringsobjekt med:
  - sannsynlighet i prosent
  - risikokategori
  - faktorer som øker sannsynligheten for avslag
  - faktorer som reduserer sannsynligheten for avslag
  - ferdig formatert forklaringstekst

---

### Forklaringsmekanisme (treeinterpreter)

Forklaringene er basert på `treeinterpreter`, som dekomponerer Random Forest-prediksjonen i:

- **bias** (grunnrisiko)
- bidrag fra hver enkelt feature-interaksjon

Tekniske feature-navn oversettes til lesbare beskrivelser via et eksplisitt `FEATURE_MAP`, for eksempel:

- `"ÅDT, total bakke"` →  
  *Total trafikkmengde per år × kurvatur, stigning*

Forklaringen brukes både:
- direkte i frontend
- som strukturert input til språkmodellen i steg 2

Dette gir:
- høy transparens
- etterprøvbar beslutningsstøtte
- bedre kontroll med språkmodellens begrunnelse

---

## Prinsipper for ansvarlig bruk

- Ingen enkeltmodell fatter vedtak alene
- ML brukes til risikovurdering og strukturering
- Språkmodellens forslag er rådgivende
- Resultatene skal kunne etterprøves i hvert steg

Arkitekturen er bevisst bygget for å kombinere:
> statistisk presisjon + forklarbarhet + helhetlig vurdering

---

## Oppsummering

Balanced Random Forest-modellen fungerer som et **første, kvantitativt risikosteg** i en flertrinns beslutningsstøttemodell.  
Ved å kombinere forklarbar maskinlæring, språkmodell og egen kompleksitetsklassifisering oppnås:

- bedre strukturerte vurderinger
- mer konsistente begrunnelser
- tydeligere prioritering av vanskelige saker

Modellen er utviklet for støtte og kvalitetssikring – ikke automatisert myndighetsutøvelse.

---
