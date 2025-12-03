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
Steg 2 og 3 er dokumentert i sine filer

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

## Vurdering av treningslogg og modellresultater  
*(Treningskjøring 2025-11-14 kl. 13:27)*

Denne seksjonen dokumenterer og vurderer en konkret treningskjøring av modellen, basert på automatisk generert treningslogg.

### Treningsinformasjon

- **Tidspunkt for trening:** 2025-11-14 13:27:57  
- **Lagret modellfil:** `balanced_rf_model_20251114_132757.pkl.gz`  
- **Modell:** Balanced Random Forest  
- **Bruksområde:** Steg 1 i fler-stegs beslutningsstøttesystem

Treningslogg, modellfil og feature-utvalg er lagret for sporbarhet og etterprøvbarhet.

---

### Vurdering av identifiserte topp-features

De 10 viktigste forklaringsvariablene består i hovedsak av **interaksjoner mellom trafikkmengde, tunge kjøretøy, terreng og lokale vegforhold**:

Dette er **faglig konsistent** med kjente risikofaktorer i veiforvaltning og indikerer at modellen har lært **meningsfulle strukturer**, ikke tilfeldige korrelasjoner.

At interaksjonsledd dominerer, er forventet gitt:
- eksplisitt bruk av polynomiske interaksjoner
- kompleks, ikke-lineær problemstilling

---

### Vurdering av klassifikasjonsresultater

Figuren viser fordelingen av modellens predikerte sannsynlighet for avslag, fordelt på:

- **Faktisk klasse 0 (ikke avslag)**  
- **Faktisk klasse 1 (avslag)**  

Kurvene representerer tetthet (sannsynlighetsfordeling), ikke absolutte tall.

---

### Tolkning av fordelingen

Figuren viser en **tydelig systematisk forskjell** mellom de to klassene:

- Saker som faktisk ender med **avslag (klasse 1)** har gjennomgående:
  - høyere predikert sannsynlighet
  - tyngdepunkt i området ca. 0,3–0,6
- Saker som faktisk **ikke får avslag (klasse 0)**:
  - dominerer i lavere sannsynligheter
  - har hovedtyngde under ca. 0,2–0,3

Det er betydelig overlapp, noe som er forventet gitt:
- komplekse faglige vurderinger
- begrenset datagrunnlag for avslag
- at enkelte avslag avgjøres av forhold som ikke er fullt observerbare i data

![alt text](image.png)

---

### Samlet vurdering

Treningsloggen indikerer at:

- Modellen er stabil og faglig rimelig
- De viktigste variablene er forståelige og konsistente
- Resultatene er godt tilpasset rollen som:
  > kvantitativ risikoskår i steg 1

Begrensningene er tydelige og kjente, og er eksplisitt håndtert gjennom:
- forklarbarhet (treeinterpreter)
- etterfølgende språkmodellvurdering
- egen modell for sakskompleksitet

Modellen vurderes som **egnet for operativ bruk innenfor den beskrevne arkitekturen**, under forutsetning av at den brukes rådgivende og sammen med øvrige trinn i beslutningsstøttesystemet.


---
