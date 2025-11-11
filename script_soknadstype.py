import pandas as pd
import csv

# Read the CSV file
df = pd.read_csv("data_2022-2025.csv", sep=";")


# Start med å kopiere vedtaket inn i Søknadstype
df["Søknadstype"] = df["EGS.VEDTAK.10670"]

# Sett Søknadstype til tom streng for alle med Avslag
df.loc[df["Søknadstype"] == "Avslag", "Søknadstype"] = ""


###### DENNE MÅ FYLLES INN MANUELT!!!!!!!!!!!!!! #######
# 1023139482;Gardsbruk;;05.05.2025;;Avslag;28.05.2025;;;;;POINT(299390.433 7042211.376);Stjørdal;5035;FV6798 S1D1 m2717;1400;9;30;Lite streng;E - Lokale adkomstveger;25;1;99999;-1.7
#= Flytting av eksisterende avkjørsel ("Endret bruk")

# Mapping of URLs to correct soknadstype values
url_to_soknadstype = {
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=617065#": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=616964#": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=575634": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=585681": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=605220": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=587126": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Document/Details/Simplified/2?module=Document&subtype=2&recno=2066252&VerID=1887714": "Utvidet bruk",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=458041": "Utvidet bruk",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=623128#": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=589138": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Document/Details/Simplified/2?module=Document&subtype=2&recno=3521066&VerID=3317826": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Document/Details/Simplified/2?module=Document&subtype=2&recno=2129335&VerID=1947298": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=608023": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Document/Details/Simplified/2?module=Document&subtype=2&recno=2354955&VerID=2160461": "Varig tillatelse",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=480350": "Utvidet bruk",
    "https://trondelag.public360online.com/locator/DMS/Document/Details/Simplified/2?module=Document&subtype=2&recno=2127666&VerID=1945739": "Utvidet bruk",
    "https://trondelag.public360online.com/locator/DMS/Case/Details/Simplified/2?module=Case&subtype=2&recno=579158": "Endret bruk",
    "https://trondelag.public360online.com/locator/DMS/Document/Details/Simplified/2?module=Document&subtype=2&recno=2086796&VerID=1907746": "Utvidet bruk",
}

# Oppdater soknadstype kun for rader med Avslag og matchende URL
df.loc[
    (df["EGS.VEDTAK.10670"] == "Avslag") &
    (df["EGS.ARKIVREFERANSE, URL.12050"].isin(url_to_soknadstype)),
    "soknadstype"
] = df["EGS.ARKIVREFERANSE, URL.12050"].map(url_to_soknadstype)

# Lagre til ny fil
df.to_csv("data_2022-2025.csv", sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)
