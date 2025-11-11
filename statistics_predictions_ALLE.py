import pandas as pd

# Load the dataset
file_path = "prediksjoner_test_dynamisk.csv"
df = pd.read_csv(file_path)

# Define the four confusion matrix categories
conditions = {
    "Godkjent=Godkjent": (df["VEDTAK_BINÆR"] == "Godkjent") & (df["Prediksjon"] == "Godkjent"),
    "Godkjent=Avslag": (df["VEDTAK_BINÆR"] == "Godkjent") & (df["Prediksjon"] == "Avslag"),
    "Avslag=Godkjent": (df["VEDTAK_BINÆR"] == "Avslag") & (df["Prediksjon"] == "Godkjent"),
    "Avslag=Avslag": (df["VEDTAK_BINÆR"] == "Avslag") & (df["Prediksjon"] == "Avslag")
}

# Define prediction-based categories
prediction_categories = {
    "Predikert Godkjent": df["Prediksjon"] == "Godkjent",
    "Predikert Avslag": df["Prediksjon"] == "Avslag"
}

# Select relevant features for analysis
features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense",
    "Avkjørsler", "Trafikkulykker", "Kurvatur, horisontal", "Kurvatur, stigning",
    "Avkjørsel, holdningsklasse", "Funksjonsklasse"
]

# Analyze each category
summary_stats = {}

# Confusion matrix categories
for label, condition in conditions.items():
    subset = df[condition]
    stats = subset[features].describe(include="all")
    søknadstype_counts = subset["Søknadstype"].value_counts()
    summary_stats[label] = {
        "Feature Stats": stats,
        "Søknadstype Counts": søknadstype_counts
    }

# Prediction categories + Søknadstype count
for label, condition in prediction_categories.items():
    subset = df[condition]
    stats = subset[features].describe(include="all")
    søknadstype_counts = subset["Søknadstype"].value_counts()
    summary_stats[label] = {
        "Feature Stats": stats,
        "Søknadstype Counts": søknadstype_counts
    }

# Display summary statistics for each category
for category, content in summary_stats.items():
    print(f"\nKategori: {category}\n")
    if isinstance(content, dict):
        print("Feature Stats:\n", content["Feature Stats"])
        print("\nSøknadstype Counts:\n", content["Søknadstype Counts"])
    else:
        print(content)