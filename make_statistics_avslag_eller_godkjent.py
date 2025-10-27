import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("data_2022-2025.csv", sep=";", encoding="utf-8")

# Create binary target column
df["VEDTAK_BINÆR"] = df["EGS.VEDTAK.10670"].apply(lambda x: "Avslag" if x == "Avslag" else "Godkjent")

# Split into two groups
df_avslag = df[df["VEDTAK_BINÆR"] == "Avslag"]
df_godkjent = df[df["VEDTAK_BINÆR"] == "Godkjent"]

# Create output folder
os.makedirs("statistics_binary", exist_ok=True)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Drop columns with too many missing values
df_avslag = df_avslag.dropna(axis=1, thresh=10)
df_godkjent = df_godkjent.dropna(axis=1, thresh=10)

# Summary statistics for numeric columns
summary_avslag = df_avslag[numeric_cols].describe().T
summary_godkjent = df_godkjent[numeric_cols].describe().T

summary_avslag.to_csv("statistics_binary/statistikk_avslag.csv")
summary_godkjent.to_csv("statistics_binary/statistikk_godkjent.csv")

# Plot numeric comparisons
for col in numeric_cols:
    if col in df_avslag.columns and col in df_godkjent.columns:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df_avslag[col], label="Avslag", fill=True, color="red")
        sns.kdeplot(df_godkjent[col], label="Godkjent", fill=True, color="green")
        plt.title(f"Sammenligning av '{col}' mellom Avslag og Godkjent")
        plt.xlabel(col)
        plt.ylabel("Tetthet")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"statistics_binary/{col}_sammenligning.png")
        plt.close()

# Frequency tables for categorical columns
for col in categorical_cols:
    if col in df_avslag.columns and col in df_godkjent.columns:
        freq_avslag = df_avslag[col].value_counts(normalize=True).rename("Avslag")
        freq_godkjent = df_godkjent[col].value_counts(normalize=True).rename("Godkjent")
        freq_df = pd.concat([freq_avslag, freq_godkjent], axis=1).fillna(0)
        freq_df.to_csv(f"statistics_binary/frekvens_{col}.csv")

        # Plot
        freq_df.plot(kind="bar", figsize=(10, 5), color=["red", "green"])
        plt.title(f"Frekvensfordeling av '{col}'")
        plt.xlabel(col)
        plt.ylabel("Andel")
        plt.tight_layout()
        plt.savefig(f"statistics_binary/frekvens_{col}_plot.png")
        plt.close()

print("Statistikk og grafer er lagret i mappen 'statistics_binary'.")


from sklearn.preprocessing import LabelEncoder

# Kopier dataframe
df_encoded = df[numeric_cols].copy()

# Label-encode kategoriske kolonner
for col in categorical_cols:
    if df[col].nunique() <= 50:  # Begrens til lav-kardinalitet for å unngå støy
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))

# Fjern rader med manglende verdier
df_encoded = df_encoded.dropna()

# Beregn korrelasjonsmatrise
corr_matrix = df_encoded.corr()

# Lagre som CSV
os.makedirs("statistics_binary", exist_ok=True)
corr_matrix.to_csv("statistics_binary/korrelasjonsmatrise.csv")

# Tegn heatmap uten annot=True for ytelse
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", square=True)
plt.title("Korrelasjonsmatrise mellom faktorer og vedtak")
plt.tight_layout()
plt.savefig("statistics_binary/korrelasjonsmatrise.png")
plt.close()

print("Korrelasjonsmatrise lagret som CSV og PNG i mappen 'statistics_binary'.")
