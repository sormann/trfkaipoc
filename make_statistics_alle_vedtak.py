import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Load data
df_plot = pd.read_csv("data_2022-2025.csv", sep=";", encoding="utf-8")

# Define target and features
target_col = "EGS.VEDTAK.10670"
features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense",
    "Avkjørsler", "Trafikkulykker",
    "EGS.BRUKSOMRÅDE.1256", "LOK.KOMMUNE",
    "Avkjørsel, holdningsklasse", "Funksjonsklasse"
]

# Encode target variable
target_encoder = LabelEncoder()
df_plot[target_col] = target_encoder.fit_transform(df_plot[target_col])

# Lag mapping basert på encoder
print(target_encoder.classes_)

# Encode categorical variables
label_encoders = {}
for col in df_plot.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_plot[col] = le.fit_transform(df_plot[col])
    label_encoders[col] = le


target_mapping = dict(zip(range(len(target_encoder.classes_)), target_encoder.classes_))
print("Mapping:", target_mapping)

# Lag plott én og én
os.makedirs("statistics_all", exist_ok=True)

for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target_col, y=feature, data=df_plot)
    
    # Dynamisk mapping-tekst
    mapping_text = ", ".join([f"{k}={v}" for k, v in target_mapping.items()])
    print(mapping_text)
    
    plt.xlabel(f"{target_col} ({mapping_text})")
    plt.ylabel(feature)
    plt.title(f"Fordeling av '{feature}' per vedtakstype")
    plt.tight_layout()
    filename = f"statistics_all/{feature.replace(',', '').replace(' ', '_')}_vs_vedtak.png"
    plt.savefig(filename)
    plt.close()


# Krysstabeller med prosentfordeling
categorical_vars = [
    "EGS.BRUKSOMRÅDE.1256", "LOK.KOMMUNE",
    "Avkjørsel, holdningsklasse", "Funksjonsklasse"
]

for var in categorical_vars:
    print(f"\nProsentvis fordeling av vedtakstyper innen '{var}':")
    crosstab = pd.crosstab(df_plot[var], df_plot["EGS.VEDTAK.10670"], normalize='index') * 100
    print(crosstab.round(1))

# Visualiseringer
sns.set_theme(style="whitegrid")
for var in categorical_vars:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_plot, x=var, hue="EGS.VEDTAK.10670", multiple="fill", shrink=0.8)
    plt.title(f"Relativ fordeling av vedtakstyper etter '{var}'")
    plt.xticks(rotation=45)
    filename = f"statistics_all/{var.replace(',', '').replace(' ', '_')}_vs_vedtak.png"
    plt.savefig(filename)
    plt.close()

print("Alle individuelle plott er lagret i mappen 'statistics_all'.")
