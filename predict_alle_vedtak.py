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


# Modellering
features = [
    "ÅDT, total", "ÅDT, andel lange kjøretøy", "Fartsgrense",
    "Avkjørsler", "Trafikkulykker",
    "EGS.BRUKSOMRÅDE.1256", "LOK.KOMMUNE",
    "Avkjørsel, holdningsklasse", "Funksjonsklasse"
]

df_model = df_plot[features + ["EGS.VEDTAK.10670"]].dropna()

# Label encoding
label_encoders = {}
for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Split data
X = df_model.drop("EGS.VEDTAK.10670", axis=1)
y = LabelEncoder().fit_transform(df_model["EGS.VEDTAK.10670"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fremgangsindikator for modelltrening
print("\nTrener Random Forest-modell...")
for i in tqdm(range(100), desc="Modelltrening", ncols=100):
    time.sleep(0.01)  # Simulerer fremgang
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluering
y_pred = model.predict(X_test)

# Add prediction results to test set
test_results = X_test.copy()
test_results[target_col] = y_test
predicted_labels = pd.Series(y_pred, index=test_results.index, name="Predicted")
test_results = test_results.join(predicted_labels)
test_results["Correct"] = test_results[target_col] == test_results["Predicted"]


# Split into correct and incorrect predictions
correct_mask = y_pred == y_test
X_correct = X_test[correct_mask]
X_incorrect = X_test[~correct_mask]
y_correct = y_test[correct_mask]
y_incorrect = y_test[~correct_mask]

# Create output folder
os.makedirs("predict_alle_vedtak", exist_ok=True)

# Plot distributions for numeric features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
for feature in numeric_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(X_correct[feature], color='green', label='Riktig predikert', kde=True, stat="density")
    sns.histplot(X_incorrect[feature], color='red', label='Feil predikert', kde=True, stat="density")
    plt.title(f"Distribusjon av '{feature}' for riktige vs feil prediksjoner")
    plt.xlabel(feature)
    plt.ylabel("Tetthet")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"predict_alle_vedtak/{feature}_distribution.png")
    plt.close()


# Summary statistics
summary_correct = X_correct.describe().T
summary_incorrect = X_incorrect.describe().T

# Top 5 differences
diff = (summary_correct["mean"] - summary_incorrect["mean"]).abs().sort_values(ascending=False)
top5_features = diff.head(5).index.tolist()
top5_summary = pd.DataFrame({
    "Feature": top5_features,
    "Mean (Riktig)": summary_correct.loc[top5_features, "mean"],
    "Mean (Feil)": summary_incorrect.loc[top5_features, "mean"],
    "Difference": diff.loc[top5_features]
})

# Save summary
top5_summary.to_csv("predict_alle_vedtak/top5_feature_differences.csv", index=False)

# Save full summaries
summary_correct.to_csv("predict_alle_vedtak/summary_correct.csv")
summary_incorrect.to_csv("predict_alle_vedtak/summary_incorrect.csv")

print("Analyse fullført. Grafer og oppsummeringstabeller er lagret i mappen 'predict_alle_vedtak'.")