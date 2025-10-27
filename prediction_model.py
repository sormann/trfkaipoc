import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data_2022-2025.csv", sep=";")

# Drop rows with missing vedtak
df = df[df["EGS.VEDTAK.10670"].notna()]

# Encode vedtak types
le = LabelEncoder()
df["vedtak_encoded"] = le.fit_transform(df["EGS.VEDTAK.10670"])

# Define favorable vedtak types
favorable_types = ["Utvidet bruk", "Varig tillatelse"]

# Feature selection (expand as needed)
features = [
    "ÅDT, total",
    "ÅDT, andel lange kjøretøy",
    "Fartsgrense",
    "Avkjørsel, holdningsklasse",
    "Funksjonsklasse",
    "Avkjørsler",
    "Trafikkulykker",
    "EGS.BRUKSOMRÅDE.1256", 
    "Kurvatur, horisontal", 
    "Kurvatur, stigning"
]

# Encode categorical features
df_encoded = pd.get_dummies(df[features])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df_encoded, df["vedtak_encoded"], test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
probs = model.predict_proba(X_test)
pred_indices = probs.argmax(axis=1)
pred_labels = le.inverse_transform(pred_indices)
pred_confidences = probs.max(axis=1)

# Decision logic
def classify(pred_label, confidence, threshold=0.8):
    if pred_label in favorable_types and confidence >= threshold:
        return "AUTO_APPROVE"
    else:
        return "MANUAL_REVIEW"

# Apply logic
results = pd.DataFrame({
    "VEGOBJEKT_ID": df.loc[X_test.index, "OBJ.VEGOBJEKT-ID"].values,
    "predicted_vedtak": pred_labels,
    "confidence": pred_confidences,
    "decision": [classify(label, conf, threshold=0.8) for label, conf in zip(pred_labels, pred_confidences)]
})
results.to_csv("vedtak_predictions.csv", index=False, sep=";")

# Define threshold
threshold = 0.8

# Filter predictions above threshold
high_conf_mask = pred_confidences >= threshold
high_conf_preds = pred_indices[high_conf_mask]
high_conf_true = y_test[high_conf_mask]

# Print evaluation for high-confidence predictions
labels = sorted(y_test.unique())  # classes in test set
if len(high_conf_preds) > 0:
    print("\n📊 Evaluation for predictions with confidence ≥ {:.0f}%:".format(threshold * 100))
    print(classification_report(high_conf_true, high_conf_preds, labels=labels, target_names=[le.classes_[i] for i in labels]))
else:
    print("\n⚠️ No predictions above the confidence threshold of {:.0f}%.".format(threshold * 100))
