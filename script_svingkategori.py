import pandas as pd
import numpy as np
import csv

# Load the dataset
df = pd.read_csv("data_2022-2025.csv", sep=";")

# Define the categorization function
def kategoriser_sving(radius):
    try:
        radius = abs(float(radius))
        if radius >= 3000:
            return f"Rett veg"
        elif radius >= 1000:
            return f"Veldig slak sving (svingradius {radius}m)"
        elif radius >= 500:
            return f"Slak sving (svingradius {radius}m)"
        elif radius >= 0:
            return "Krapp sving"
    except:
        pass
    return "Ukjent"

# Apply the function to create the new column
df["Svingkategori"] = df["Kurvatur, horisontal"].apply(kategoriser_sving)

# Count the number of each category
sving_telling = df["Svingkategori"].value_counts()

# Print the counts
print("Opptelling av svingkategorier:")
print(sving_telling)

# Save the updated dataset
df.to_csv("data_2022-2025.csv", sep=";", index=False, quoting=csv.QUOTE_NONNUMERIC)
