import csv
import httpx

# Input and output file paths
input_file = 'data_avkjorsel.csv'
output_file = 'data_ulykker.csv'

# Define the API endpoint
url = "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/570"

# Read the input CSV and process each row
with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile, delimiter=';')

    # Read the header and append new columns
    header = next(reader)
    header.append("Trafikkulykker")
    writer.writerow(header)

    # Create an HTTP client
    with httpx.Client() as client:
        for row in reader:
            vegsystemreferanse = row[14]
            vsr_uten_m = vegsystemreferanse.split(" m")

            # Prepare API parameters
            params = {
                "vegsystemreferanse": vsr_uten_m[0],
                "inkluder": "lokasjon"
            }

            # Send GET request
            response = client.get(url, params=params)

            # Initialize default values
            ulykker= 0

            # If request is successful, extract data
            if response.status_code == 200:
                data = response.json()
                objekter = data.get("objekter", [])

                if objekter:
                    lokasjon2 = []
                    meter_liste = []
                    vsr_m = float((vsr_uten_m[1].split(" K"))[0])

                    for x in range(len(objekter) - 1):
                        lokasjon = objekter[x].get("lokasjon", {})
                        vegsystemreferanser = (lokasjon.get("vegsystemreferanser", []))
                        for element in vegsystemreferanser:
                            meter = element.get("strekning", {}).get("meter")
                            meter_liste.append(meter)
                    
                    for m in meter_liste:
                        if (abs(vsr_m - m) <= 500):
                            ulykker += 1

            print(ulykker)
            row.append(ulykker)
            writer.writerow(row)

print(f"Data with avkjørsler values has been written to {output_file}.")