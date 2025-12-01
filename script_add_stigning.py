import csv
import httpx

# Input and output file paths
input_file = 'data_annotert.csv'
output_file = 'data_annotert2.csv'

# Define the API endpoint
url = "https://nvdbapiles.atlas.vegvesen.no/vegobjekter/api/v4/vegobjekter/825"

# Read the input CSV and process each row
with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile, delimiter=';')

    # Read the header and append new columns
    header = next(reader)
    header.append("Kurvatur, stigning")
    writer.writerow(header)

    # Create an HTTP client
    with httpx.Client() as client:
        for row in reader:
            vegsystemreferanse = row[14]

            # Prepare API parameters
            params = {
                "vegsystemreferanse": vegsystemreferanse,
                "inkluder": "egenskaper"
            }

            # Send GET request
            response = client.get(url, params=params)

            # Initialize default values
            kurvatur = ""

            # If request is successful, extract data
            if response.status_code == 200:
                data = response.json()
                objekter = data.get("objekter", [])

                if objekter:
                    egenskaper = objekter[0].get("egenskaper", [])
                    kurvatur = next((e["verdi"] for e in egenskaper if e["navn"] == "Stigning"), "")

            # Append the extracted values to the row and write to output
            row.append(kurvatur)
            writer.writerow(row)

print(f"Data with kurvatur values has been written to {output_file}.")