
import pandas as pd

# Les CSV-filer
base = pd.read_csv('data_2022-2025.csv', sep=';', dtype=str)
ann  = pd.read_csv('data_annotert.csv', sep=';', dtype=str)
base.columns = [c.strip() for c in base.columns]
ann.columns  = [c.strip() for c in ann.columns]

# Nøkler (inkluderer kolonnen som mangler i annotasjon, men vi håndterer det)
keys = ['OBJ.VEGOBJEKT-ID', 'EGS.SAKSNUMMER.1822', 'EGS.SØKNAD MOTTATT DATO.12048']

# Sjekk hvilke nøkler som finnes i annotasjonsfila og base
keys_in_ann = [k for k in keys if k in ann.columns]
keys_in_base = [k for k in keys if k in base.columns]

# Bruk kun felles nøkler for matching
match_keys = list(set(keys_in_ann) & set(keys_in_base))
print(f"Bruker følgende nøkler for matching: {match_keys}")

# Lag annotasjons-subsett med felles nøkler + Vurdering
ann2 = ann[match_keys + ['Vurdering']].copy()
ann2 = (ann2.groupby(match_keys, as_index=False)
             .agg({'Vurdering': lambda s: s.dropna().iloc[-1] if s.dropna().size > 0 else None}))

# Merge
merged = base.merge(ann2, on=match_keys, how='left')

# Validering
annotated_total   = ann['Vurdering'].notna().sum()
transferred_total = merged['Vurdering'].notna().sum()
missing = annotated_total - transferred_total

print('Nøkler brukt for matching:', match_keys)
print(f"Antall 'Vurdering' rader i data_annotert.csv: {annotated_total}")
print(f"Antall 'Vurdering' overført til data_2022-2025.csv: {transferred_total}")
if missing == 0:
    print('✅ Alle vurdering-verdier er overført basert på valgt nøkkel.')
else:
    print(f"⚠️ {missing} vurdering-verdier i annotasjonen ble ikke matchet/overført med denne nøkkelen.")

# Finn annotasjonsrader som ikke ble matchet
base_keys_set = set(tuple(row) for row in base[match_keys].values)
ann_keys_set  = set(tuple(row) for row in ann2[match_keys].values)
missing_keys  = list(ann_keys_set - base_keys_set)

if missing_keys:
    print("\nAnnotasjonsrader som ikke ble matchet:")
    unmatched_rows = ann2[ann2.apply(lambda r: tuple(r[match_keys]) in missing_keys, axis=1)]
    print(unmatched_rows.to_csv(sep=';', index=False))

# Finn annotasjonsnøkler som matcher flere ganger i base
dup_counts = base.groupby(match_keys).size().reset_index(name='count')
duplicates = dup_counts[dup_counts['count'] > 1]

if not duplicates.empty:
    print("\nAnnotasjonsnøkler som matcher flere ganger i base:")
    print(duplicates.to_csv(sep=';', index=False))

    # Tilhørende annotasjonsrader
    dup_keys = [tuple(row) for row in duplicates[match_keys].values]
    dup_ann_rows = ann2[ann2.apply(lambda r: tuple(r[match_keys]) in dup_keys, axis=1)]
    print("\nAnnotasjonsrader for disse nøklene:")
    print(dup_ann_rows.to_csv(sep=';', index=False))

# Lagre resultatfil
out_path = 'data_2022-2025_med_vurdering_keys2.csv'
merged.to_csv(out_path, sep=';', index=False)
