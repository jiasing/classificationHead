import pandas as pd

INPUT  = "data/classification_processed/juliet_labelled.json"
OUTPUT = "data/classification_processed/juliet_labelled.csv"

df = pd.read_json(INPUT, lines=True)

print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

df.to_csv(OUTPUT, index=False)
print(f"Saved to {OUTPUT}")
