# This code is to clean the csv and json generated for Juliet

import re
import pandas as pd
import os

def clean_function(code):
    # Remove block comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove line comments
    code = re.sub(r'//[^\n]*', '', code)
    # Remove Juliet hint markers that leak the label
    code = re.sub(r'/\*\s*(FLAW|FIX|incl)\s*\*/', '', code, flags=re.IGNORECASE)
    # Remove label-leaking substrings wherever they appear
    code = re.sub(r'good', '', code, flags=re.IGNORECASE)
    code = re.sub(r'bad', '', code, flags=re.IGNORECASE)
    # Collapse excess blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()

# Load your parsed output
df = pd.read_json("data/classification_processed/1.preprocess - juliet_functions.json", lines=True)

print(f"Before cleaning: {len(df)} rows")

# Apply cleaning to every function
df['function_code'] = df['function_code'].apply(clean_function)
df['filename'] = df['filename'].apply(clean_function)

# Drop any rows where cleaning left an empty string
df = df[df['function_code'].str.strip().str.len() > 0]

print(f"After cleaning: {len(df)} rows")

# Save as a new file — always preserve the previous stage
os.makedirs("data/classification_processed", exist_ok=True)
df.to_json("data/classification_processed/2.juliet_cleaned.json", orient='records', lines=True)
print("Saved to data/classification_processed/2.juliet_cleaned.json")


# TODO: Come back to variable name normalisation with simple lexer or a tool like `srcml` or `tree-sitter` to reliably identify which tokens are user-defined names vs keywords vs types. 
# We skip this for now as this is a first pass.
