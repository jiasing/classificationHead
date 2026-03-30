# src/build_labels.py
# This script encodes your CWE table into integer IDs and enriches your cleaned dataset:

import pandas as pd
import json
import os

# ── Define your full CWE → (category, error_type) mapping ────────────────────
# Derived directly from the table you provided

CWE_MAP = {
    # Memory Error — Null Pointer Dereference
    476: ("Memory Error", "Null Pointer Dereference"),
    588: ("Memory Error", "Null Pointer Dereference"),
    690: ("Memory Error", "Null Pointer Dereference"),
    # Memory Error — Incorrect Variable Usage
    457: ("Memory Error", "Incorrect Variable Usage"),
    665: ("Memory Error", "Incorrect Variable Usage"),
    # Memory Error — Memory Overflow
    121: ("Memory Error", "Memory Overflow"),
    122: ("Memory Error", "Memory Overflow"),
    123: ("Memory Error", "Memory Overflow"),
    124: ("Memory Error", "Memory Overflow"),
    126: ("Memory Error", "Memory Overflow"),
    127: ("Memory Error", "Memory Overflow"),
    190: ("Memory Error", "Memory Overflow"),
    191: ("Memory Error", "Memory Overflow"),
    226: ("Memory Error", "Memory Overflow"),
    244: ("Memory Error", "Memory Overflow"),
    401: ("Memory Error", "Memory Overflow"),
    415: ("Memory Error", "Memory Overflow"),
    416: ("Memory Error", "Memory Overflow"),
    562: ("Memory Error", "Memory Overflow"),
    590: ("Memory Error", "Memory Overflow"),
    591: ("Memory Error", "Memory Overflow"),
    680: ("Memory Error", "Memory Overflow"),
    761: ("Memory Error", "Memory Overflow"),
    762: ("Memory Error", "Memory Overflow"),
    785: ("Memory Error", "Memory Overflow"),
    # Memory Error — Uncontrolled Resource Consumption
    789: ("Memory Error", "Uncontrolled Resource Consumption"),
    # Logic Organization — Improper Condition Organization
    398: ("Logic Organization", "Improper Condition Organization"),
    440: ("Logic Organization", "Improper Condition Organization"),
    478: ("Logic Organization", "Improper Condition Organization"),
    483: ("Logic Organization", "Improper Condition Organization"),
    484: ("Logic Organization", "Improper Condition Organization"),
    506: ("Logic Organization", "Improper Condition Organization"),
    510: ("Logic Organization", "Improper Condition Organization"),
    511: ("Logic Organization", "Improper Condition Organization"),
    546: ("Logic Organization", "Improper Condition Organization"),
    561: ("Logic Organization", "Improper Condition Organization"),
    570: ("Logic Organization", "Improper Condition Organization"),
    571: ("Logic Organization", "Improper Condition Organization"),
    617: ("Logic Organization", "Improper Condition Organization"),
    # Logic Organization — Uncontrolled Resource Consumption
    400: ("Logic Organization", "Uncontrolled Resource Consumption"),
    674: ("Logic Organization", "Uncontrolled Resource Consumption"),
    835: ("Logic Organization", "Uncontrolled Resource Consumption"),
    # Logic Organization — Wrong Function Call Sequence
    325: ("Logic Organization", "Wrong Function Call Sequence"),
    364: ("Logic Organization", "Wrong Function Call Sequence"),
    366: ("Logic Organization", "Wrong Function Call Sequence"),
    367: ("Logic Organization", "Wrong Function Call Sequence"),
    404: ("Logic Organization", "Wrong Function Call Sequence"),
    459: ("Logic Organization", "Wrong Function Call Sequence"),
    605: ("Logic Organization", "Wrong Function Call Sequence"),
    666: ("Logic Organization", "Wrong Function Call Sequence"),
    667: ("Logic Organization", "Wrong Function Call Sequence"),
    672: ("Logic Organization", "Wrong Function Call Sequence"),
    675: ("Logic Organization", "Wrong Function Call Sequence"),
    773: ("Logic Organization", "Wrong Function Call Sequence"),
    775: ("Logic Organization", "Wrong Function Call Sequence"),
    832: ("Logic Organization", "Wrong Function Call Sequence"),
    # Sanitizer — Control Expression Error
    369: ("Sanitizer", "Control Expression Error"),
    606: ("Sanitizer", "Control Expression Error"),
    # Sanitizer — Fault Input Type
    15:  ("Sanitizer", "Fault Input Type"),
    23:  ("Sanitizer", "Fault Input Type"),
    36:  ("Sanitizer", "Fault Input Type"),
    78:  ("Sanitizer", "Fault Input Type"),
    90:  ("Sanitizer", "Fault Input Type"),
    114: ("Sanitizer", "Fault Input Type"),
    134: ("Sanitizer", "Fault Input Type"),
    176: ("Sanitizer", "Fault Input Type"),
    256: ("Sanitizer", "Fault Input Type"),
    272: ("Sanitizer", "Fault Input Type"),
    284: ("Sanitizer", "Fault Input Type"),
    319: ("Sanitizer", "Fault Input Type"),
    377: ("Sanitizer", "Fault Input Type"),
    426: ("Sanitizer", "Fault Input Type"),
    427: ("Sanitizer", "Fault Input Type"),
    475: ("Sanitizer", "Fault Input Type"),
    526: ("Sanitizer", "Fault Input Type"),
    534: ("Sanitizer", "Fault Input Type"),
    535: ("Sanitizer", "Fault Input Type"),
    615: ("Sanitizer", "Fault Input Type"),
    620: ("Sanitizer", "Fault Input Type"),
    # Sanitizer — Incorrect Variable Usage
    194: ("Sanitizer", "Incorrect Variable Usage"),
    195: ("Sanitizer", "Incorrect Variable Usage"),
    196: ("Sanitizer", "Incorrect Variable Usage"),
    197: ("Sanitizer", "Incorrect Variable Usage"),
    222: ("Sanitizer", "Incorrect Variable Usage"),
    223: ("Sanitizer", "Incorrect Variable Usage"),
    681: ("Sanitizer", "Incorrect Variable Usage"),
    843: ("Sanitizer", "Incorrect Variable Usage"),
    # Sanitizer — Incorrect Function Return Value (CWE247 is listed under Sanitizer)
    247: ("Sanitizer", "Incorrect Function Return Value"),
    # Signature — Incorrect Function Return Value
    242: ("Signature", "Incorrect Function Return Value"),
    252: ("Signature", "Incorrect Function Return Value"),
    253: ("Signature", "Incorrect Function Return Value"),
    273: ("Signature", "Incorrect Function Return Value"),
    327: ("Signature", "Incorrect Function Return Value"),
    328: ("Signature", "Incorrect Function Return Value"),
    338: ("Signature", "Incorrect Function Return Value"),
    390: ("Signature", "Incorrect Function Return Value"),
    391: ("Signature", "Incorrect Function Return Value"),
    396: ("Signature", "Incorrect Function Return Value"),
    397: ("Signature", "Incorrect Function Return Value"),
    467: ("Signature", "Incorrect Function Return Value"),
    479: ("Signature", "Incorrect Function Return Value"),
    676: ("Signature", "Incorrect Function Return Value"),
    685: ("Signature", "Incorrect Function Return Value"),
    688: ("Signature", "Incorrect Function Return Value"),
    780: ("Signature", "Incorrect Function Return Value"),
    # Signature — Incorrect Variable Usage
    188: ("Signature", "Incorrect Variable Usage"),
    259: ("Signature", "Incorrect Variable Usage"),
    321: ("Signature", "Incorrect Variable Usage"),
    464: ("Signature", "Incorrect Variable Usage"),
    468: ("Signature", "Incorrect Variable Usage"),
    469: ("Signature", "Incorrect Variable Usage"),
    480: ("Signature", "Incorrect Variable Usage"),
    481: ("Signature", "Incorrect Variable Usage"),
    482: ("Signature", "Incorrect Variable Usage"),
    500: ("Signature", "Incorrect Variable Usage"),
    563: ("Signature", "Incorrect Variable Usage"),
    587: ("Signature", "Incorrect Variable Usage"),
    758: ("Signature", "Incorrect Variable Usage"),
}

# ── Build integer encoders for both label levels ──────────────────────────────
# Collect all unique values in order
all_categories  = ["Clean"] + sorted(set(v[0] for v in CWE_MAP.values()))
all_error_types = ["Clean"] + sorted(set(v[1] for v in CWE_MAP.values()))

# Create the categories labelling. ie, cat_id = 0 means "Clean", cat_id = 1 means "Logic Organization".
# Found in label_maps.json
CATEGORY_TO_ID  = {cat:  i for i, cat  in enumerate(all_categories)}
# Create the error labelling. ie, error_id = 0 means "Clean", error_id = 5 means "Incorrect Variable Usage". 
# Found in label_maps.json
ERROR_TYPE_TO_ID = {et:  i for i, et   in enumerate(all_error_types)}

# Save these mappings — you'll need them again at inference time
os.makedirs("data/classification_processed", exist_ok=True)
with open("data/classification_processed/label_maps.json", "w") as f:
    json.dump({
        "category_to_id":   CATEGORY_TO_ID,
        "error_type_to_id": ERROR_TYPE_TO_ID,
        "id_to_category":   {str(v): k for k, v in CATEGORY_TO_ID.items()},
        "id_to_error_type": {str(v): k for k, v in ERROR_TYPE_TO_ID.items()},
    }, f, indent=2)
print("Saved label_maps.json")
print(f"Categories  ({len(CATEGORY_TO_ID)}): {list(CATEGORY_TO_ID.keys())}")
print(f"Error types ({len(ERROR_TYPE_TO_ID)}): {list(ERROR_TYPE_TO_ID.keys())}")


def apply_labels(df):
    """
    Takes juliet_cleaned.json and adds category_id and error_type_id columns.
    label=0 (clean) functions get the 'Clean' class for both levels.
    label=1 (vulnerable) functions get their CWE-derived labels.
    """
    category_ids  = []
    error_type_ids = []
    skipped = 0

    for _, row in df.iterrows():
        if row['label'] == 0:
            # Safe/clean function — no vulnerability category
            category_ids.append(CATEGORY_TO_ID["Clean"])
            error_type_ids.append(ERROR_TYPE_TO_ID["Clean"])
        else:
            cwe = int(row['cwe']) if pd.notna(row['cwe']) else None
            if cwe and cwe in CWE_MAP:
                cat, err = CWE_MAP[cwe]
                category_ids.append(CATEGORY_TO_ID[cat])
                error_type_ids.append(ERROR_TYPE_TO_ID[err])
            else:
                # CWE not in our mapping table — mark as Clean and flag
                category_ids.append(CATEGORY_TO_ID["Clean"])
                error_type_ids.append(ERROR_TYPE_TO_ID["Clean"])
                skipped += 1

    df = df.copy()
    df['category_id']   = category_ids
    df['error_type_id'] = error_type_ids

    if skipped:
        print(f"Warning: {skipped} rows had CWEs not in our mapping — labelled as Clean")

    return df


if __name__ == "__main__":
    # Load the cleaned data into df
    df = pd.read_json("data/classification_processed/juliet_cleaned.json", lines=True)
    print(f"Loaded {len(df)} functions")

    # Add category_id and error_type_id columns. 0 (clean), 1(dirty).
    df = apply_labels(df)

    print(f"\nCategory distribution:\n{df['category_id'].value_counts()}")
    print(f"\nError type distribution:\n{df['error_type_id'].value_counts()}")

    df.to_json("data/classification_processed/juliet_labelled.json", orient='records', lines=True)
    print("\nSaved to data/classification_processed/juliet_labelled.json")