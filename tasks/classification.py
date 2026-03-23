# src/preprocess.py

import os
import re
import json
import pandas as pd
from tqdm import tqdm  # shows a progress bar — very useful for large datasets

# ── STEP 1: Configuration ──────────────────────────────────────────────────────
# Change JULIET_ROOT to wherever you put the Juliet folder on your laptop.
# The output will be saved into data/processed/

JULIET_ROOT = "/Users/jiasing/Juliet"          # adjust this path
OUTPUT_DIR  = "/Users/jiasing/Juliet/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "juliet_functions.csv")


# ── STEP 2: Extract CWE number from a folder or filename ──────────────────────
# Juliet names everything like "CWE121_Stack_Buffer_Overflow"
# We want just the number: 121

def extract_cwe(text):
    """Pull the CWE number out of a string like 'CWE121_Stack_Overflow'."""
    match = re.search(r'CWE(\d+)', text)
    if match:
        return int(match.group(1))
    return None


# ── STEP 3: Determine if a file is 'bad' (vulnerable) or 'good' (safe) ────────
# Juliet uses _bad and _good in filenames, but also _a, _b, _c etc. as helpers.
# Files ending in _bad.c are always vulnerable.
# Files ending in _good.c are always safe.
# Helper files (_a.c, _b.c, main.cpp) should be skipped — they're not labelled.

def get_label(filename):
    """
    Returns:
        1  if the file contains vulnerable (bad) code
        0  if the file contains safe (good) code
        None if we should skip this file
    """
    name = filename.lower()
    if '_bad' in name:
        return 1
    elif '_good' in name:
        return 0
    else:
        return None  # skip helper files


# ── STEP 4: Extract individual functions from a C file ────────────────────────
# This uses a simple heuristic: find blocks of text that look like functions.
# It's not a full C parser — for a first pass this is fine.
# Later you can swap in a proper parser like tree-sitter if needed.

def extract_functions(source_code):
    """
    Very roughly splits a C file into individual function bodies.
    Returns a list of strings, each being one function's source text.
    """
    functions = []

    # Strategy: find every '{' ... '}' block that's preceded by something
    # that looks like a function signature (return type + name + parentheses).
    # We track brace depth to handle nested braces correctly.

    # First, strip single-line // comments and block /* */ comments
    # so they don't confuse our brace counting
    source_code = re.sub(r'//[^\n]*', '', source_code)
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)

    # A loose pattern for a C function signature:
    # e.g.  void bad(void)  or  static int goodB2(char *data)
    func_pattern = re.compile(
        r'(\w[\w\s\*]+\s+\w+\s*\([^;{]*\))\s*\{',
        re.DOTALL
    )

    for match in func_pattern.finditer(source_code):
        start = match.start()
        brace_start = match.end() - 1  # position of the opening '{'

        # Now walk forward counting braces to find the matching closing '}'
        depth = 0
        i = brace_start
        while i < len(source_code):
            if source_code[i] == '{':
                depth += 1
            elif source_code[i] == '}':
                depth -= 1
                if depth == 0:
                    # Found the matching closing brace
                    func_body = source_code[start:i+1]
                    functions.append(func_body.strip())
                    break
            i += 1

    return functions


# ── STEP 5: Walk the entire Juliet directory tree ─────────────────────────────
# os.walk() visits every subfolder recursively and gives us the list of files
# in each folder. We check each file, skip ones we don't want, and process
# the ones we do.

def process_juliet(juliet_root):
    """
    Walks the Juliet dataset directory and returns a list of records.
    Each record is a dict with: cwe, label, filename, function_code
    """
    records = []
    skipped = 0

    # Collect all .c and .cpp files first so tqdm can show total progress
    all_files = []
    for dirpath, dirnames, filenames in os.walk(juliet_root):
        for fname in filenames:
            if fname.endswith(('.c', '.cpp')):
                all_files.append((dirpath, fname))

    print(f"Found {len(all_files)} C/C++ files. Processing...")

    for dirpath, fname in tqdm(all_files):
        # Determine label from filename
        label = get_label(fname)
        if label is None:
            skipped += 1
            continue  # skip helper/main files

        # Extract CWE from the directory path (more reliable than filename)
        cwe = extract_cwe(dirpath)
        if cwe is None:
            cwe = extract_cwe(fname)  # fallback: try the filename

        # Read the file
        filepath = os.path.join(dirpath, fname)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except Exception as e:
            print(f"Could not read {filepath}: {e}")
            skipped += 1
            continue

        # Extract functions from the source
        functions = extract_functions(source)

        if not functions:
            # If no functions found, store the whole file as one entry
            # (some Juliet files are structured unusually)
            records.append({
                'cwe':           cwe,
                'label':         label,
                'filename':      fname,
                'function_code': source[:5000],  # cap at 5000 chars
                'num_functions': 0
            })
        else:
            for func in functions:
                records.append({
                    'cwe':           cwe,
                    'label':         label,
                    'filename':      fname,
                    'function_code': func,
                    'num_functions': len(functions)
                })

    print(f"Skipped {skipped} files (helpers/main/unreadable).")
    return records


# ── STEP 6: Save results ───────────────────────────────────────────────────────

def save_results(records, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.DataFrame(records)

    # Basic sanity check — print a summary before saving
    print(f"\nTotal function samples: {len(df)}")
    print(f"Vulnerable (label=1):   {(df['label'] == 1).sum()}")
    print(f"Safe (label=0):         {(df['label'] == 0).sum()}")
    print(f"Unique CWE types:       {df['cwe'].nunique()}")
    print(f"CWE distribution (top 10):\n{df['cwe'].value_counts().head(10)}")

    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Also save as JSON (useful for later model training)
    json_file = output_file.replace('.csv', '.json')
    df.to_json(json_file, orient='records', lines=True)
    print(f"Saved to {json_file}")


# ── STEP 7: Main entry point ───────────────────────────────────────────────────
# This block only runs when you execute the file directly:
#   python src/preprocess.py
# It does NOT run when another script imports functions from this file.

if __name__ == "__main__":
    records = process_juliet(JULIET_ROOT)
    save_results(records, OUTPUT_FILE)