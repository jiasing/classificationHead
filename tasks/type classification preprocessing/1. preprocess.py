# src/preprocess.py

import os
import re
import json
import pandas as pd
from tqdm import tqdm

# ── STEP 1: Configuration ──────────────────────────────────────────────────────

JULIET_ROOT = "/Users/jiasing/Juliet"
OUTPUT_DIR  = "data/classification_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "juliet_functions.csv")


# ── STEP 2: Extract CWE number from a folder or filename ──────────────────────

def extract_cwe(text):
    """Pull the CWE number out of a string like 'CWE121_Stack_Overflow'."""
    match = re.search(r'CWE(\d+)', text)
    if match:
        return int(match.group(1))
    return None


# ── STEP 3: Split file into OMITBAD / OMITGOOD blocks ─────────────────────────
# Each Juliet file contains BOTH vulnerable and safe code in the same file,
# separated by #ifndef OMITBAD and #ifndef OMITGOOD preprocessor blocks.
# We must split on these boundaries and label each block independently.
# Nested #ifdef/#ifndef inside each block (e.g. #ifdef _WIN32) are tracked
# with a depth counter so we don't mistake them for the closing #endif.

def split_omit_blocks(source_code):
    """
    Returns a list of (label, code) tuples:
        label=1  for #ifndef OMITBAD  blocks  (vulnerable code)
        label=0  for #ifndef OMITGOOD blocks  (safe code)
    The #ifndef / #endif wrapper lines are not included in the returned code.
    """
    results = []

    lines = source_code.splitlines()

    in_block       = False
    current_label  = None
    current_lines  = []
    depth          = 0   # depth of nested #if / #ifdef / #ifndef

    omitbad_re  = re.compile(r'^\s*#\s*ifndef\s+OMITBAD\b')
    omitgood_re = re.compile(r'^\s*#\s*ifndef\s+OMITGOOD\b')
    any_if_re   = re.compile(r'^\s*#\s*(if|ifdef|ifndef)\b')
    endif_re    = re.compile(r'^\s*#\s*endif\b')

    for line in lines:
        if not in_block:
            if omitbad_re.match(line):
                in_block      = True
                current_label = 1
                current_lines = []
                depth         = 1
            elif omitgood_re.match(line):
                in_block      = True
                current_label = 0
                current_lines = []
                depth         = 1
            # lines outside any OMIT block are discarded (includes #include,
            # global vars, the main() wrapper, etc.)
        else:
            if any_if_re.match(line):
                depth += 1
                current_lines.append(line)
            elif endif_re.match(line):
                depth -= 1
                if depth == 0:
                    # Reached the closing #endif for this OMIT block
                    results.append((current_label, '\n'.join(current_lines)))
                    in_block      = False
                    current_label = None
                    current_lines = []
                else:
                    # This #endif belongs to a nested block — keep it
                    current_lines.append(line)
            else:
                current_lines.append(line)

    return results


# ── STEP 4: Extract individual functions from a code block ────────────────────
# Simple heuristic using regex + brace-counting.
# Comments are stripped first so they don't confuse brace counting.

def extract_functions(code):
    """
    Splits a block of C code into individual function bodies.
    Returns a list of strings, each being one function's source text.
    """
    # Strip comments before brace counting
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    functions = []

    func_pattern = re.compile(
        r'(\w[\w\s\*]+\s+\w+\s*\([^;{]*\))\s*\{',
        re.DOTALL
    )

    for match in func_pattern.finditer(code):
        start      = match.start()
        brace_pos  = match.end() - 1

        depth = 0
        i = brace_pos
        while i < len(code):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    functions.append(code[start:i+1].strip())
                    break
            i += 1

    return functions


# ── STEP 5: Walk the entire Juliet directory tree ─────────────────────────────

def process_juliet(juliet_root):
    """
    Walks the Juliet dataset and returns a list of labelled function records.
    Each record: { cwe, label, filename, function_code, num_functions }
    """
    records = []
    skipped_unreadable = 0
    skipped_no_block   = 0

    all_files = []
    for dirpath, _, filenames in os.walk(juliet_root):
        # Skip the testcasesupport folder entirely — it contains main.cpp,
        # main_linux.cpp, io.c and other harness files, not test cases
        if 'testcasesupport' in dirpath:
            continue
        for fname in filenames:
            if fname.endswith(('.c', '.cpp')) and fname not in ('main.cpp', 'main_linux.cpp'):
                all_files.append((dirpath, fname))

    print(f"Found {len(all_files)} C/C++ files. Processing...")

    for dirpath, fname in tqdm(all_files):
        cwe = extract_cwe(dirpath) or extract_cwe(fname)

        filepath = os.path.join(dirpath, fname)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except Exception as e:
            print(f"Could not read {filepath}: {e}")
            skipped_unreadable += 1
            continue

        blocks = split_omit_blocks(source)

        if not blocks:
            skipped_no_block += 1
            continue

        for label, block_code in blocks:
            functions = extract_functions(block_code)

            if not functions:
                # Store the whole block if no individual functions were found
                records.append({
                    'cwe':           cwe,
                    'label':         label,
                    'filename':      fname,
                    'function_code': block_code.strip()[:5000],
                    'num_functions': 0,
                })
            else:
                for func in functions:
                    records.append({
                        'cwe':           cwe,
                        'label':         label,
                        'filename':      fname,
                        'function_code': func,
                        'num_functions': len(functions),
                    })

    print(f"Skipped {skipped_unreadable} unreadable files.")
    print(f"Skipped {skipped_no_block} files with no OMITBAD/OMITGOOD blocks.")
    return records


# ── STEP 6: Save results ───────────────────────────────────────────────────────

def save_results(records, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.DataFrame(records)

    print(f"\nTotal function samples: {len(df)}")
    print(f"Vulnerable (label=1):   {(df['label'] == 1).sum()}")
    print(f"Safe       (label=0):   {(df['label'] == 0).sum()}")
    print(f"Unique CWE types:       {df['cwe'].nunique()}")
    print(f"CWE distribution (top 10):\n{df['cwe'].value_counts().head(10)}")

    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    json_file = output_file.replace('.csv', '.json')
    df.to_json(json_file, orient='records', lines=True)
    print(f"Saved to {json_file}")


# ── STEP 7: Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    records = process_juliet(JULIET_ROOT)
    save_results(records, OUTPUT_FILE)
