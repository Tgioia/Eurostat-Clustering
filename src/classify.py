import ollama
import json
import os
import re
import csv

def classify_batch(batch_terms):
    prompt = f"""You are an automated statistical data classifier.
Categorize the following Eurostat terms into exactly five lists.

DEFINITIONS:
"M" (Measures): Quantitative metrics or time concepts (e.g., annual, total, index).
"N" (Dimension Names): Broad categories (e.g., age, sex, geopolitical entity).
"A" (Dimension Values): Specific members of a category (e.g., female, 15-24 years).
"U" (Units): Measurements (e.g., kilogram, percentage, euro).
"O" (Other): Unreadable database codes, alphanumeric IDs (e.g., bs bs1, tps001), or irrelevant text.

REQUIRED FORMAT:
### JSON
{{ "M": [], "N": [], "A": [], "U": [], "O": [] }}
### COMMENTARY
[Optional brief notes on uncertain terms]

TERMS TO CLASSIFY:
{', '.join(batch_terms)}
"""
    
    # temperature 0.0 makes the AI generate text faster and more predictably
    response = ollama.chat(
        model='ibm/granite4.1:3b', 
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.0}
    )
    
    raw_content = response['message']['content']
    
    if "### COMMENTARY" in raw_content:
        json_part, comments_part = raw_content.split("### COMMENTARY", 1)
    else:
        json_part = raw_content
        
    clean_json = re.sub(r'//.*', '', json_part)
    
    try:
        json_match = re.search(r'\{.*\}', clean_json, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"M":[], "N":[], "A":[], "U":[], "O":[]}
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"M":[], "N":[], "A":[], "U":[], "O":[]}

def initialize_csvs(output_dir):
    """Creates the 4 CSV files (and the 'O' file just in case) and writes the header row."""
    os.makedirs(output_dir, exist_ok=True)
    file_map = {
        'M': 'measures.csv',
        'N': 'dimension_names.csv',
        'A': 'dimension_values.csv',
        'U': 'units.csv',
        'O': 'other_discarded.csv' # Good to keep track of what the AI threw away
    }
    
    for key, filename in file_map.items():
        filepath = os.path.join(output_dir, filename)
        # 'w' mode here overwrites previous runs so you start fresh
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['term'])
            
    return file_map

def append_to_csvs(batch_result, file_map, output_dir):
    """Appends ONLY the newly classified terms to the bottom of the CSVs."""
    for key, filename in file_map.items():
        if key in batch_result and isinstance(batch_result[key], list) and len(batch_result[key]) > 0:
            filepath = os.path.join(output_dir, filename)
            # 'a' mode appends to the file instead of rewriting it
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for term in batch_result[key]:
                    writer.writerow([term])

def run_classification():
    input_file = 'outputs/global_vocabulary.txt'
    output_dir = 'outputs'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f.readlines() if line.strip()]
    
    # Setup the files once at the very beginning
    file_map = initialize_csvs(output_dir)
    
    batch_size = 50
    total_batches = (len(terms) // batch_size) + 1
    
    print(f"Starting classification: {len(terms)} terms across {total_batches} batches...")
    
    for i in range(0, len(terms), batch_size):
        batch = terms[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        # 1. Ask the AI
        result = classify_batch(batch)
        
        # 2. Instantly append the new results to the CSV files
        append_to_csvs(result, file_map, output_dir)
        
    print("\n--- Classification complete! ---")

if __name__ == "__main__":
    run_classification()