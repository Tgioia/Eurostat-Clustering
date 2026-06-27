import os
import glob
import pandas as pd
import re

from extractData import extract_dates, load_nuts_vocabulary, extract_geography


def is_eurostat_data_value(text):
    """
    Riconosce se una stringa è un dato statistico di Eurostat.
    Cattura: "123", "-0.4", "12.5 e", "-0.1 bep", e il simbolo ":" (dato mancante).
    """
    # Eurostat usa spesso ":" o ": c" per i dati mancanti/confidenziali
    if text.startswith(':'):
        return True
        
    # Regex per: [segno opzionale][numeri][punto opzionale][numeri][spazio opzionale][lettere opzionali]
    pattern = re.compile(r'^[-+]?\d*\.?\d+\s*[a-zA-Z]*$')
    return bool(pattern.match(text))

def clean_strings(text_list):
    """
    Pulisce le stringhe scartando i dati numerici e le flag.
    """
    cleaned = set()
    for item in text_list:
        text = str(item).strip()
        
        # Tieni la stringa solo se NON è vuota e NON è un valore numerico/flag
        if text and not is_eurostat_data_value(text):
            cleaned.add(text.lower())
            
    return cleaned

    
def main():
    print("Initializing Eurostat Vocabulary Pipeline...")
    
    RAW_DATA_DIR = 'data/eurostat_2000_tables/*.csv'
    NUTS_FILE = 'data/NUTS2021-NUTS2024.xlsx'
    
    print("Loading NUTS vocabulary...")
    nuts_vocab = load_nuts_vocabulary(NUTS_FILE)
    
    global_vocabulary = set()
    
    file_list = glob.glob(RAW_DATA_DIR)
    print(f"Found {len(file_list)} tables to process.\n")
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            d_t = extract_dates(df.columns)
            
            raw_strings = df.iloc[:, 0].tolist()
            
            table_title = filename.replace('.csv', '').replace('_', ' ')
            raw_strings.append(table_title)
            
            s_t = clean_strings(raw_strings)
            
            geo_t = extract_geography(s_t, nuts_vocab)
            
            v_t = s_t - geo_t - d_t
            
            global_vocabulary.update(v_t)
            print(f"[SUCCESS] {filename} -> Added {len(v_t)} unique terms.")
            
        except Exception as e:
            print(f"[ERROR] Could not process {filename}: {e}")
            
    print("\n--- Extraction Complete ---")
    print(f"Total unique terms in Global Vocabulary: {len(global_vocabulary)}")
    
    output_dir = './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'global_vocabulary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for term in sorted(global_vocabulary):
            f.write(f"{term}\n")
            
    print(f"Saved global vocabulary to {output_file}")

if __name__ == "__main__":
    main()