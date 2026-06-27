import re
import pandas as pd


def load_nuts_vocabulary(filepath):
    nuts_df = pd.read_excel(filepath)
    
    nuts_set = set(nuts_df['NUTS label'].astype(str).str.lower().str.strip())
    
    return set(nuts_set)
def extract_geography(text_list, nuts_vocab):
    geo_t = set()
    for text in text_list:
        clean_text = str(text).lower().strip()
        if clean_text in nuts_vocab:
            geo_t.add(clean_text)
            
    return geo_t


def extract_dates(columns):
    d_t = set()
    pattern = re.compile(
        r'\b(?:19|20)\d{2}[-]?Q[1-4]\b|'              
        r'\b(?:19|20)\d{2}[-]?M(?:0[1-9]|1[0-2])\b|'  
        r'\b(?:19|20)\d{2}\b'                         
    )
    
    for col in columns:
        col_str = str(col)
        matches = pattern.findall(col_str)
        d_t.update(matches)
        
    return d_t
