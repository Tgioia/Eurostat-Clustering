import pandas as pd
import glob
import os

data_path = 'data/eurostat_2000_tables/*.csv'
file_list = glob.glob(data_path)

tables = {}

for file in file_list:
    filename = os.path.basename(file)
    df = pd.read_csv(file)
    tables[filename] = df
    
print(f"Loaded {len(tables)} tables.")