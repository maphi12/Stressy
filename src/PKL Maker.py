# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 22:14:34 2025

@author: malak
"""

import pandas as pd
import pickle

# Step 1: Load CSV
df = pd.read_csv(
    r"C:\Users\malak\OneDrive - Morgan State University\Team 8 Group Project Folder\data_stress-2.csv",
    sep=";", decimal=",", na_values=["NULL"]
)

# Step 2: Save DataFrame as a .pkl inside your WESAD folder
output_path = r"C:\Users\malak\OneDrive - Morgan State University\Team 8 Group Project Folder\WESAD\data_stress.pkl"

with open(output_path, "wb") as f:
    pickle.dump(df, f)

print(f"Pickle file saved at: {output_path}")
