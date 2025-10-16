# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 2025

@author: malak
"""

# ====== Stress Dataset Exploration ======
# This script loads the stress dataset, explores it, visualizes features,
# and outputs correlations and label distribution.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")  # Plot style

# read pickle file and stored it's data
file = pd.read_pickle(
    r"C:\Users\malak\OneDrive - Morgan State University\Team 8 Group Project Folder\WESAD\data_stress.pkl"
)


print("===== Dataset Overview =====")
print("Shape:", file.shape)
print("\n\nInfo:")
print("\n===== Columns =====")


i = 0
for col in file.columns:
    i += 1
    print(f"{i}   {col}")
    


plt.figure(figsize=(6,4))
sns.countplot(x="Stress Levels", data=file)
plt.title("Label Distribution")
plt.show()

print("\nLabel counts:")
print(file["Stress Levels"].value_counts())

# --- Feature Distributions (Dynamic) ---
numeric_features = file.select_dtypes(include='number').columns.tolist()
# Remove label column if accidentally numeric
if "Stress Levels" in numeric_features:
    numeric_features.remove("Stress Levels")

for feature in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(file[feature], bins=30, kde=True, color="blue")
    plt.title(f"{feature} Distribution")
    plt.show()

# Optional: explore relationships
# sns.pairplot(df, hue="Stress Levels", vars=numeric_features)
# plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(file.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# --- Findings ---
print("\n===== Findings =====")
print(f"- Dataset has {len(numeric_features)} numerical features + stress labels.")
print("- Stress labels may be imbalanced; check counts above.")
print("- Features show variability across stress levels; check distributions above.")
print("- Correlated features should be considered for preprocessing.")
