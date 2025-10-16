# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:29:32 2025

@author: malak
"""

# 🧠 Stress Level Prediction with Random Forest

# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Dataset ---
file_path = r"C:\Users\malak\OneDrive - Morgan State University\Team 8 Group Project Folder\WESAD\data_stress.pkl"
df = pd.read_pickle(file_path)

# --- Features & Target ---
X = df.drop("Stress Levels", axis=1)
y = df["Stress Levels"]

# Fill missing values with mean
X = X.fillna(X.mean())

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Random Forest Model ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- Predictions ---
y_pred = rf_model.predict(X_test_scaled)

# --- Evaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
