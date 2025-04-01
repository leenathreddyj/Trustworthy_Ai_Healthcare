import pandas as pd

df = pd.read_csv('data/heart_disease_uci.csv')

print("Columns in the dataset:")
print(df.columns.tolist())  # Show columns as a list

print("\nFirst few rows:")
print(df.head())
