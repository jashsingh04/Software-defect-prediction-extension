
import pandas as pd

from selector import PATH


file_path = PATH


DATA = pd.read_csv(file_path)
df = DATA

df = df.rename(columns={'problems': 'defects'})
df = df.rename(columns={'bug': 'defects'})

df['defects'] = df['defects'].replace({1: True, 0: False})
df['defects'] = df['defects'].replace({'yes': 'TRUE', 'no': 'FALSE'})
if df['defects'].dtype != bool:
    df['defects'] = df['defects'].astype(bool)
# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)

DATA = pd.read_csv(file_path)
df = DATA
# Separate the target variable and features

X = DATA.drop(['defects'], axis=1)
y = DATA['defects']



# Inspect DataFrame
print(df.head())  # Print first few rows
print(df.columns)  # Print column names
print(df.dtypes)  # Print data types
