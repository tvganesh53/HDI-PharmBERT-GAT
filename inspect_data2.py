import pandas as pd
df = pd.read_csv(r'C:\Users\tvgan\Downloads\herb_drug_final.csv')
print('Shape:', df.shape)
print()
print('Columns:', list(df.columns))
print()
# Find label/type column
for col in df.columns:
    if df[col].nunique() < 20:
        print(f'Column [{col}] unique values:', df[col].value_counts().to_dict())
