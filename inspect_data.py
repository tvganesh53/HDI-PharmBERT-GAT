import pandas as pd
df = pd.read_csv(r'C:\Users\tvgan\Downloads\herb_drug_final.csv', nrows=5)
print('Shape will be checked separately')
print('Columns:', list(df.columns))
print()
print(df.head(3).to_string())
