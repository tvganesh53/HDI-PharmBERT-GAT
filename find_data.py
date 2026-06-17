import os
files = os.listdir('.')
data_files = [f for f in files if any(ext in f.lower() for ext in ['.csv', '.json', '.xlsx', '.tsv', '.txt', 'dataset', 'data', 'hdi', 'herb'])]
print('Data files:')
for f in sorted(data_files):
    size = os.path.getsize(f)
    print(f'  {f}  ({size/1024:.1f} KB)')
