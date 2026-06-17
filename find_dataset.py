import os
search_dirs = [
    r'C:\Users\tvgan\OneDrive',
    r'C:\Users\tvgan\Documents',
    r'C:\Users\tvgan\Downloads',
    r'C:\Users\tvgan\Desktop',
]
extensions = ['.csv', '.tsv', '.json', '.xlsx']
keywords = ['hdi', 'herb', 'drug', 'fusion', 'interaction', 'food', 'pharmfusion', 'dataset', 'train']

for base in search_dirs:
    for root, dirs, files in os.walk(base):
        # Skip venv and node_modules
        dirs[:] = [d for d in dirs if d not in ['venv', '.venv', 'node_modules', '__pycache__']]
        for f in files:
            flower = f.lower()
            if any(ext in flower for ext in extensions) and any(kw in flower for kw in keywords):
                full = os.path.join(root, f)
                size = os.path.getsize(full)
                print(f'{size/1024:8.1f} KB  {full}')
