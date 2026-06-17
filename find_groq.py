import os, glob

# Add your actual Groq key here
secrets = [
    'YOUR_HF_TOKEN',
    'YOUR_API_KEY',
    'YOUR_API_KEY',
    'YOUR_API_KEY',
]

# Find Groq key pattern
import re
for f in glob.glob('*.py'):
    txt = open(f, encoding='utf-8', errors='ignore').read()
    matches = re.findall(r'gsk_[A-Za-z0-9]{50,}', txt)
    if matches:
        print(f'{f}: {matches[0][:30]}...')
