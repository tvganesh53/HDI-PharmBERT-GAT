import os, re, glob

def scrub(txt):
    txt = re.sub(r'hf_[A-Za-z0-9]{30,}', 'YOUR_HF_TOKEN', txt)
    txt = re.sub(r'gsk_[A-Za-z0-9]{30,}', 'YOUR_GROQ_KEY', txt)
    txt = re.sub(r'sk-[A-Za-z0-9_\-]{20,}', 'YOUR_API_KEY', txt)
    return txt

# Only scan files in current folder, not subfolders
for f in glob.glob('*.py') + glob.glob('*.json') + glob.glob('*.yml') + glob.glob('*.md') + glob.glob('*.txt') + glob.glob('*.env*'):
    try:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        clean = scrub(txt)
        if clean != txt:
            open(f, 'w', encoding='utf-8').write(clean)
            print(f'Scrubbed: {f}')
    except:
        pass
print('Done')
