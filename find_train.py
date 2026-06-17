import os
files = os.listdir('.')
fusion_files = [f for f in files if 'fusion' in f.lower() or 'phase7' in f.lower() or 'phase_7' in f.lower() or 'train' in f.lower()]
print('Candidate files:')
for f in sorted(fusion_files):
    print(' ', f)
