with open('predictor_pharmbert.py', 'rb') as f:
    content = f.read()

idx = content.find(b'results.append')
chunk = content[idx:idx+300]
print(repr(chunk))
# Show hex of closing brace area
close_idx = chunk.find(b'})')
print('Around }):', repr(chunk[close_idx-5:close_idx+20]))
