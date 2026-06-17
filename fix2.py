import os, sys
sys.path.insert(0, '.')

# Read current file
with open('predictor_pharmbert.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the results section
new_lines = []
skip = False
for i, line in enumerate(lines):
    if '            results.append({' in line:
        skip = True
        new_lines.append('            classifications = [\n')
        new_lines.append('                {"label": self.label_names[i], "score": round(scores[i], 4), "reasoning": ""}\n')
        new_lines.append('                for i in range(len(self.label_names))\n')
        new_lines.append('            ]\n')
        new_lines.append('            classifications.sort(key=lambda x: x["score"], reverse=True)\n')
        new_lines.append('            results.append({"classifications": classifications})\n')
    elif skip and 'return results[0]' in line:
        skip = False
        new_lines.append('        return {"outputs": results}\n')
    elif skip and line.strip() in ['"label":      top_label,', '"score":      round(top_score, 4),', '"all_scores": all_scores,', '})']:
        continue
    else:
        new_lines.append(line)

with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Done')
