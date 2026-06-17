with open('app_phase_g.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if "@app.get('/debug/predictor'" in line:
        skip = True
        new_lines.append(line)
    elif skip and 'async def debug_predictor' in line:
        new_lines.append(line)
    elif skip and 'return {' in line and 'status' in line:
        skip = False
        new_lines.append(line)
    elif skip:
        if 'from predictor_pharmbert import Predictor' in line and 'GROQ_API_KEY' not in line:
            new_lines.append(line)
        elif 'from predictor_pharmbert import Predictor, GROQ_API_KEY' in line:
            new_lines.append("        from predictor_pharmbert import Predictor\n")
        elif 'GROQ_API_KEY' in line:
            continue
        elif 'groq_key' in line:
            new_lines.append("        return {'status': 'ok', 'model': p.model_name, 'labels': p.label_names, 'is_loaded': p.is_loaded}\n")
            skip = False
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('app_phase_g.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Done')
