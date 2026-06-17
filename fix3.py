with open('app_phase_g.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = "        return {'status': 'ok', 'groq_key_set': bool(GROQ_API_KEY), 'groq_key_prefix': GROQ_API_KEY[:8] if GROQ_API_KEY else 'empty'}\n    except Exception as e:\n        return {'status': 'error', 'error': str(e), 'groq_env': os.getenv('GROQ_API_KEY', 'NOT SET')[:8]}"

new = "        return {'status': 'ok', 'model': p.model_name, 'labels': p.label_names, 'is_loaded': p.is_loaded}\n    except Exception as e:\n        return {'status': 'error', 'error': str(e)}"

if old in content:
    content = content.replace(old, new)
    with open('app_phase_g.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed')
else:
    print('Not found')
