with open('predictor_pharmbert.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if 'Build model architecture from config only' in line:
        skip = True
        new_lines.append('        # Build and load model\n')
        new_lines.append('        self._model = BertForSequenceClassification.from_pretrained(\n')
        new_lines.append('            tokenizer_name,\n')
        new_lines.append('            num_labels=num_labels,\n')
        new_lines.append('            ignore_mismatched_sizes=True,\n')
        new_lines.append('        )\n')
        new_lines.append('        # Force load ALL weights from checkpoint\n')
        new_lines.append('        self._model.load_state_dict(checkpoint["model_state_dict"], strict=False)\n')
        new_lines.append('        self._model.classifier.weight.data.copy_(checkpoint["model_state_dict"]["classifier.weight"])\n')
        new_lines.append('        self._model.classifier.bias.data.copy_(checkpoint["model_state_dict"]["classifier.bias"])\n')
    elif skip and ('Load ALL weights' in line or 'BertConfig' in line or 'config =' in line or 'BertForSequence' in line or 'load_state_dict' in line or 'Missing keys' in line or 'Unexpected' in line or 'Classifier weight' in line or 'hashlib' in line or 'missing' in line or 'unexpected' in line):
        continue
    else:
        if skip and line.strip() == '':
            skip = False
        new_lines.append(line)

with open('predictor_pharmbert.py', 'w', encoding='utf-8', newline='\n') as f:
    f.writelines(new_lines)
print('Done')
