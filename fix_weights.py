with open('predictor_pharmbert.py', 'rb') as f:
    content = f.read()

old = b'Build model architecture\r\n        self._model = BertForSequenceClassification.from_pretrained(\r\n            tokenizer_name,\r\n            num_labels=num_labels,\r\n            ignore_mismatched_sizes=True,\r\n        )\r\n\r\n        # Load trained weights\r\n        self._model.load_state_dict(checkpoint["model_state_dict"])'

new = b'Build model architecture from config only\r\n        from transformers import BertConfig\r\n        config = BertConfig.from_pretrained(tokenizer_name, num_labels=num_labels)\r\n        self._model = BertForSequenceClassification(config)\r\n\r\n        # Load ALL weights from checkpoint\r\n        self._model.load_state_dict(checkpoint["model_state_dict"])'

if old in content:
    content = content.replace(old, new)
    with open('predictor_pharmbert.py', 'wb') as f:
        f.write(content)
    print('Fixed successfully')
else:
    print('Not found')
    idx = content.find(b'load_state_dict')
    print(repr(content[idx-20:idx+60]))
