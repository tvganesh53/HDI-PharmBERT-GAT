with open('predictor_pharmbert.py', 'rb') as f:
    content = f.read()

old = b'        # Build model architecture from config only\r\n        from transformers import BertConfig\r\n        config = BertConfig.from_pretrained(tokenizer_name, num_labels=num_labels)\r\n        self._model = BertForSequenceClassification(config)\r\n\r\n        # Load ALL weights from checkpoint\r\n        self._model.load_state_dict(checkpoint["model_state_dict"])'

new = b'        # Build model using from_pretrained then override ALL weights\r\n        self._model = BertForSequenceClassification.from_pretrained(\r\n            tokenizer_name,\r\n            num_labels=num_labels,\r\n            ignore_mismatched_sizes=True,\r\n        )\r\n        # Override ALL weights including classifier\r\n        missing, unexpected = self._model.load_state_dict(checkpoint["model_state_dict"], strict=False)\r\n        log.info("Weight loading - missing: %s, unexpected: %s", len(missing), len(unexpected))\r\n        # Force classifier weights\r\n        self._model.classifier.weight.data = checkpoint["model_state_dict"]["classifier.weight"]\r\n        self._model.classifier.bias.data = checkpoint["model_state_dict"]["classifier.bias"]'

if old in content:
    content = content.replace(old, new)
    with open('predictor_pharmbert.py', 'wb') as f:
        f.write(content)
    print('Fixed')
else:
    print('Not found - showing current load section')
    idx = content.find(b'Build model architecture')
    print(repr(content[idx:idx+400]))
