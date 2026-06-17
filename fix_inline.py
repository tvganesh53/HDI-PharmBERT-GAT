with open('app_phase_g.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''def _load_predictor() -> Any:
    try:
        from predictor_pharmbert import Predictor
        p = Predictor()
        p.load()
        return p
    except Exception as exc:
        log.warning("Predictor not loaded (%s) \u2014 running in stub mode.", exc)
        return None'''

new = '''def _load_predictor() -> Any:
    try:
        import torch
        from transformers import AutoTokenizer, BertForSequenceClassification
        from pathlib import Path

        model_path = Path(os.getenv("PHARMBERT_MODEL_PATH", "pharmbert_p8_best.pt"))
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        log.info("Loading PharmBERT from %s", model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        label_names = checkpoint.get("label_names", ["No Effect","Possible","Positive","Negative","Harmful"])
        tokenizer_name = checkpoint.get("tokenizer_name", "dmis-lab/biobert-base-cased-v1.2")
        num_labels = len(label_names)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = BertForSequenceClassification.from_pretrained(
            tokenizer_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.classifier.weight.data.copy_(checkpoint["model_state_dict"]["classifier.weight"])
        model.classifier.bias.data.copy_(checkpoint["model_state_dict"]["classifier.bias"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        log.info("PharmBERT loaded on %s with labels %s", device, label_names)

        class InlinePredictor:
            def __init__(self):
                self.model_name = "pharmbert-herb-drug"
                self.is_loaded = True
                self.label_names = label_names
                self._model = model
                self._tokenizer = tokenizer
                self._device = device

            def predict(self, inputs, **kwargs):
                import torch
                texts = [inputs] if isinstance(inputs, str) else inputs
                results = []
                for text in texts:
                    enc = self._tokenizer(
                        text, max_length=128, truncation=True,
                        padding="max_length", return_tensors="pt"
                    )
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = self._model(**enc)
                        probs = torch.softmax(out.logits, dim=-1)[0]
                    scores = probs.cpu().tolist()
                    classifications = [
                        {"label": self.label_names[i], "score": round(scores[i], 4), "reasoning": ""}
                        for i in range(len(self.label_names))
                    ]
                    classifications.sort(key=lambda x: x["score"], reverse=True)
                    results.append({"classifications": classifications})
                return {"outputs": results}

        return InlinePredictor()
    except Exception as exc:
        log.warning("Predictor not loaded (%s) - running in stub mode.", exc)
        return None'''

if old in content:
    content = content.replace(old, new)
    with open('app_phase_g.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed')
else:
    print('Pattern not found')
