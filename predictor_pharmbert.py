import os, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

MODEL_REPO = "tvganesh538/hdi-models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Harmful", "Negative", "No Effect", "Positive", "Possible"]


class PharmBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels=5):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.pooler_output)


class PharmBERTPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        if self._loaded:
            return
        print("Downloading pharmbert_p9_best.pt ...")
        ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="pharmbert_p9_best.pt",
            cache_dir="/tmp/hdi_models"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        base = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.model = PharmBERTClassifier(base, num_labels=5).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"PharmBERT loaded — missing: {missing}, unexpected: {unexpected}")
        self.model.eval()
        self._loaded = True

    def predict(self, texts):
        """Returns list of dicts — one per text.
        Each dict has keys: top_label, top_score, scores (dict label->score)
        This is the format pipeline_hdi.py expects from self._bert.predict().
        """
        if not self._loaded:
            self.load()
        results = []
        for text in texts:
            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=256, padding="max_length"
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                # Prior correction — down-weight Harmful (77.3%) and boost minority classes
                import torch as _torch
                freq = _torch.tensor([0.021, 0.051, 0.080, 0.075, 0.773], device=logits.device)
                correction = _torch.log(1.0 / freq)
                logits = logits + correction
                probs = _torch.softmax(logits, dim=-1).squeeze().tolist()
            top_idx = int(torch.argmax(logits, dim=-1).item())
            results.append({
                "top_label": LABEL_NAMES[top_idx],
                "top_score": probs[top_idx],
                "scores": {LABEL_NAMES[i]: probs[i] for i in range(len(LABEL_NAMES))}
            })
        return results


pharmbert_predictor = PharmBERTPredictor()
