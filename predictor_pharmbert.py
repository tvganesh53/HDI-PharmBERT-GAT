import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Harmful", "Negative", "No Effect", "Positive", "Possible"]


class PharmBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels=5):
        super().__init__()
        self.bert       = base_model
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))


class PharmBERTPredictor:
    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self._loaded   = False

    @property
    def is_loaded(self):
        return self._loaded

    def load(self):
        if self._loaded:
            return
        print("Downloading pharmbert_p10_best.pt ...")
        ckpt_path = hf_hub_download(
            repo_id="tvganesh538/hdi-models",
            filename="pharmbert_p10/pharmbert_p10_best.pt",
            repo_type="model",
        )

        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        base           = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.model     = PharmBERTClassifier(base, num_labels=5).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"PharmBERT P10 loaded | missing={missing} | unexpected={unexpected}")
        self.model.eval()
        self._loaded = True

    def predict(self, texts):
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
                probs  = torch.softmax(logits, dim=-1).squeeze().tolist()
            top_idx = int(torch.argmax(logits, dim=-1).item())
            results.append({
                "top_label": LABEL_NAMES[top_idx],
                "top_score": probs[top_idx],
                "scores":    {LABEL_NAMES[i]: probs[i] for i in range(len(LABEL_NAMES))}
            })
        return results


pharmbert_predictor = PharmBERTPredictor()
