from huggingface_hub import HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

content = '''import os, torch
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
        print("Downloading pharmbert_p8_best.pt ...")
        ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="pharmbert_p8_best.pt",
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
        if not self._loaded:
            self.load()
        outputs = []
        for text in texts:
            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=256, padding="max_length"
            ).to(DEVICE)
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            top_idx = int(torch.argmax(logits, dim=-1).item())
            top_label = LABEL_NAMES[top_idx]
            top_score = probs[top_idx]
            all_scores = [{"label": LABEL_NAMES[i], "score": probs[i]} for i in range(len(LABEL_NAMES))]
            outputs.append({
                "severity": {
                    "top_label": top_label,
                    "top_score": top_score,
                    "all_scores": all_scores
                },
                "interaction_type": None,
                "summary": f"Herb interaction — {top_label} ({top_score*100:.1f}%)"
            })
        return {"outputs": outputs}


pharmbert_predictor = PharmBERTPredictor()
'''

with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
    f.write(content)

api = HfApi(token=TOKEN)
api.upload_file(path_or_fileobj='predictor_pharmbert.py',
                path_in_repo='predictor_pharmbert.py',
                repo_id=REPO_ID, repo_type='space',
                commit_message='Fix predict() output shape to match app_phase_g classify loop')
print('Uploaded')
