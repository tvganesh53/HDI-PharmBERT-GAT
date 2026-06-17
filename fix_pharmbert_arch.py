from huggingface_hub import hf_hub_download, HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

NEW_CONTENT = '''
import os, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download

MODEL_REPO = "tvganesh538/hdi-models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Harmful", "Negative", "No Effect", "Positive", "Possible"]

class PharmBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels=5):
        super().__init__()
        self.bert = base_model
        self.classifier = nn.Linear(768, num_labels)   # flat — matches checkpoint

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output          # [B, 768]
        return self.classifier(pooled)      # [B, 5]


class PharmBERTPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print("Downloading pharmbert_p8_best.pt from HF hub...")
        ckpt_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="pharmbert_p8_best.pt",
            cache_dir="/tmp/hdi_models"
        )
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        base = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.model = PharmBERTClassifier(base, num_labels=5).to(DEVICE)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Support both bare state_dict and wrapped checkpoint
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"PharmBERT loaded — missing: {missing}, unexpected: {unexpected}")
        self.model.eval()
        self._loaded = True

    def predict(self, texts: list[str]) -> list[dict]:
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
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            top_idx = int(torch.argmax(logits, dim=-1).item())
            results.append({
                "top_label": LABEL_NAMES[top_idx],
                "top_score": probs[top_idx],
                "all_scores": [
                    {"label": LABEL_NAMES[i], "score": probs[i]}
                    for i in range(len(LABEL_NAMES))
                ]
            })
        return results


pharmbert_predictor = PharmBERTPredictor()
'''

with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
    f.write(NEW_CONTENT.strip())

api = HfApi(token=TOKEN)
api.upload_file(
    path_or_fileobj='predictor_pharmbert.py',
    path_in_repo='predictor_pharmbert.py',
    repo_id=REPO_ID,
    repo_type='space',
    commit_message='Fix PharmBERT classifier: flat Linear(768->5) to match checkpoint'
)
print('Uploaded predictor_pharmbert.py')
