"""
predictor_fusion.py — PharmFusion P8 predictor (gat_emb_dim=16, F1=1.0)
Downloads model + GAT embeddings from HF model repo at startup.
No .pt files needed in the Space image.
"""
import os
import logging
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

REPO_ID       = "tvganesh538/hdi-models"
FUSION_FILE   = "pharmfusion_p9_best.pt"
GAT_EMB_FILE  = "pharmgat_node_embeddings.pt"
BERT_NAME     = "dmis-lab/biobert-base-cased-v1.2"
LABEL_NAMES   = ["Food", "Herb"]
MAX_LENGTH    = 128
GAT_EMB_DIM   = 16
HIDDEN_DIM    = 512
CACHE_DIR     = "/app/model_cache"

# ── Architecture ────────────────────────────────────────────────────────────

class FusionLayer(nn.Module):
    def __init__(
        self,
        bert_dim:   int = 768,
        gat_dim:    int = 16,
        hidden_dim: int = 512,
        num_labels: int = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.bert_proj  = nn.Linear(bert_dim, hidden_dim)
        self.gat_proj   = nn.Linear(gat_dim,  hidden_dim)
        self.gate       = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, bert_cls, gat_emb):
        b = self.bert_proj(bert_cls)
        g = self.gat_proj(gat_emb)
        gate_val = self.gate(torch.cat([b, g], dim=-1))
        fused = gate_val * b + (1 - gate_val) * g
        attn_out, _ = self.cross_attn(
            fused.unsqueeze(1), fused.unsqueeze(1), fused.unsqueeze(1)
        )
        return self.classifier(attn_out.squeeze(1))


class PharmFusionModel(nn.Module):
    def __init__(
        self,
        bert_name:  str,
        gat_dim:    int,
        hidden_dim: int,
        num_labels: int,
        dropout:    float,
    ):
        super().__init__()
        self.bert   = BertModel.from_pretrained(bert_name)
        self.fusion = FusionLayer(
            self.bert.config.hidden_size, gat_dim, hidden_dim, num_labels, dropout
        )

    def forward(self, input_ids, attention_mask, token_type_ids, gat_emb):
        cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state[:, 0, :]
        return self.fusion(cls, gat_emb)


# ── Predictor ───────────────────────────────────────────────────────────────

class PharmFusionPredictor:
    model_name = "pharmfusion-p8"

    def __init__(self):
        self._model          = None
        self._tokenizer      = None
        self._node_embeddings = None   # Tensor [N, 16]
        self._node_to_idx    = None    # dict {node_id: int}
        self._device         = "cpu"
        self._loaded         = False
        self._label_names    = LABEL_NAMES

    # ── public ──────────────────────────────────────────────────────────────

    def load(self):
        if self._loaded:
            return
        log.info("PharmFusion P8: downloading from %s …", REPO_ID)
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Download fusion model
        fusion_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FUSION_FILE,
            cache_dir=CACHE_DIR,
        )
        # Download GAT embeddings
        gat_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=GAT_EMB_FILE,
            cache_dir=CACHE_DIR,
        )
        log.info("PharmFusion P8: fusion=%s  gat=%s", fusion_path, gat_path)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load GAT embeddings
        gat_data = torch.load(gat_path, map_location="cpu", weights_only=False)
        self._node_embeddings = gat_data["node_embeddings"].float()  # [2289, 16]
        self._node_to_idx     = gat_data["node_to_idx"]
        actual_gat_dim = self._node_embeddings.shape[1]
        log.info(
            "PharmFusion P8: GAT %d nodes, dim=%d",
            self._node_embeddings.shape[0],
            actual_gat_dim,
        )

        # Load fusion model checkpoint
        ckpt = torch.load(fusion_path, map_location="cpu", weights_only=False)
        label_names = ckpt.get("label_names", LABEL_NAMES)
        self._label_names = label_names

        cfg        = ckpt.get("config", {})
        gat_dim    = cfg.get("gat_emb_dim", actual_gat_dim)
        hidden_dim = cfg.get("hidden_dim",  HIDDEN_DIM)
        dropout    = cfg.get("dropout",     0.3)

        model = PharmFusionModel(
            BERT_NAME, gat_dim, hidden_dim, len(label_names), dropout
        )
        state = ckpt.get("model_state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        self._model = model.to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
        self._loaded    = True
        log.info(
            "PharmFusion P8: loaded ✓  labels=%s  gat_dim=%d  macro_f1=%.4f",
            self._label_names,
            gat_dim,
            ckpt.get("macro_f1", float("nan")),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _get_gat_emb(self, node_id: str) -> torch.Tensor:
        """Return GAT embedding for node_id, or zeros if unknown."""
        if node_id and node_id in self._node_to_idx:
            return self._node_embeddings[self._node_to_idx[node_id]]
        return torch.zeros(self._node_embeddings.shape[1])

    def predict(self, texts: list[str], node_ids: list[str] | None = None) -> list[dict]:
        """
        texts:    list of pre-formatted input strings
        node_ids: optional list of Food_Herb_IDs (e.g. "F00001") for GAT lookup
                  — if None or unknown, zeros are used (graceful fallback)
        Returns list of dicts: {top_label, top_score, scores}
        """
        if not self._loaded:
            self.load()

        if node_ids is None:
            node_ids = [""] * len(texts)

        enc = self._tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        gat_batch = torch.stack(
            [self._get_gat_emb(nid) for nid in node_ids]
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(
                input_ids      = enc["input_ids"],
                attention_mask = enc["attention_mask"],
                token_type_ids = enc.get("token_type_ids",
                                         torch.zeros_like(enc["input_ids"])),
                gat_emb        = gat_batch,
            )
            probs = torch.softmax(logits, dim=-1).cpu()

        results = []
        for prob_row in probs:
            scores = {
                self._label_names[i]: round(float(prob_row[i]), 6)
                for i in range(len(self._label_names))
            }
            top_idx   = int(prob_row.argmax())
            top_label = self._label_names[top_idx]
            top_score = float(prob_row[top_idx])
            results.append({
                "top_label": top_label,
                "top_score": top_score,
                "scores":    scores,
            })
        return results


# Singleton
pharmfusion_predictor = PharmFusionPredictor()
