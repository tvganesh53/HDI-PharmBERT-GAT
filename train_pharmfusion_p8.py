"""
train_pharmfusion_p8.py
=======================
Retrain PharmFusion with correct gat_emb_dim=16 (matching actual node embeddings).

Changes vs P7:
  - gat_emb_dim: 2  →  16
  - Uses real GAT node embeddings (dim=16) from pharmgat_p7_node_embeddings (2).pt
  - Everything else identical to P7 architecture

Usage:
    python train_pharmfusion_p8.py

Output:
    pharmfusion_p8_best.pt   ← upload this to HF Space

Requirements:
    pip install torch transformers pandas scikit-learn tqdm
"""

from __future__ import annotations
import os, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "bert_name":     "dmis-lab/biobert-base-cased-v1.2",
    "gat_emb_dim":   16,          # ← KEY CHANGE from P7 (was 2)
    "hidden_dim":    512,
    "dropout":       0.3,
    "batch_size":    32,
    "epochs":        20,
    "lr_bert":       1e-5,
    "lr_fusion":     5e-5,
    "weight_decay":  0.01,
    "warmup_ratio":  0.1,
    "patience":      5,
    "max_length":    128,
    "seed":          42,
    "val_size":      0.15,
    "test_size":     0.10,
}

DATA_PATH    = r"C:\Users\tvgan\Downloads\herb_drug_final.csv"
GAT_EMB_PATH = r"C:\Users\tvgan\OneDrive\Desktop\nodejs\pharmgat_p7_node_embeddings (2).pt"
OUTPUT_PATH  = r"C:\Users\tvgan\OneDrive\Desktop\nodejs\pharmfusion_p8_best.pt"

LABEL_NAMES  = ["Food", "Herb"]
LABEL2ID     = {"Food": 0, "Herb": 1}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Dataset ───────────────────────────────────────────────────────────────────
class FusionDataset(Dataset):
    def __init__(self, texts, node_ids, labels, tokenizer,
                 node_to_idx, node_embeddings, max_length=128):
        self.texts          = texts
        self.node_ids       = node_ids
        self.labels         = labels
        self.tokenizer      = tokenizer
        self.node_to_idx    = node_to_idx
        self.node_embeddings = node_embeddings   # Tensor [N, 16]
        self.max_length     = max_length
        self.gat_dim        = node_embeddings.shape[1]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # GAT embedding lookup
        nid = self.node_ids[idx]
        if nid in self.node_to_idx:
            emb = self.node_embeddings[self.node_to_idx[nid]]
        else:
            emb = torch.zeros(self.gat_dim)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "gat_emb":        emb.float(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Model ─────────────────────────────────────────────────────────────────────
class FusionLayer(nn.Module):
    def __init__(self, bert_dim=768, gat_dim=16, hidden_dim=512,
                 num_labels=2, dropout=0.3):
        super().__init__()
        self.bert_proj  = nn.Linear(bert_dim, hidden_dim)
        self.gat_proj   = nn.Linear(gat_dim,  hidden_dim)
        self.gate       = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                                 batch_first=True, dropout=dropout)
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
        fused    = gate_val * b + (1 - gate_val) * g
        fused_seq = fused.unsqueeze(1)
        attn_out, _ = self.cross_attn(fused_seq, fused_seq, fused_seq)
        return self.classifier(attn_out.squeeze(1))


class PharmFusionModel(nn.Module):
    def __init__(self, bert_name, gat_dim, hidden_dim, num_labels, dropout):
        super().__init__()
        self.bert   = BertModel.from_pretrained(bert_name)
        self.fusion = FusionLayer(
            bert_dim   = self.bert.config.hidden_size,
            gat_dim    = gat_dim,
            hidden_dim = hidden_dim,
            num_labels = num_labels,
            dropout    = dropout,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, gat_emb):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0, :]
        return self.fusion(cls, gat_emb)

# ── Training helpers ──────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
                gat_emb        = batch["gat_emb"].to(device),
            )
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    per_f1   = f1_score(all_labels, all_preds, average=None).tolist()
    return acc, macro_f1, per_f1

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load GAT embeddings ───────────────────────────────────────────────────
    print(f"\nLoading GAT embeddings from {GAT_EMB_PATH} …")
    gat_data        = torch.load(GAT_EMB_PATH, map_location="cpu", weights_only=False)
    node_embeddings = gat_data["node_embeddings"].float()   # [2289, 16]
    node_to_idx     = gat_data["node_to_idx"]
    print(f"  GAT nodes: {node_embeddings.shape[0]}, dim: {node_embeddings.shape[1]}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset from {DATA_PATH} …")
    df = pd.read_csv(DATA_PATH)
    print(f"  Total rows: {len(df)}")

    # Filter valid labels
    df = df[df["Type"].isin(LABEL_NAMES)].copy()
    df["label"] = df["Type"].map(LABEL2ID)
    print(f"  After filter: {len(df)} rows")
    print(f"  Label distribution:\n{df['Type'].value_counts().to_string()}")

    texts    = df["text_bert_input"].fillna("").tolist()
    node_ids = df["Food_Herb_ID"].fillna("").tolist()
    labels   = df["label"].tolist()

    # ── Split ─────────────────────────────────────────────────────────────────
    idx = list(range(len(texts)))
    train_idx, temp_idx = train_test_split(idx, test_size=CONFIG["val_size"] + CONFIG["test_size"],
                                            random_state=CONFIG["seed"], stratify=labels)
    val_ratio = CONFIG["val_size"] / (CONFIG["val_size"] + CONFIG["test_size"])
    val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_ratio,
                                          random_state=CONFIG["seed"],
                                          stratify=[labels[i] for i in temp_idx])

    print(f"\n  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {CONFIG['bert_name']} …")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["bert_name"])

    def make_dataset(idxs):
        return FusionDataset(
            texts          = [texts[i] for i in idxs],
            node_ids       = [node_ids[i] for i in idxs],
            labels         = [labels[i] for i in idxs],
            tokenizer      = tokenizer,
            node_to_idx    = node_to_idx,
            node_embeddings = node_embeddings,
            max_length     = CONFIG["max_length"],
        )

    train_ds = make_dataset(train_idx)
    val_ds   = make_dataset(val_idx)
    test_ds  = make_dataset(test_idx)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding PharmFusion P8 (gat_emb_dim={CONFIG['gat_emb_dim']}) …")
    model = PharmFusionModel(
        bert_name  = CONFIG["bert_name"],
        gat_dim    = CONFIG["gat_emb_dim"],
        hidden_dim = CONFIG["hidden_dim"],
        num_labels = len(LABEL_NAMES),
        dropout    = CONFIG["dropout"],
    ).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ── Optimizer (separate LR for BERT vs fusion head) ───────────────────────
    bert_params   = list(model.bert.parameters())
    fusion_params = list(model.fusion.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bert_params,   "lr": CONFIG["lr_bert"],   "weight_decay": CONFIG["weight_decay"]},
        {"params": fusion_params, "lr": CONFIG["lr_fusion"],  "weight_decay": CONFIG["weight_decay"]},
    ])

    total_steps   = len(train_loader) * CONFIG["epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Class weights to handle Herb/Food imbalance (14708 Herb vs 2858 Food)
    label_counts  = [LABEL2ID[l] for l in df["Type"]]
    class_counts  = [label_counts.count(i) for i in range(len(LABEL_NAMES))]
    class_weights = torch.tensor(
        [max(class_counts) / c for c in class_counts], dtype=torch.float
    ).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    print(f"  Class weights: Food={class_weights[0]:.2f}  Herb={class_weights[1]:.2f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for up to {CONFIG['epochs']} epochs …\n")
    best_f1       = 0.0
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            optimizer.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
                gat_emb        = batch["gat_emb"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, val_f1, val_per = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f} | "
              f"val_macro_f1={val_f1:.4f} | per_class={[round(f,4) for f in val_per]} | "
              f"time={elapsed:.0f}s")

        if val_f1 > best_f1:
            best_f1      = val_f1
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            print(f"  ✓ New best F1: {best_f1:.4f} — saved")
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (patience={CONFIG['patience']})")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\nEvaluating best model on test set …")
    model.load_state_dict(best_state)
    test_acc, test_f1, test_per = evaluate(model, test_loader, device)
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Test macro F1:  {test_f1:.4f}")
    print(f"  Per-class F1:   {[round(f,4) for f in test_per]}")
    print(f"  Labels:         {LABEL_NAMES}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    # Recompute all metrics on test set cleanly
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels_test = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                token_type_ids = batch["token_type_ids"].to(device),
                gat_emb        = batch["gat_emb"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels_test.extend(batch["label"].tolist())

    final_acc      = accuracy_score(all_labels_test, all_preds)
    final_macro    = f1_score(all_labels_test, all_preds, average="macro")
    final_weighted = f1_score(all_labels_test, all_preds, average="weighted")
    final_per      = f1_score(all_labels_test, all_preds, average=None).tolist()

    checkpoint = {
        "model_state_dict": best_state,
        "config":           CONFIG,
        "label_names":      LABEL_NAMES,
        "metrics": {
            "accuracy":     final_acc,
            "macro_f1":     final_macro,
            "weighted_f1":  final_weighted,
            "per_class_f1": final_per,
        },
        "phase":    8,
        "macro_f1": final_macro,
    }

    torch.save(checkpoint, OUTPUT_PATH)
    print(f"\n✅ Saved to {OUTPUT_PATH}")
    print(f"   Accuracy:    {final_acc:.4f}")
    print(f"   Macro F1:    {final_macro:.4f}")
    print(f"   Weighted F1: {final_weighted:.4f}")
    print(f"   Per-class:   {[round(f,4) for f in final_per]}  ({LABEL_NAMES})")
    print("\nNext step: upload pharmfusion_p8_best.pt to HF Space and update predictor_fusion.py")


if __name__ == "__main__":
    main()
PYEOF