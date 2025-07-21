import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from src.fusion_dataloader import DatasetLoader  # make sure this path is valid

# --------------------- Dataset ---------------------
class TextImageFusionDataset(Dataset):
    def __init__(self, text_embeds_path, image_embeds_path, label_map):
        with open(text_embeds_path, 'r') as f:
            text_data = json.load(f)
        with open(image_embeds_path, 'r') as f:
            image_embeddings = json.load(f)

        self.data = [item for item in text_data
                     if item['id'].lower() in image_embeddings and 'text_embedding' in item]
        self.image_embeddings = image_embeddings
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        doc_id = item['id'].lower()
        text_embed = torch.tensor(item['text_embedding'], dtype=torch.float)
        image_embed = torch.tensor(self.image_embeddings[doc_id], dtype=torch.float)
        label = self.label_map[item['label'].lower()]
        return text_embed, image_embed, label

# --------------------- Model ---------------------
class EarlyFusionAttentionModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=512, num_classes=10):
        super(EarlyFusionAttentionModel, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, text_embed, image_embed):
        t = self.text_proj(text_embed)
        i = self.image_proj(image_embed)
        fusion_input = torch.stack([t, i], dim=1)
        attn_out, _ = self.attention(fusion_input, fusion_input, fusion_input)
        flat = attn_out.reshape(attn_out.size(0), -1)
        return self.classifier(flat)

# --------------------- Evaluation ---------------------
def evaluate(model, dataloader, device, label_map):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for text_embed, image_embed, label in dataloader:
            text_embed, image_embed = text_embed.to(device), image_embed.to(device)
            outputs = model(text_embed, image_embed)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(label.tolist())

    print("✅ Classification Report:")
    target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print(classification_report(y_true, y_pred, target_names=target_names))
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy: {acc:.2%}")
    return y_true, y_pred

# --------------------- Main Inference ---------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using device:", device)

    # Load label mapping
    loader = DatasetLoader(parquet_dir="./Data/split")  # adjust if needed
    _, label2id, _ = loader.export_to_json()
    label_map = {k.lower(): v for k, v in label2id.items()}

    # Load test dataset
    test_dataset = TextImageFusionDataset(
        text_embeds_path="data/test_text_embeddings.json",
        image_embeds_path="data/test_image_embeddings.json",
        label_map=label_map
    )

    if len(test_dataset) == 0:
        print("❌ No test samples available. Check embeddings and file paths.")
        exit()

    test_loader = DataLoader(test_dataset, batch_size=8)

    # Load model and weights
    model = EarlyFusionAttentionModel(num_classes=len(label_map)).to(device)
    checkpoint_path = "checkpoints/final_fusion_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Loaded model from {checkpoint_path}")

    # Run inference
    evaluate(model, test_loader, device, label_map)