import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import pandas as pd
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from src.fusion_dataloader import DatasetLoader

# Generate and Save Image Embeddings
def generate_image_embeddings(image_paths, output_json):
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    embeddings = {}
    for img_path in tqdm(image_paths, desc="Generating image embeddings"):
        try:
            id = os.path.splitext(os.path.basename(img_path))[0].lower()
            image = Image.open(img_path).convert("RGB")
            image = image_transform(image).unsqueeze(0)
            with torch.no_grad():
                emb = resnet(image).squeeze().numpy().tolist()
                embeddings[id] = emb
        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")

    with open(output_json, 'w') as f:
        json.dump(embeddings, f)
    print(f"✅ Saved image embeddings to {output_json} ({len(embeddings)} embeddings)")

# Generate Text Embeddings and Save
def generate_text_embeddings(json_data, output_json_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval()

    data = []
    for item in json_data:
        text = item.get("text", "")
        if not text:
            print(f"❌ Skipping item {item['id']} due to missing or empty text")
            continue
        data.append({
            "id": item["id"].lower(),
            "text": text,
            "label": item["label"],
            "image_path": item["image_path"]
        })

    for item in tqdm(data, desc="Generating text embeddings"):
        text = item.get("text", "")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            item["text_embedding"] = outputs.last_hidden_state[:, 0, :].squeeze(0).tolist()

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved text embeddings to {output_json_path} ({len(data)} embeddings)")

# Dataset class
class TextImageFusionDataset(Dataset):
    def __init__(self, text_embeds_path, image_embeds_path, label_map, indices=None):
        with open(text_embeds_path, 'r') as f:
            text_data = json.load(f)
        self.text_data = [text_data[i] for i in indices] if indices is not None else text_data
        with open(image_embeds_path, 'r') as f:
            self.image_embeddings = json.load(f)
        self.label_map = label_map
        self.data = [item for item in self.text_data if item['id'].lower() in self.image_embeddings and 'text_embedding' in item]
        print(f"✅ Loaded {len(self.data)} matched text-image pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        doc_id = item['id'].lower()
        text_embed = torch.tensor(item['text_embedding'], dtype=torch.float)
        image_embed = torch.tensor(self.image_embeddings[doc_id], dtype=torch.float)
        label = self.label_map[item['label'].lower()]
        return text_embed, image_embed, label

# Fusion Model
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
        out = self.classifier(flat)
        return out

# Helper functions
def get_class_weights(label_list, label_map, device):
    labels = [label_map[label.lower()] for label in label_list]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def weighted_balanced_accuracy(y_true, y_pred, class_weights):
    unique_classes = np.unique(y_true)
    recall_per_class = []
    for cls in unique_classes:
        tp = sum((np.array(y_true) == cls) & (np.array(y_pred) == cls))
        fn = sum((np.array(y_true) == cls) & (np.array(y_pred) != cls))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class.append(recall)
    weighted_recall = sum(class_weights[i] * recall for i, recall in enumerate(recall_per_class))
    return weighted_recall

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for text_embed, image_embed, label in dataloader:
        text_embed, image_embed, label = text_embed.to(device), image_embed.to(device), label.to(device)
        output = model(text_embed, image_embed)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (output.argmax(dim=1) == label).sum().item()
        total += label.size(0)
    acc = total_correct / total
    return total_loss / len(dataloader), acc

def evaluate(model, dataloader, device, label_map):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for text_embed, image_embed, label in dataloader:
            text_embed, image_embed = text_embed.to(device), image_embed.to(device)
            output = model(text_embed, image_embed)
            preds = output.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(label.tolist())
    target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    print(classification_report(y_true, y_pred, target_names=target_names))
    return accuracy_score(y_true, y_pred), y_true, y_pred

def plot_accuracy(train_acc_list, val_acc_list, fold=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    title = f'Training and Validation Accuracy (Fold {fold})' if fold else 'Training and Validation Accuracy'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'checkpoints/accuracy_plot_fold_{fold}.png' if fold else 'checkpoints/accuracy_plot.png'
    os.makedirs('checkpoints', exist_ok=True)
    plt.savefig(filename)
    plt.close()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Saved model to {path}")

def run_kfold_training(text_embeds_path, image_embeds_path, label_map, num_folds=5, epochs=10, batch_size=8, lr=1e-4, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(text_embeds_path, 'r') as f:
        text_data = json.load(f)
    all_labels = [item["label"].lower() for item in text_data]
    all_ids = [item["id"].lower() for item in text_data]
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    best_wba = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_ids, all_labels), 1):
        print(f"\n🔁 Fold {fold}/{num_folds}")
        train_dataset = TextImageFusionDataset(text_embeds_path, image_embeds_path, label_map, indices=train_idx)
        val_dataset = TextImageFusionDataset(text_embeds_path, image_embeds_path, label_map, indices=val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        class_weights = get_class_weights([text_data[i]['label'] for i in train_idx], label_map, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model = EarlyFusionAttentionModel(num_classes=len(label_map)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        train_acc_list, val_acc_list = [], []
        fold_best_wba = 0
        early_stop_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, y_true, y_pred = evaluate(model, val_loader, device, label_map)
            wba = weighted_balanced_accuracy(y_true, y_pred, class_weights.cpu().numpy())
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | WBA: {wba:.4f}")
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            if wba > fold_best_wba:
                fold_best_wba = wba
                save_model(model, f"checkpoints/best_model_fold_{fold}.pt")
                if wba > best_wba:
                    best_wba = wba
                    best_model_state = model.state_dict()
                    save_model(model, "checkpoints/best_model.pt")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            scheduler.step(wba)
        plot_accuracy(train_acc_list, val_acc_list, fold)
        fold_results.append(fold_best_wba)
    print(f"\nCross-Validation Results: Mean WBA = {np.mean(fold_results):.4f}, Std = {np.std(fold_results):.4f}")
    return best_model_state, best_wba

# Main Pipeline
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset using DatasetLoader
    loader = DatasetLoader(parquet_dir="./Data/split")
    data, label2id, id2label = loader.export_to_json(output_dir="./data")

    # Generate image and text embeddings for train, validation, and combined
    for split in ["train", "validation", "test"]:
        image_paths = [item["image_path"] for item in data[split]]
        generate_image_embeddings(image_paths, f"data/{split}_image_embeddings.json")
        generate_text_embeddings(data[split], f"data/{split}_text_embeddings.json")
    combined_data = data["train"] + data["validation"]
    combined_image_paths = [item["image_path"] for item in combined_data]
    generate_image_embeddings(combined_image_paths, "data/combined_image_embeddings.json")
    generate_text_embeddings(combined_data, "data/combined_text_embeddings.json")

    # Verify matching IDs
    with open("data/combined_text_embeddings.json") as f:
        text_ids = [item["id"].lower() for item in json.load(f)]
    with open("data/combined_image_embeddings.json") as f:
        image_ids = list(json.load(f).keys())
    print(f"Combined text IDs (sample):", text_ids[:5])
    print(f"Combined image IDs (sample):", image_ids[:5])
    print(f"Combined matched IDs: {len(set(text_ids) & set(image_ids))}/{len(text_ids)}")

    # Create label map
    label_map = {k.lower(): v for k, v in label2id.items()}

    # Run k-fold training
    best_model_state, best_wba = run_kfold_training(
        text_embeds_path="data/combined_text_embeddings.json",
        image_embeds_path="data/combined_image_embeddings.json",
        label_map=label_map,
        num_folds=5,
        epochs=10,
        batch_size=8,
        lr=1e-4,
        patience=3
    )
    print(f"Best cross-validation WBA: {best_wba:.4f}")

    # Train final model on combined data
    combined_dataset = TextImageFusionDataset(
        text_embeds_path="data/combined_text_embeddings.json",
        image_embeds_path="data/combined_image_embeddings.json",
        label_map=label_map
    )
    val_dataset = TextImageFusionDataset(
        text_embeds_path="data/validation_text_embeddings.json",
        image_embeds_path="data/validation_image_embeddings.json",
        label_map=label_map
    )

    if len(combined_dataset) > 0 and len(val_dataset) > 0:
        train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        class_weights = get_class_weights([item['label'] for item in combined_data], label_map, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        final_model = EarlyFusionAttentionModel(num_classes=len(label_map)).to(device)
        final_model.load_state_dict(best_model_state)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        train_acc_list, val_acc_list = [], []
        best_final_wba = 0

        for epoch in range(10):
            train_loss, train_acc = train_one_epoch(final_model, train_loader, optimizer, criterion, device)
            val_acc, y_true, y_pred = evaluate(final_model, val_loader, device, label_map)
            wba = weighted_balanced_accuracy(y_true, y_pred, class_weights.cpu().numpy())
            print(f"Final Training Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2%}, Val Acc = {val_acc:.2%}, WBA = {wba:.4f}")
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            if wba > best_final_wba:
                best_final_wba = wba
                save_model(final_model, "checkpoints/final_fusion_model.pt")
            scheduler.step(wba)
        plot_accuracy(train_acc_list, val_acc_list)

        # Evaluate on test set (if available)
        test_data = data['test']
        generate_image_embeddings([item["image_path"] for item in test_data], "data/test_image_embeddings.json")
        generate_text_embeddings(test_data, "data/test_text_embeddings.json")
        test_dataset = TextImageFusionDataset(
            text_embeds_path="data/test_text_embeddings.json",
            image_embeds_path="data/test_image_embeddings.json",
            label_map=label_map
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=8)
            test_acc, y_true, y_pred = evaluate(final_model, test_loader, device, label_map)
            test_wba = weighted_balanced_accuracy(y_true, y_pred, class_weights.cpu().numpy())
            print(f"Test Set: Acc = {test_acc:.2%}, WBA = {test_wba:.4f}")
    else:
        print("❌ No matched samples found in combined or validation datasets. Check filenames and IDs.")