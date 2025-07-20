import torch
import json
from PIL import Image
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from src.fusion_model import EarlyFusionAttentionModel
from src.fusion_dataloader import DatasetLoader

def load_ocr_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["image_path"], data["full_text"]

def generate_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

def generate_image_embedding(image_path, transform, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor).squeeze().flatten()
    return embedding

def predict_from_json(json_path, model_path="checkpoints/final_fusion_model.pt", label_map_path="data/label_map.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load cleaned JSON ---
    image_path, full_text = load_ocr_json(json_path)

    # --- Load Tokenizer & Text Model ---
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").eval()

    # --- Load ResNet50 ---
    resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Generate Embeddings ---
    text_emb = generate_text_embedding(full_text, tokenizer, bert_model)
    image_emb = generate_image_embedding(image_path, transform, resnet)

    # --- Load Classifier ---
    loader = DatasetLoader(parquet_dir="./Data/split")
    data, label2id, id2label = loader.export_to_json(output_dir="./data")
    label_map = {k.lower(): v for k, v in label2id.items()}

    # with open(label_map_path, "r") as f:
    #     label_map = json.load(f)
    id2label = {v: k for k, v in label_map.items()}

    model = EarlyFusionAttentionModel(num_classes=len(label_map))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # --- Predict ---
    text_emb = text_emb.unsqueeze(0).to(device)
    image_emb = image_emb.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_emb, image_emb)
        pred = torch.argmax(logits, dim=1).item()
        pred_label = id2label[pred]

    print(f"\n📄 Prediction for: {image_path}")
    print(f"✅ Predicted Class: {pred_label}")

# --- Example usage ---
if __name__ == "__main__":
    json_path = "./temp/Aadhar_1.json"
    predict_from_json(json_path)