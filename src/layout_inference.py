from transformers import AutoProcessor, LayoutLMv3ForSequenceClassification, default_data_collator, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from llmv3_dataloader import DatasetLoader
import torch
import numpy as np
import pandas as pd

# Initialize DatasetLoader
loader = DatasetLoader(parquet_dir="./Data/split")

# Load test dataset
test_dataset, label2id, id2label = loader.load_test_data()
num_labels = len(label2id)

# Load processor and model
processor = AutoProcessor.from_pretrained("./saved_model", apply_ocr=False)
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "./saved_model",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    class_acc = {f"class_{id2label[int(k)]}_acc": round(v["recall"], 4) for k, v in report.items() if k.isdigit()}
    return {
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **class_acc
    }

# Initialize Trainer for inference
trainer = Trainer(
    model=model,
    args=None,
    eval_dataset=test_dataset,
    processing_class=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

# Perform inference
print("Running inference on test set...")
predictions = trainer.predict(test_dataset)
metrics = predictions.metrics

# Print metrics
print("\nTest Set Metrics:")
print(f"Overall Accuracy: {metrics['test_overall_accuracy']:.4f}")
print(f"Macro F1: {metrics['test_macro_f1']:.4f}")
print(f"Weighted F1: {metrics['test_weighted_f1']:.4f}")
print("Class-wise Accuracy (Recall):")
for key, value in metrics.items():
    if key.startswith("test_class_"):
        print(f"{key.replace('test_', '')}: {value:.4f}")

# Save predictions
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
results = {
    "predictions": [id2label[p] for p in preds],
    "labels": [id2label[l] for l in labels]
}
pd.DataFrame(results).to_csv("./test_results/test_predictions.csv", index=False)
print("\nPredictions saved to ./test_results/test_predictions.csv")