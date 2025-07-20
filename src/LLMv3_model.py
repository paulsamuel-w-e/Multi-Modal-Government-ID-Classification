from transformers import (
    AutoProcessor,
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    default_data_collator,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from llmv3_dataloader import DatasetLoader
import torch
import numpy as np
import torch.nn as nn

# Load dataset
loader = DatasetLoader(parquet_dir="./Data/split")
dataset, label2id, id2label = loader.load_my_data()
num_labels = len(label2id)

# Compute class weights
train_labels = dataset["train"]["labels"]
class_weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom Trainer
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    class_acc = {f"class_{k}_acc": round(v["recall"], 4) for k, v in report.items() if k.isdigit()}
    return {
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **class_acc
    }

# Load processor and model
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    max_steps=1000,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1"
)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=processor,  # Updated to address deprecation warning
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

# Train
trainer.train()

# Save model
output_dir = "./saved_model"
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

print(f"Model saved to: {output_dir}")