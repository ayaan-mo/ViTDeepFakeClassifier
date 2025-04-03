from PIL import Image
from pathlib import Path
import random
import torch
import numpy as np
import evaluate
import logging
from datasets import Dataset, DatasetDict
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

# Setup logging
logging.basicConfig(
    filename="training_log.txt",
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_split_dataset(root_folder, max_per_class=600, val_ratio=0.2, test_ratio=0.1):
    all_data = []

    for label_name, label in [("Real", 0), ("Fake", 1)]:
        folder_path = Path(root_folder) / label_name
        image_paths = list(folder_path.rglob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        selected_paths = random.sample(image_paths, min(max_per_class, len(image_paths)))

        for p in selected_paths:
            try:
                img = Image.open(p).convert("RGB")
                all_data.append({"image": img, "label": label})
            except Exception as e:
                print(f"Could not load image {p}: {e}")

    # Shuffle and split
    random.shuffle(all_data)
    total = len(all_data)
    test_size = int(test_ratio * total)
    val_size = int(val_ratio * total)

    test_data = all_data[:test_size]
    val_data = all_data[test_size:test_size+val_size]
    train_data = all_data[test_size+val_size:]

    return train_data, val_data, test_data

# Load and split the dataset
train_data, val_data, test_data = load_and_split_dataset(
    "/Users/muhammadhamzasohail/Desktop/GAN-Dataset", 
    max_per_class=600
)

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

logging.info("=== ViT Deepfake Training Started ===")
logging.info(f"Training samples: {len(dataset['train'])}")
logging.info(f"Validation samples: {len(dataset['validation'])}")
logging.info(f"Test samples: {len(dataset['test'])}")

# Image Processor
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(example_batch):
    inputs = image_processor([img for img in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

# Apply transform
dataset["train"] = dataset["train"].with_transform(transform)
dataset["validation"] = dataset["validation"].with_transform(transform)

# Collate function
def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

# Load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={"0": "real", "1": "fake"},
    label2id={"real": 0, "fake": 1}
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vit-deepfake-finetune",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
    report_to="none",
    load_best_model_at_end=True
)

# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=image_processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./vit-deepfake-finetune/best_model")
image_processor.save_pretrained("./vit-deepfake-finetune/best_model")

# Evaluate on test set
dataset["test"] = dataset["test"].with_transform(transform)
metrics = trainer.evaluate(eval_dataset=dataset["test"])

test_acc = metrics["eval_accuracy"]
logging.info(f"Test Set Accuracy: {test_acc:.4f}")
logging.info("=== Training Complete ===")

print("Test Set Accuracy:", test_acc)
