from PIL import Image
from pathlib import Path
import random
import torch
import numpy as np
import evaluate
import logging
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

logging.basicConfig(
    filename="training_log.txt",
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_dataset_split(split_folder, max_per_class=200):
    data = []
    for label_name, label in [("Real", 0), ("Fake", 1)]:
        folder_path = Path(split_folder) / label_name
        image_paths = list(folder_path.rglob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        selected_paths = random.sample(image_paths, min(max_per_class, len(image_paths)))

        for p in selected_paths:
            try:
                img = Image.open(p).convert("RGB")
                data.append({"image": img, "label": label})
            except Exception as e:
                print(f"Could not load image {p}: {e}")
    return data

train_data = load_dataset_split("/Users/muhammadhamzasohail/Desktop/dataset/Train", max_per_class=200)
val_data   = load_dataset_split("/Users/muhammadhamzasohail/Desktop/dataset/Validation", max_per_class=200)
test_data  = load_dataset_split("/Users/muhammadhamzasohail/Desktop/dataset/Test", max_per_class=200)

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

logging.info("=== ViT Deepfake Training Started ===")
logging.info(f"Training samples: {len(dataset['train'])}")
logging.info(f"Validation samples: {len(dataset['validation'])}")
logging.info(f"Test samples: {len(dataset['test'])}")

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(example_batch):
    inputs = image_processor([img for img in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

dataset["train"] = dataset["train"].with_transform(transform)
dataset["validation"] = dataset["validation"].with_transform(transform)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch])
    }

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={"0": "real", "1": "fake"},
    label2id={"real": 0, "fake": 1}
)

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

metric = evaluate.load("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=image_processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./vit-deepfake-finetune/best_model")
image_processor.save_pretrained("./vit-deepfake-finetune/best_model")

dataset["test"] = dataset["test"].with_transform(transform)
metrics = trainer.evaluate(eval_dataset=dataset["test"])

test_acc = metrics["eval_accuracy"]
logging.info(f"Test Set Accuracy: {test_acc:.4f}")
logging.info("=== Training Complete ===")

print("Test Set Accuracy:", test_acc)

train_loss = [log["loss"] for log in trainer.state.log_history if "loss" in log]
eval_loss = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
steps_train = [log["step"] for log in trainer.state.log_history if "loss" in log]
steps_eval = [log["step"] for log in trainer.state.log_history if "eval_loss" in log]
plt.figure(figsize=(10, 6))
plt.plot(steps_train, train_loss, label="Training Loss")
plt.plot(steps_eval, eval_loss, label="Validation Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
