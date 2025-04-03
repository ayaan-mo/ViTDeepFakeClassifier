from PIL import Image
from pathlib import Path
import random
import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

def load_images_from_folder(folder_path, label, max_images=200):
    image_paths = list(Path(folder_path).glob("*"))
    selected = random.sample(image_paths, min(len(image_paths), max_images))
    return [{"image": Image.open(str(p)).convert("RGB"), "label": label} for p in selected]

real_images = load_images_from_folder("/Users/muhammadhamzasohail/Desktop/Test/Real", label=0)
fake_images = load_images_from_folder("/Users/muhammadhamzasohail/Desktop/Test/Fake", label=1)

data = real_images + fake_images
random.shuffle(data)

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
prepared_ds = {"train": dataset["train"], "validation": dataset["test"]}

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform(example_batch):
    inputs = image_processor([img for img in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

prepared_ds["train"] = prepared_ds["train"].with_transform(transform)
prepared_ds["validation"] = prepared_ds["validation"].with_transform(transform)

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
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=image_processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./vit-deepfake-finetune/best_model")
image_processor.save_pretrained("./vit-deepfake-finetune/best_model")

metrics = trainer.evaluate()
print("Evaluation Accuracy:", metrics["eval_accuracy"])

model = ViTForImageClassification.from_pretrained(
    "./vit-deepfake-finetune/best_model",
    id2label={"0": "real", "1": "fake"},
    label2id={"real": 0, "fake": 1}
)
image_processor = ViTImageProcessor.from_pretrained("./vit-deepfake-finetune/best_model")