
from pathlib import Path
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

model_path = "./vit-deepfake-finetune/best_model"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(
    model_path,
    id2label={"0": "real", "1": "fake"},
    label2id={"real": 0, "fake": 1}
)

image_path = "/Users/muhammadhamzasohail/Desktop/IMG_9570.jpg"

img = Image.open(image_path).convert("RGB")
inputs = feature_extractor(img, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_label = model.config.id2label[str(pred_id)]

print(f"{Path(image_path).name}: predicted = {pred_label}")
