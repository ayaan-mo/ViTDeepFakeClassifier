# inference.py

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

def load_test_images(folder_path, max_images=3, label=None):
    """
    Load up to max_images from the folder_path.
    Each image is assigned the provided label.
    """
    image_paths = list(Path(folder_path).glob("*"))
    selected = image_paths[:max_images]
    test_images = []
    for p in selected:
        img = Image.open(str(p)).convert("RGB")
        test_images.append({"image": img, "path": str(p), "label": label})
    return test_images

real_test_images = load_test_images("/Users/muhammadhamzasohail/Desktop/Dataset/Test/Real", max_images=50, label=0)
fake_test_images = load_test_images("/Users/muhammadhamzasohail/Desktop/Dataset/Test/Fake", max_images=50, label=1)
test_images = real_test_images + fake_test_images


correct = 0
for img_obj in test_images:
    img = img_obj["image"]
    img_path = img_obj["path"]
    true_label = "real" if img_obj["label"] == 0 else "fake"

    inputs = feature_extractor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = model.config.id2label[str(pred_id)]
    
    if pred_label == true_label:
        correct += 1

    print(f"{Path(img_path).name}: predicted = {pred_label}, actual = {true_label}")

print(f"Accuracy on test images: {correct}/{len(test_images)}")
