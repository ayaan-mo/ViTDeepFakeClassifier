from pathlib import Path
from PIL import Image
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load model
model_path = "./vit-deepfake-finetune/best_model"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(
    model_path,
    id2label={"0": "real", "1": "fake"},
    label2id={"real": 0, "fake": 1}
)
model.eval()

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = model.config.id2label[str(pred_id)]
    return pred_label

def choose_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp")]
    )
    if file_path:
        pred_label = predict_image(file_path)
        messagebox.showinfo("Prediction", f"Prediction: {pred_label}")

# GUI setup
root = tk.Tk()
root.title("Deepfake Detector")
root.geometry("300x100")

btn = tk.Button(root, text="Choose Image", command=choose_and_predict, font=("Helvetica", 14))
btn.pack(expand=True)

root.mainloop()
