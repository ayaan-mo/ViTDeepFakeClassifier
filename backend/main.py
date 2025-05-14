from pathlib import Path
from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import shutil

# Local model path
model_path = "backend/vit-deepfake-finetune/best_model"

# Initialize FastAPI app
app = FastAPI()

# Use AutoImageProcessor (works with preprocessor_config.json)
# Use AutoImageProcessor (works with preprocessor_config.json)
processor = AutoImageProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=True  # Set to True to use the fast processor
)


model = ViTForImageClassification.from_pretrained(
    model_path,
    local_files_only=True,
    id2label={"0": "real", "1": "fake"},
    label2id={"real": 0, "fake": 1}
)

# Function to process and predict on the uploaded image
def predict_image(image: Image.Image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = model.config.id2label[str(pred_id)]
    return pred_label

@app.post("/predict/")
async def upload_file(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    prediction = predict_image(image)
    return {"filename": file.filename, "prediction": prediction}