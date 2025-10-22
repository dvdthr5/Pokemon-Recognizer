from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np, io, json
from config import MODEL_PATH, CLASS_NAMES_PATH

app = FastAPI()
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((180, 180))
    x = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    pred = model.predict(x, verbose=0)
    idx = int(np.argmax(pred))
    conf = float(np.max(pred))
    return {"class": class_names[idx], "confidence": conf}