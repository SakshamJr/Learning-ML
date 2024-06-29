from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import pickle
import uvicorn
import os
from io import BytesIO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your trained model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your class labels
class_labels = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None, "image_url": None})

@app.post("/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Assuming your model outputs a single class index
    class_index = np.argmax(prediction)

    result = class_labels[class_index]

    # Save the uploaded image to the static folder
    image_path = f"static/uploads/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(contents)

    return templates.TemplateResponse("index.html", {"request": request, "result": result, "image_url": image_path})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
