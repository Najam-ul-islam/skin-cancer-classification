from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
from io  import BytesIO
import tensorflow as tf
import uvicorn

app = FastAPI()

def deserialize_sparse_categorical_crossentropy(config, custom_objects=None):
    """Manually deserializes SparseCategoricalCrossentropy for older models."""
    return tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=config['config'].get('reduction'),
        name=config['config'].get('name'),
        from_logits=config['config'].get('from_logits'),
        ignore_index=config['config'].get('ignore_class')
    )

# Load your model, passing the custom deserialization function
custom_objects = {'SparseCategoricalCrossentropy': deserialize_sparse_categorical_crossentropy}
try:
    MODEL = tf.keras.models.load_model('CNNmodel.h5', custom_objects=custom_objects)
except OSError as e:
    print(f"Error loading model: {e}")


CLASS_NAMES = ["benign", "malignant"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get('/')
def home():
    return {"message": "Welcome to Skin Cancer Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    if confidence >=0.60:
        return {"class": predicted_class,"confidence": float(confidence)}
    else:
        return {"class": "unknown","confidence": float(confidence)}




if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)