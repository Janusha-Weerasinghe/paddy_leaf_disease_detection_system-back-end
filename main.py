from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import logging
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Setup CORS to allow requests from specified origins
origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to the pre-trained models (loaded from environment variables)


MODEL_PATHS = {
    "plant_classifier": "C:/Users/Janusha_chamali/PycharmProjects/PLDDS_API/Models",
    "model1": "C:/Users/Janusha_chamali/PycharmProjects/PLDDS_API/Models/Model1/Model_T_v2.h5",
    "model2": "C:/Users/Janusha_chamali/PycharmProjects/PLDDS_API/Models/Model2/Model_P_v5.h5",
   
}

# Classes corresponding to each model
MODEL_CLASSES = {
    "plant_classifier": ["Blight", "Blast ", "Turngo", "Healthy"],
    "model1": [
        "Blight",
        "Blast",
        "Turngo",
        "Healthy"
    ],
    "model2": [
        "leaf_area", 
        "Des_area", 
        
    ],
}

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Load models
MODELS = {}
# for model_id, model_path in MODEL_PATHS.items():
#     if not model_path or not os.path.exists(model_path):
#         logging.error(f"Model file for {model_id} not found at path: {model_path}")
#         continue
#     try:
#         MODELS[model_id] = tf.keras.models.load_model(model_path)
#         logging.info(f"Successfully loaded {model_id} model from {model_path}")
#     except Exception as e:
#         logging.error(f"Error loading {model_id} model from {model_path}: {e}")

for model_id, model_path in MODEL_PATHS.items():
    if not model_path or not os.path.exists(model_path):
        logging.error(f"Model file for {model_id} not found at path: {model_path}")
        continue
    try:
        # Try loading the model without compile to avoid batch_shape errors
        MODELS[model_id] = tf.keras.models.load_model(model_path, compile=False)
        logging.info(f"Successfully loaded {model_id} model from {model_path}")
    except Exception as e:
        logging.error(f"Error loading {model_id} model from {model_path}: {e}")



# Define endpoint for checking server status
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Function to read uploaded file as an image
def read_file_as_image(data: bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Function to classify the plant type
# async def classify_plant(image: Image.Image) -> str:
#     image = image.resize((224, 224))
#     image_array = np.array(image) / 255.0
#     img_batch = np.expand_dims(image_array, axis=0)
#     predictions = MODELS["plant_classifier"].predict(img_batch)
#     predicted_class_idx = np.argmax(predictions[0])
#     predicted_plant = MODEL_CLASSES["plant_classifier"][predicted_class_idx]
#     return predicted_plant

async def classify_plant2(image: Image.Image) -> str:
    # Resize image to model's expected input size
    image = image.resize((224, 224))

    # Convert image to numpy array and normalize pixels
    image_array = np.array(image).astype(np.float32) / 255.0

    # Add batch dimension: shape becomes (1, 224, 224, 3)
    img_batch = np.expand_dims(image_array, axis=0)

    # Predict using the loaded plant classifier model
    predictions = MODELS["plant_classifier"].predict(img_batch)

    # Get index of highest probability class
    predicted_class_idx = np.argmax(predictions[0])

    # Map index to plant class name
    predicted_plant = MODEL_CLASSES["plant_classifier"][predicted_class_idx]

    return predicted_plant

async def classify_plant(image: Image.Image, model_id: str) -> str:
    # Resize image to match the model input dimensions
    image = image.resize((224, 224))

    # Convert image to numpy array and normalize pixel values
    image_array = np.array(image) / 255.0

    # Expand dimensions to create batch
    img_batch = np.expand_dims(image_array, axis=0)

    # Make predictions using the appropriate model
    model = MODELS.get(model_id)  # Get the model corresponding to model_id

    if model is None:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found.")

    predictions = model.predict(img_batch)

    # Get predicted class index and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = MODEL_CLASSES.get(model_id, [])[predicted_class_idx]
    return predicted_class



@app.post("/predict/{model_id}")
async def predict(model_id: str, file: UploadFile = File(...)):
    # Check if the model ID is valid and the model is loaded
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found or failed to load.")

    # Check if the uploaded file is an image
    if not file.content_type.startswith('image'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # Read file as image
        image = Image.open(BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        # Classify the plant type using the correct model_id
        plant_type = await classify_plant(image, model_id)

        # Check if the plant type matches the model's expected plant
        model_to_plant = {
            "model1": "Model1",
            "model2": "Model2",
           
        }

        expected_plant_type = model_to_plant.get(model_id, "")
        # if plant_type != expected_plant_type:
        #      raise HTTPException(status_code=400,
        #                          detail=f"Uploaded image is of a {plant_type}, but model {model_id} expects a {expected_plant_type}.")

        # Resize image to match model input dimensions
        image = image.resize((224, 224))

        # Convert image to numpy array and normalize pixel values
        image_array = np.array(image) / 255.0

        # Expand dimensions to create batch
        img_batch = np.expand_dims(image_array, axis=0)

        # Make predictions
        predictions = MODELS[model_id].predict(img_batch)

        # Get predicted class index and confidence
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = MODEL_CLASSES[model_id][predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        return {
            'class': predicted_class,
            'confidence': confidence
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# Function to classify the plant type
async def classify_plant(image: Image.Image, model_id: str) -> str:
    # Debugging log to check if model is loaded correctly
    logging.info(f"Using model: {model_id} for classification")

    if model_id not in MODELS:
        logging.error(f"Model {model_id} not found!")
        raise Exception(f"Model {model_id} is not loaded correctly.")

    # Resize image to match plant_classifier model input dimensions
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    img_batch = np.expand_dims(image_array, axis=0)

    # Debugging log for image shape
    logging.info(f"Image batch shape: {img_batch.shape}")

    try:
        # Make predictions using the plant_classifier model
        predictions = MODELS[model_id].predict(img_batch)
        logging.info(f"Prediction successful: {predictions}")
    except Exception as e:
        logging.error(f"Error making predictions with {model_id}: {e}")
        raise e

    # Get predicted class index and return the plant type
    predicted_class_idx = np.argmax(predictions[0])
    predicted_plant = MODEL_CLASSES[model_id][predicted_class_idx]

    # Debugging log for output
    logging.info(f"Predicted plant type: {predicted_plant}")

    return predicted_plant


# Function to classify the plant type
async def classify_plant2(image: Image.Image, model_id: str) -> str:
    # Resize image to match plant_classifier model input dimensions
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    img_batch = np.expand_dims(image_array, axis=0)

    # Make predictions using the plant_classifier model
    predictions = MODELS["plant_classifier"].predict(img_batch)

    # Get predicted class index and return the plant type
    predicted_class_idx = np.argmax(predictions[0])
    predicted_plant = MODEL_CLASSES["plant_classifier"][predicted_class_idx]
    return predicted_plant

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
