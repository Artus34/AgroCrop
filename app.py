import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Crop Recommendation API",
    description="An API to recommend the best crop to plant based on environmental factors.",
    version="1.0.0"
)

# --- 2. Load Saved Artifacts ---
# Load the necessary files that were created during model training.
try:
    with open('Best_Crop_Recommendation_Model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('crop_dict.pkl', 'rb') as f:
        crop_dict = pickle.load(f)
        # Create a reverse mapping from number to crop name
        reverse_crop_map = {v: k for k, v in crop_dict.items()}

except FileNotFoundError as e:
    raise RuntimeError(f"Could not load a necessary file: {e}. Please ensure all .pkl files from the notebook are in the same directory.")

# --- 3. Define the API Input Model ---
# This class defines the structure for the data you'll send to the API.
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

    # This provides a default example for the API documentation
    class Config:
        json_schema_extra = {
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.88,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.9
            }
        }

# --- 4. Define API Endpoints ---
@app.get("/", summary="API Root", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Crop Recommendation API!"}

@app.post("/recommend", summary="Recommend a Crop", tags=["Prediction"])
def recommend_crop(data: CropInput):
    """
    Receives environmental factors and returns a recommended crop.

    - **N**: Nitrogen content in the soil.
    - **P**: Phosphorus content in the soil.
    - **K**: Potassium content in the soil.
    - **temperature**: Temperature in Celsius.
    - **humidity**: Relative humidity in %.
    - **ph**: pH value of the soil.
    - **rainfall**: Rainfall in mm.
    """
    try:
        # Step A: Convert the input data into a NumPy array
        # The order of features must be exactly the same as in the training data
        input_array = np.array([[
            data.N, data.P, data.K, data.temperature,
            data.humidity, data.ph, data.rainfall
        ]])

        # Step B: Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_array)

        # Step C: Make a prediction
        predicted_crop_num = model.predict(input_scaled)[0]

        # Step D: Map the predicted number back to the crop name
        recommended_crop_name = reverse_crop_map.get(predicted_crop_num)

        if not recommended_crop_name:
            raise HTTPException(status_code=500, detail="Could not find a crop name for the prediction.")

        return {"recommended_crop": recommended_crop_name.capitalize()}

    except Exception as e:
        # For any other errors, return a generic server error
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

