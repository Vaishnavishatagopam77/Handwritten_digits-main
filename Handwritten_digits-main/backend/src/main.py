"""
This is a FastAPI application that receives an HTTP GET request at the root
endpoint ("/") and returns the classification probability of a digit image.
"""

# Import necessary libraries
import numpy as np                          # For handling numerical arrays
from fastapi import FastAPI, Request        # FastAPI framework and HTTP Request object
from model.predict import classify_digit    # Custom digit classification function

# Create an instance of the FastAPI application
app = FastAPI()

# Define a GET endpoint at the root URL ("/")
@app.get("/")
async def get_classify_digit(info: Request):  # This function handles GET requests and accepts request data
    # Await and parse the JSON body from the GET request
    req_info = await info.json()              # Extracts and reads JSON data from the incoming request

    # Convert the 'image' data into a NumPy array
    img = np.array(req_info["image"])         # Converts list to NumPy array for model compatibility

    # Pass the image to the classification function to get prediction probabilities
    prob = classify_digit(img)                # Returns a array of class probabilities

    # Convert the probabilities into a standard Python list and return as a JSON response
    return {"prob": prob.tolist()}            # Convert NumPy array to list for JSON serialization
