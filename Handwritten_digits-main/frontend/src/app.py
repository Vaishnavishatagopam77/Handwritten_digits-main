"""
Project: Prototyping a Machine Learning Application with Streamlit.
Streamlit app integrated with a pretrained ViT model for image classification.
"""

# Importing required libraries
import os  # To access environment variables like backend URL
import cv2  # For image processing and color space conversion
import matplotlib.pyplot as plt  # For plotting prediction probabilities
import numpy as np  # For numerical operations and handling arrays
import requests  # To send HTTP requests to the backend
import streamlit as st  # To build the web app interface
from streamlit_drawable_canvas import st_canvas  # For drawable canvas component in Streamlit

# Canvas size (square) for user input image
CANVAS_SIZE = 250


def classify_digit(img):
    """
    Sends a request to the backend with the image data
    to perform a classification on the image.
    """
    # Retrieve the URL of the backend from the environment variables
    url_backend = os.environ["URL_BACKEND"]

    # Send a GET request to the backend with the image data as a JSON payload
    request = requests.get(url_backend, json={"image": img.tolist()})

    # Retrieve the predicted probabilities from the response
    answer = request.json()
    prob = answer["prob"]

    # Return probabilities as a NumPy array
    return np.array(prob)


def main():
    """
    Main Streamlit function
    Read an image and show a probability
    """

    # Set the title and caption for the Streamlit app
    st.set_page_config(page_title="Handwritten Digits Recognition")
    st.title("Handwritten Digits Recognition")
    st.caption(
        "App integrated with a pretrained ViT model for image classification"
    )

    # Initialize variables for the probability and the canvas image
    probs = None  # Will hold model predictions
    canvas_image = None  # Will hold the user-drawn image

    # Create two columns for the Streamlit app layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(":1234: Draw a number here")

        # Create a canvas component for users to draw an image
        canvas_image = st_canvas(
            fill_color="black",  # Background color of canvas
            stroke_width=20,  # Thickness of the stroke
            stroke_color="gray",  # Color of the stroke (drawing color)
            width=CANVAS_SIZE,  # Width of the canvas
            height=CANVAS_SIZE,  # Height of the canvas
            drawing_mode="freedraw",  # Freehand drawing
            key="canvas",  # Unique key for Streamlit session state
            display_toolbar=True,  # Show drawing toolbar
        )

        # If canvas is drawn and image data is available
        if canvas_image is not None and canvas_image.image_data is not None:

            # Button to trigger classification
            if st.button("Classify!"):
                # Convert the canvas image to RGB format using OpenCV
                img = cv2.cvtColor(canvas_image.image_data, cv2.COLOR_RGBA2RGB)

                # Show a spinner while waiting for backend response
                with st.spinner("Wait for it..."):
                    # Classify the digit and store the predicted probabilities
                    probs = classify_digit(img) * 100.0  # Convert to percentage

    # If predictions are available, display them
    if probs is not None:
        with col2:
            st.subheader(":white_check_mark: Prediction")

            # Display the predicted digit and its probability
            st.metric(label="Predicted digit:", value=f"{probs.argmax()}")

            # Plot the predicted probabilities as a horizontal bar chart
            fig, ax = plt.subplots(figsize=(6, 4))
            class_names = list(range(10))  # Digits 0 to 9

            # Create a horizontal bar plot
            ax.barh(class_names, probs, height=0.55, align="center")

            # Display probability values next to each bar
            for i, (c, p) in enumerate(zip(class_names, probs)):
                ax.text(p + 2, i - 0.2, f"{p:.2f}%")  # Add text next to each bar

            # Styling and axis formatting
            ax.grid(axis="x")
            ax.set_xlabel("Probability")
            ax.set_xlim([0, 120])
            ax.set_xticks(range(0, 101, 20))
            ax.set_ylabel("Digit")
            ax.set_yticks(range(10))
            fig.tight_layout()

            # Show the plot in Streamlit
            st.pyplot(fig)


# Entry point for Streamlit app
if __name__ == "__main__":
    main()
