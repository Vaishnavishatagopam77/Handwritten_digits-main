"""
Digits pretrained classifier using ViT.
Source: https://huggingface.co/farleyknight-org-username/vit-base-mnist
"""

import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor

pretrained_model = "farleyknight-org-username/vit-base-mnist"

# Create a ViT image processor for the pretrained model
processor = ViTImageProcessor.from_pretrained(pretrained_model)

# Load the pretrained ViT model for image classification
model = AutoModelForImageClassification.from_pretrained(pretrained_model)


def classify_digit(img):
    """
    Classify a given image of a digit using the pretrained ViT model.

    Args:
        img (PIL.Image.Image): The input image to be classified.

    Returns:
        A numpy array of probabilities for each digit class (0-9).
    """
    
    # Step 1: Preprocess the input image
    # - `processor` is a ViTImageProcessor from Hugging Face.
    # - It resizes, normalizes, and converts the image into tensor format.
    # - return_tensors="pt" tells it to return PyTorch tensors.
    inputs = processor(images=img, return_tensors="pt")
    
    # Step 2: Perform inference using the pretrained ViT model
    # - The model expects inputs as keyword arguments (like **inputs)
    # - The model outputs raw scores (logits) for each class (0-9)
    prob = model(**inputs)

    # Step 3: Apply softmax to convert logits into class probabilities
    prob = torch.nn.functional.softmax(prob.logits, dim=1)[0]

    # Step 4: Detach the tensor from the computation graph
    # - .detach() removes gradient tracking (not needed in inference).
    # - .numpy() converts the PyTorch tensor into a NumPy array.
    prob = prob.detach().numpy()

    # Step 5: Return the probability array
    # - Output will look like: [0.01, 0.03, ..., 0.85, 0.04]
    # - Each value corresponds to the probability of digits 0–9
    return prob
