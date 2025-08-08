# Handwritten Digits Recognition 🧠✍️
This project implements a full-stack application to recognize handwritten digits using deep learning.

# 🔧 Project Structure
```bash
Handwritten_digits-main/
│
├── backend/
│   ├── src/                  # Model training & prediction logic
│   └── environment.yml       # Backend environment config
│
├── frontend/
│   ├── src/                  # Web UI to draw/submit digits
│   └── environment.yml       # Frontend dependencies
│
├── requirements.txt          # Common requirements
└── README.md                 # Project documentation
```
# ⚙️ Tech Stack
Backend: Python, TensorFlow/Keras (for digit recognition model)

Frontend: HTML/CSS/JS (or Streamlit/Flask GUI)

Deployment: Localhost/Webapp interface

# 🎯 Features
User-friendly interface to draw digits (0–9)

Real-time digit prediction using a trained CNN

Clear modular separation between frontend & backend

# 🚀 Getting Started
Clone the repo
git clone https://github.com/vaishnavishatagopam77/Handwritten_digits-main.git

Navigate to backend and frontend folders to set up environments

cd backend
conda env create -f environment.yml

cd ../frontend
conda env create -f environment.yml
Run the backend server and frontend app

# 📁 Dataset
MNIST handwritten digit dataset from Kaggle

