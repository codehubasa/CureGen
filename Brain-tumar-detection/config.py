# Configuration settings for Brain Tumor Detection App

import os

# Flask Configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Upload Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model Configuration
MODEL_MRI = 'models/brain_tumor_mri_model.keras'
MODEL_CT = 'models/brain_tumor_model.keras'

# Image Processing
IMG_SIZE = (224, 224)

# Model Classes
TUMOR_CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

# Security
SECRET_KEY = 'your-secret-key-here-change-in-production'

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
