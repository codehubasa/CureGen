# Brain Tumor Detection AI

A Flask-based web application for detecting brain tumors from MRI and CT scan images using deep learning models.

## Features

- **MRI & CT Scan Support**: Analyze both MRI and CT scan images
- **Real-time Predictions**: Fast inference using pre-trained TensorFlow models
- **Detailed Results**: Shows confidence scores and class probabilities
- **PDF Report Generation**: Download analysis results as PDF reports
- **Drag & Drop Upload**: Easy image upload interface
- **Progress Visualization**: Real-time processing progress indicator

## Supported Tumor Types

- Glioma
- Meningioma
- Pituitary
- No Tumor

## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Installation

1. **Clone or download the project**
   ```bash
   cd "Brain tumar detection"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure models are in place**
   - `models/brain_tumor_mri_model.keras` - MRI model
   - `models/brain_tumor_model.keras` - CT scan model

## Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open in browser**
   ```
   http://localhost:5000
   ```

3. **Upload and Analyze**
   - Select scan type (MRI or CT)
   - Upload an image
   - Click "Start AI Processing"
   - View results and download PDF report

## API Endpoints

### GET Routes

- **`/`** - Home page with introduction
- **`/upload-page`** - Upload and analysis page
- **`/result`** - Result display page
- **`/health`** - Health check endpoint (returns model status)
- **`/uploads/<filename>`** - Serve uploaded images

### POST Routes

- **`/predict`** - Main prediction endpoint
  - **Parameters:**
    - `image` (file): Image file to analyze
    - `scan_type` (form): "mri" or "ct"
  - **Response:** JSON with prediction result, confidence, and class probabilities

## Project Structure

```
Brain tumar detection/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── models/
│   ├── brain_tumor_mri_model.keras # Pre-trained MRI model
│   └── brain_tumor_model.keras     # Pre-trained CT scan model
├── templates/
│   ├── index.html                  # Home page
│   ├── detection.html              # Upload and processing page
│   ├── result.html                 # Results display page
│   ├── upload.html                 # Alternative upload page
│   └── auth.html                   # Authentication page (optional)
└── uploads/                        # Temporary uploaded images (auto-created)
```

## Error Handling

The application includes comprehensive error handling for:
- Missing or invalid image files
- Unsupported file formats
- File size limits (16MB max)
- Model loading failures
- Prediction errors

## Configuration

Edit `app.py` to modify:
- **UPLOAD_FOLDER** - Directory for uploaded images
- **ALLOWED_EXTENSIONS** - Supported image formats
- **MAX_CONTENT_LENGTH** - Maximum upload file size
- **IMG_SIZE** - Input image size for models (default: 224x224)

## Performance Notes

- Processing time depends on model complexity and hardware
- GPU acceleration is recommended for faster predictions
- Typical prediction takes 1-5 seconds on CPU

## Troubleshooting

1. **Model not found error**
   - Ensure model files exist in `models/` directory
   - Check file names match exactly

2. **Port already in use**
   - Change port in `app.py`: `app.run(debug=True, port=5001)`

3. **Uploads folder not created**
   - App auto-creates it on first run
   - Or manually create `uploads/` directory

4. **TensorFlow performance issues**
   - Consider upgrading to GPU-enabled TensorFlow
   - Use `tensorflow-gpu` instead of `tensorflow`

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, check:
1. Console output for error messages
2. `/health` endpoint to verify model status
3. Ensure all dependencies are correctly installed
