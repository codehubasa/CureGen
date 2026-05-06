# Quick Start Guide

## Step 1: Install Python Dependencies

Open Command Prompt or PowerShell and navigate to your project folder:

```bash
cd "C:\Users\ankan\Desktop\Brain tumar detection"
```

Install required packages:

```bash
pip install -r requirements.txt
```

## Step 2: Verify Model Files

Ensure both model files exist in the `models/` folder:
- `models/brain_tumor_mri_model.keras` 
- `models/brain_tumor_model.keras`

If models don't exist, place them in the models directory.

## Step 3: Run the Application

Start the Flask development server:

```bash
python app.py
```

You should see output like:
```
============================================================
Brain Tumor Detection API
============================================================
Upload folder: C:\Users\ankan\Desktop\Brain tumar detection\uploads
Models loaded:
  - MRI model: ✓
  - CT model: ✓
============================================================
```

## Step 4: Access the Application

Open your web browser and go to:
```
http://localhost:5000
```

## Step 5: Test the Application

1. **Home Page**: Click "Get Started" or navigate to the detection page
2. **Select Scan Type**: Choose between MRI or CT scan
3. **Upload Image**: Drag and drop or click to upload a medical image
4. **Process**: Click "Start AI Processing" button
5. **View Results**: See predictions with confidence scores
6. **Download Report**: Get a PDF report of the analysis

## Troubleshooting

### Issue: Models not loading
**Solution**: 
- Check that model files exist with exact names
- Verify file paths in app.py are correct
- Check the console output for error messages

### Issue: Port 5000 already in use
**Solution**:
Open `app.py` and change the port:
```python
app.run(debug=True, port=5001)  # Use 5001 instead
```

Then access: `http://localhost:5001`

### Issue: TensorFlow takes long to import
**Solution**: 
This is normal on first run. Subsequent runs will be faster.
For GPU acceleration, install: `pip install tensorflow-gpu`

### Issue: Out of memory
**Solution**:
- Close other applications
- Reduce batch processing if needed
- Consider using GPU-accelerated TensorFlow

## Advanced Configuration

Edit `config.py` to modify:
- **PORT**: Change server port
- **MAX_FILE_SIZE**: Adjust max upload size
- **IMG_SIZE**: Change model input size (if using different models)

## File Uploads

Uploaded images are stored in `uploads/` folder with timestamp prefixes.

To clean up old uploads:
```bash
# Windows
del uploads\*

# Or delete manually from the uploads folder
```

## API Testing

Test the `/health` endpoint to verify models are loaded:
```
http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "models": {
    "mri_model": "loaded",
    "ct_model": "loaded"
  },
  "timestamp": "2024-02-11T..."
}
```

## Next Steps

1. Customize the frontend design if needed
2. Add authentication (see `auth.html` template)
3. Connect to a database to store results
4. Deploy to cloud (Heroku, AWS, Google Cloud, etc.)

## Development vs Production

### Development (Current Setup)
- `debug=True` for hot reload
- Development server
- Basic error messages

### Production
Change these in app.py before deploying:
```python
app.run(debug=False)  # Set to False
# Use production server like gunicorn
```

## Need Help?

1. Check console output for error messages
2. Visit `/health` endpoint for model status
3. Verify all files and folders exist
4. Ensure Python version is 3.8 or higher

## System Requirements

- **Minimum RAM**: 4GB
- **Disk Space**: 2GB (for models)
- **Python**: 3.8+
- **Internet**: Not required after initial setup

Enjoy using Brain Tumor Detection AI!
