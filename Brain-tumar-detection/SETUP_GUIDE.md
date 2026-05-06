# Brain Tumor Detection AI - Setup & User Guide

## 📋 Table of Contents
1. [Installation](#installation)
2. [Features](#features)
3. [Running the Application](#running-the-application)
4. [User Authentication](#user-authentication)
5. [How to Use](#how-to-use)
6. [API Endpoints](#api-endpoints)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
cd "Brain tumar detection"
pip install -r requirements.txt
```

### Step 2: Verify Models
Ensure the following model files exist in the `models/` directory:
- `brain_tumor_mri_model.keras` - For MRI scan analysis
- `brain_tumor_model.keras` - For CT scan analysis

### Step 3: Create Database
The application automatically creates a SQLite database (`users.db`) on first run.

---

## ✨ Features

### User Authentication
- **Sign Up**: New users can create accounts with email and password
- **Sign In**: Registered users can log in securely
- **Session Management**: User sessions are maintained during the session
- **Password Security**: All passwords are hashed using Werkzeug security

### Tumor Detection
- **MRI Support**: Analyze Magnetic Resonance Imaging scans
- **CT Support**: Analyze Computed Tomography scans
- **Real-time Results**: Get predictions with confidence scores
- **Detailed Analysis**: View probability for all tumor types
- **PDF Reports**: Generate and download analysis reports

### Tumor Classes
- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## 🚀 Running the Application

### Start the Flask Server
```bash
cd "C:\Users\ankan\Desktop\Brain tumar detection"
python app.py
```

### Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

### Server Output
You should see:
```
============================================================
Brain Tumor Detection API
============================================================
✓ MRI model loaded successfully
✓ CT model loaded successfully
Running on http://127.0.0.1:5000
```

---

## 🔐 User Authentication

### Sign Up Process
1. Click **"Try Detection"** on the home page
2. Click **"Sign up"** link to switch to registration
3. Enter:
   - **Full Name**: Your name
   - **Email**: Valid email address (must be unique)
   - **Password**: At least 6 characters
4. Click **"Create Account"**
5. You'll be redirected to the upload page

### Sign In Process
1. On the authentication page, enter:
   - **Email**: Your registered email
   - **Password**: Your password
2. Click **"Sign In"**
3. You'll be redirected to the upload page

### User Data Storage
All user data is stored in the **`users.db`** SQLite database:

**Users Table:**
```
- id (Primary Key)
- name (Full Name)
- email (Unique)
- password (Hashed)
- created_at (Registration Date)
```

**Predictions Table:**
```
- id (Primary Key)
- user_id (Foreign Key to users)
- result (Tumor Type)
- confidence (Confidence Percentage)
- scan_type (MRI or CT)
- image_path (Uploaded Image Path)
- created_at (Analysis Date)
```

### Logout
Click the **"Logout"** button in the top-right corner to exit your session.

---

## 📖 How to Use

### Analyzing a Scan

1. **Sign In**: Log in to your account
2. **Select Scan Type**: Choose between:
   - 🧲 MRI Test (Magnetic Resonance Imaging)
   - 🎯 CT Scan (Computed Tomography)
3. **Upload Image**: 
   - Click the upload box
   - Select an image file (PNG, JPG, JPEG, GIF, BMP)
   - Maximum file size: 16MB
4. **Start Processing**: Click **"Start AI Processing"**
5. **View Results**: 
   - See confidence score
   - View probability for all tumor types
   - View the uploaded image
6. **Download Report**: Click **"Download Report (PDF)"** to save the analysis

---

## 🔌 API Endpoints

### Authentication Endpoints

#### Sign Up
```
POST /signup
Content-Type: application/json

{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "password123"
}

Response (201):
{
    "success": true,
    "message": "Account created successfully",
    "user_id": 1,
    "user_name": "John Doe"
}
```

#### Sign In
```
POST /login
Content-Type: application/json

{
    "email": "john@example.com",
    "password": "password123"
}

Response (200):
{
    "success": true,
    "message": "Login successful",
    "user_id": 1,
    "user_name": "John Doe"
}
```

#### Logout
```
POST /logout

Response (200):
{
    "success": true,
    "message": "Logged out successfully"
}
```

#### Check Authentication
```
GET /check-auth

Response (200):
{
    "authenticated": true,
    "user_id": 1,
    "user_name": "John Doe",
    "user_email": "john@example.com"
}
```

### Application Endpoints

#### Home Page
```
GET /
Returns: index.html
```

#### Authentication Page
```
GET /auth
Returns: auth.html
```

#### Upload/Detection Page
```
GET /upload-page
Returns: detection.html (requires authentication)
```

#### Make Prediction
```
POST /predict
Content-Type: multipart/form-data

Parameters:
- image: (file) - Brain scan image
- scan_type: (string) - "mri" or "ct"

Response (200):
{
    "success": true,
    "result": "no_tumor",
    "confidence": 95.23,
    "image_path": "/uploads/20260211_012115_scan.jpg",
    "scan_type": "mri",
    "timestamp": "2026-02-11T01:21:15.123456",
    "class_probabilities": {
        "glioma": 2.45,
        "meningioma": 1.23,
        "pituitary": 1.09,
        "no_tumor": 95.23
    }
}
```

#### Results Page
```
GET /result
Returns: result.html (requires authentication)
```

#### User Prediction History
```
GET /user-history
Returns: Last 50 predictions for the user (requires authentication)

Response (200):
{
    "success": true,
    "predictions": [
        {
            "id": 1,
            "result": "no_tumor",
            "confidence": 95.23,
            "scan_type": "mri",
            "image_path": "/uploads/...",
            "created_at": "2026-02-11 01:21:15"
        },
        ...
    ]
}
```

#### Health Check
```
GET /health

Response (200):
{
    "status": "healthy",
    "models": {
        "mri_model": "loaded",
        "ct_model": "loaded"
    },
    "timestamp": "2026-02-11T01:21:15.123456"
}
```

---

## 🔍 Troubleshooting

### Issue: "404 Endpoint not found"
**Solution**: Ensure you're using the correct route. Try `/auth` or `/upload-page`.

### Issue: Models not loading
**Solution**: 
1. Verify model files exist in `models/` directory
2. Check file names match exactly:
   - `brain_tumor_mri_model.keras`
   - `brain_tumor_model.keras`
3. Restart the Flask server

### Issue: Login fails with "Email already registered"
**Solution**: Use a different email or use "Sign In" if you already have an account.

### Issue: Password too short
**Solution**: Use a password with at least 6 characters.

### Issue: Port 5000 already in use
**Solution**: Edit `app.py` and change:
```python
app.run(debug=True, port=5001)  # Use different port
```

### Issue: "uploads" folder not found
**Solution**: The app creates this automatically. If error persists:
1. Create `uploads/` folder manually in the project directory
2. Run the application again

### Issue: Database errors
**Solution**:
1. Delete `users.db` file
2. Restart the application (it will recreate the database)

---

## 📁 Project Structure

```
Brain Tumor Detection/
├── app.py                          # Flask backend (authentication + predictions)
├── requirements.txt                # Python dependencies
├── config.py                       # Configuration
├── users.db                        # SQLite database (auto-created)
├── models/
│   ├── brain_tumor_mri_model.keras # MRI model
│   └── brain_tumor_model.keras     # CT model
├── templates/
│   ├── index.html                  # Home page
│   ├── auth.html                   # Login/Sign up page
│   ├── detection.html              # Upload page
│   ├── result.html                 # Results page
│   └── upload.html                 # Alternative upload page
├── uploads/                        # Uploaded images (auto-created)
└── README.md                       # Documentation
```

---

## 🔒 Security Notes

- **Passwords**: Stored as SHA256 hashes, never in plain text
- **Sessions**: Use Flask sessions with a secret key (change in production)
- **File Uploads**: Only image files allowed, maximum 16MB
- **SQL Injection**: Protected with parameterized queries

---

## 📝 License

This project is provided for educational and research purposes.

---

## ❓ Support

For issues:
1. Check the terminal/console output for error messages
2. Visit `/health` endpoint to verify models are loaded
3. Verify all files are in the correct directories
4. Ensure Python and all dependencies are installed correctly
