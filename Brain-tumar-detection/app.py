from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime
import sqlite3
from tensorflow.keras.layers import Dense

app = Flask(__name__, template_folder='.')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'brain-tumor-detection-key-change-in-production'

IMG_SIZE = (128, 128)
DATABASE = 'users.db'

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== DATABASE SETUP ==================
def init_db():
    """Initialize database with users table"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            scan_type TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ================== DATABASE HELPERS ==================
def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def user_exists(email):
    """Check if user exists by email"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE email = ?', (email,))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_user(email):
    """Get user by email"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(name, email, password):
    """Create new user"""
    try:
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
            (name, email, hashed_password)
        )
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None

def verify_password(user_email, password):
    """Verify user password"""
    user = get_user(user_email)
    if user:
        return check_password_hash(user['password'], password)
    return False

def save_prediction(user_id, result, confidence, scan_type, image_path):
    """Save prediction to database"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        '''INSERT INTO predictions 
           (user_id, result, confidence, scan_type, image_path) 
           VALUES (?, ?, ?, ?, ?)''',
        (user_id, result, confidence, scan_type, image_path)
    )
    conn.commit()
    conn.close()

# ================== LOAD MODELS ==================
class CustomDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

try:
    model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model.h5")
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'Dense': CustomDense})
    
    # Use the same model for both MRI and CT if only one is provided
    mri_model = loaded_model
    ct_model = loaded_model
    print("✓ Model brain_tumor_model.h5 loaded successfully for both MRI and CT")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    mri_model = None
    ct_model = None

classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']


# ================== HELPER FUNCTIONS ==================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Load and preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")


# ================== AUTHENTICATION ROUTES ==================
@app.route('/signup', methods=['POST'])
def signup():
    """User registration endpoint"""
    try:
        data = request.json
        
        # Validation
        if not data.get('name') or not data.get('email') or not data.get('password'):
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        name = data.get('name').strip()
        email = data.get('email').strip().lower()
        password = data.get('password')
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
        
        if user_exists(email):
            return jsonify({'success': False, 'error': 'Email already registered'}), 400
        
        # Create user
        user_id = create_user(name, email, password)
        
        if user_id:
            session['user_id'] = user_id
            session['user_name'] = name
            session['user_email'] = email
            return jsonify({
                'success': True, 
                'message': 'Account created successfully',
                'user_id': user_id,
                'user_name': name
            }), 201
        else:
            return jsonify({'success': False, 'error': 'Error creating account'}), 500
            
    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.json
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'success': False, 'error': 'Email and password are required'}), 400
        
        email = data.get('email').strip().lower()
        password = data.get('password')
        
        user = get_user(email)
        
        if not user or not verify_password(email, password):
            return jsonify({'success': False, 'error': 'Invalid email or password'}), 401
        
        # Set session
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user_id': user['id'],
            'user_name': user['name']
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200


@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Check if user is logged in"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user_id': session['user_id'],
            'user_name': session['user_name'],
            'user_email': session['user_email']
        }), 200
    return jsonify({'authenticated': False}), 200


# ================== APPLICATION ROUTES ================
@app.route('/')
def home():
    """Home page route"""
    return render_template("index.html")


@app.route('/auth')
def auth_page():
    """Authentication page route"""
    return render_template("auth.html")


@app.route('/upload-page')
def upload_page():
    """Upload/detection page route"""
    if 'user_id' not in session:
        return redirect(url_for('auth_page'))
    return render_template("detection.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check authentication
        if 'user_id' not in session:
            return jsonify({'error': 'User not authenticated', 'success': False}), 401
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file has a filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
        
        # Get scan type from form
        scan_type = request.form.get('scan_type', 'ct').lower()
        if scan_type not in ['mri', 'ct']:
            return jsonify({'error': 'Invalid scan type. Choose: mri or ct'}), 400
        
        # Select model based on scan type
        if scan_type == 'mri':
            if mri_model is None:
                return jsonify({'error': 'MRI model not available'}), 500
            model = mri_model
        else:
            if ct_model is None:
                return jsonify({'error': 'CT model not available'}), 500
            model = ct_model
        
        # Save file securely
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img = preprocess_image(filepath)
        
        # Make prediction
        prediction = model.predict(img, verbose=0)
        pred_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        num_classes = prediction.shape[1]
        if num_classes == 2:
            current_classes = ['No Tumor', 'Tumor Detected']
        elif num_classes == 4:
            current_classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
        else:
            current_classes = [f'Class {i}' for i in range(num_classes)]
            
        result = current_classes[pred_index]
        confidence_percent = round(confidence * 100, 2)
        
        # Save prediction to database
        save_prediction(session['user_id'], result, confidence_percent, scan_type, f'/uploads/{filename}')
        
        # Prepare response
        response_data = {
            'success': True,
            'result': result,
            'confidence': confidence_percent,
            'image_path': f'/uploads/{filename}',
            'scan_type': scan_type,
            'timestamp': datetime.now().isoformat(),
            'class_probabilities': {
                current_classes[i]: round(float(prediction[0][i]) * 100, 2) 
                for i in range(num_classes)
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/result', methods=['GET'])
def result():
    """Result page route"""
    if 'user_id' not in session:
        return redirect(url_for('auth_page'))
    return render_template("result.html")


@app.route('/user-history', methods=['GET'])
def user_history():
    """Get user prediction history"""
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'User not authenticated'}), 401
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            SELECT id, result, confidence, scan_type, image_path, created_at 
            FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 50
        ''', (session['user_id'],))
        predictions = c.fetchall()
        conn.close()
        
        return jsonify({
            'success': True,
            'predictions': [dict(p) for p in predictions]
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models_status = {
        'mri_model': 'loaded' if mri_model is not None else 'not_loaded',
        'ct_model': 'loaded' if ct_model is not None else 'not_loaded'
    }
    return jsonify({
        'status': 'healthy',
        'models': models_status,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.errorhandler(413)
def handle_large_file(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Brain Tumor Detection API")
    print("=" * 60)
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Database: {os.path.abspath(DATABASE)}")
    print("Models loaded:")
    print(f"  - MRI model: {'✓' if mri_model else '✗'}")
    print(f"  - CT model: {'✓' if ct_model else '✗'}")
    print("=" * 60)
    app.run(debug=True, port=5000)
