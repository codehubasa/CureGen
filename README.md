# 🩺 CureGen AI

An AI-powered multi-disease diagnosis platform that leverages Deep Learning, Machine Learning, and Computer Vision to assist in the early detection of critical health conditions. Built with a Flask REST API backend and an interactive web interface, CureGen AI provides real-time medical predictions, confidence scores, and automated diagnostic reports for healthcare assistance.

🚧 **Project Status:** Ongoing – New AI diagnostic modules and advanced healthcare features are actively being developed.

---

## 🚀 Features

- 🧠 Brain Tumor Detection using MRI and CT scan images
- 👁️ Eye Cataract Detection using Deep Learning (CNN)
- ❤️ Heart Disease Risk Prediction using Machine Learning
- 📊 Real-time prediction with confidence score
- 📄 Automated PDF medical report generation
- 📤 Secure image upload and preprocessing
- 🔗 RESTful API architecture with Flask
- ⚡ Fast AI inference with optimized prediction pipeline
- 📱 Responsive and user-friendly web interface
- 🔒 Input validation and backend error handling
- 🩺 Multi-disease diagnosis platform under one application
- 🚀 Modular architecture for adding future disease detection models

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Fetch API

### Backend
- Python
- Flask
- Flask-CORS
- REST API

### Artificial Intelligence & Machine Learning
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pillow

### Models
- Brain Tumor Detection (MRI & CT)
- Eye Cataract Detection (CNN)
- Heart Disease Risk Prediction (Machine Learning)

---

## 📸 Project Preview


<img width="1472" height="924" alt="image" src="https://github.com/user-attachments/assets/3c9fec38-9681-40b9-8443-79e3a9ae32b8" />
------
<img width="1435" height="922" alt="image" src="https://github.com/user-attachments/assets/d1d9a752-3ae0-4916-982d-feb2b9814a57" />
------
<img width="1436" height="921" alt="image" src="https://github.com/user-attachments/assets/0cf2bdc0-1683-45a3-8ee7-35737c6cf988" />
------
<img width="1431" height="917" alt="image" src="https://github.com/user-attachments/assets/132abb99-fdc5-4ff4-bcab-78192a166bbe" />

## 🏗️ System Architecture

```
Frontend (HTML/CSS/JavaScript)
            │
            ▼
      Flask REST API
            │
            ▼
────────────────────────────────────
│ Brain Tumor Detection Model      │
│ Eye Cataract Detection Model     │
│ Heart Disease Prediction Model   │
────────────────────────────────────
            │
            ▼
     AI Prediction Results
            │
            ▼
     PDF Report Generation
```

---

## ⚙️ How It Works

### 🧠 Brain Tumor Detection
1. Upload an MRI or CT scan image.
2. The backend preprocesses the image.
3. TensorFlow model performs inference.
4. The system returns:
   - Tumor Detected / No Tumor Detected
   - Confidence Score
   - Scan Type

### 👁️ Eye Cataract Detection
1. Upload an eye image.
2. Image is preprocessed automatically.
3. CNN model predicts cataract presence.
4. Result and confidence score are displayed instantly.

### ❤️ Heart Disease Prediction
1. Enter patient information:
   - Age
   - Blood Pressure
   - Cholesterol
   - Heart Rate
2. Machine Learning model predicts risk level.
3. System displays:
   - High / Low Risk
   - Risk Percentage
   - Personalized prediction result

---

## 🌟 Key Highlights

- AI-powered healthcare assistant
- Multi-model medical diagnosis
- Deep Learning & Machine Learning integration
- RESTful Flask backend
- Real-time predictions
- Confidence score visualization
- Automated medical reports
- Modular and scalable architecture
- Clean and responsive user interface

---

## 📂 Project Structure

```
CureGen-AI
│
├── Backend
│   ├── app.py
│   ├── requirements.txt
│   ├── brain/
│   ├── eye/
│   └── heart/
│
├── Frontend
│   ├── index.html
│   ├── api-client.js
│   ├── brain_tumor_detection.html
│   ├── eye-cataract-detection.html
│   ├── heart-attack.html
│   └── ...
│
└── README.md
```

---

## 📦 Installation

**1. Clone Repository**
```bash
git clone https://github.com/Ankana-Sadhukhan/CureGen-AI.git
```

**2. Navigate to Project**
```bash
cd CureGen-AI
```

**3. Install Backend Dependencies**
```bash
cd Backend
pip install -r requirements.txt
```

**4. Start Flask Server**
```bash
python app.py
```

The backend will run on:
```
http://127.0.0.1:5000
```

**5. Open the Frontend**

Open any HTML file inside the `Frontend` folder using your browser.

Example:
- `Frontend/index.html`
- `Frontend/brain_tumor_detection.html`
- `Frontend/eye-cataract-detection.html`
- `Frontend/heart-attack.html`

---

## 🔮 Future Enhancements

- 🤖 Explainable AI (Grad-CAM, SHAP, LIME)
- ☁️ Cloud Deployment
- 👤 Secure User Authentication
- 🏥 Hospital Dashboard
- 📊 Patient History Management
- 🧬 Additional Disease Detection Models
- 📱 Mobile Application
- 🌐 Multi-language Support
- 🔔 Smart Health Notifications

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ Star on GitHub. It helps support the project and motivates future development!

---

## 👩‍💻 Author

**AI + Full Stack Developer**

- LinkedIn: [Asmita Banerjee](https://www.linkedin.com/in/asmita-banerjee-874461364/)
- Email: banerjeeasmita19@gmail.com
