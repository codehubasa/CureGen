import unittest
import os
import tempfile
import json
import io
from PIL import Image

# Import flask app
from app import app, init_db, get_db_connection

class CureGenTestCase(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        # Use a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        app.config['DATABASE'] = self.db_path
        app.config['SECRET_KEY'] = 'test-key'
        
        # Override the database globally in the app module
        import app as app_module
        app_module.DATABASE = self.db_path
        
        # Re-initialize the temporary database
        init_db()
        
        self.client = app.test_client()
        self.ctx = app.app_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_auth_flow(self):
        # Test Check Auth Unauthenticated
        response = self.client.get('/api/check-auth')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(data['authenticated'])

        # Test Signup
        response = self.client.post('/api/signup', json={
            'name': 'Test Doctor',
            'email': 'doctor@test.com',
            'password': 'password123'
        })
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(data['success'])
        self.assertEqual(data['user_name'], 'Test Doctor')

        # Test Check Auth Authenticated
        response = self.client.get('/api/check-auth')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['authenticated'])
        self.assertEqual(data['user_name'], 'Test Doctor')

        # Test Logout
        response = self.client.post('/api/logout')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

        # Test Login
        response = self.client.post('/api/login', json={
            'email': 'doctor@test.com',
            'password': 'password123'
        })
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])

    def test_hospitals_and_chatbot(self):
        # Test Chatbot API
        response = self.client.post('/api/chatbot', json={
            'message': 'Tell me about heart attacks.'
        })
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn("heart", data['reply'].lower())

        # Test Hospital Search
        response = self.client.get('/api/hospitals?city=mumbai')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['city'], 'Mumbai')
        self.assertGreater(len(data['hospitals']), 0)

    def test_clinical_vital_predictions(self):
        # Register and login to test prediction endpoints
        self.client.post('/api/signup', json={
            'name': 'Test Patient',
            'email': 'patient@test.com',
            'password': 'password123'
        })
        
        # Test BP Tracker
        response = self.client.post('/api/predict/bp', data={
            'systolic': '135',
            'diastolic': '85'
        })
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['result'], 'Stage 1 Hypertension')

        # Test Diabetes Risk (Sugar Level)
        response = self.client.post('/api/predict/diabetes', data={
            'glucose': '160',
            'blood_pressure': '80',
            'insulin': '100',
            'bmi': '32.5',
            'age': '48'
        })
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn("Diabetes Risk", data['result'])

    def test_image_diagnostics_and_pdf_generation(self):
        # Register and login
        self.client.post('/api/signup', json={
            'name': 'Xray Patient',
            'email': 'xray@test.com',
            'password': 'password123'
        })

        # Create a mock grayscale image in memory for chest x-ray
        img_io = io.BytesIO()
        img = Image.new('L', (200, 200), color=128)
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        
        # Test Lung Disease Prediction
        response = self.client.post('/api/predict/lungs', data={
            'image': (img_io, 'xray.jpg')
        }, content_type='multipart/form-data')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        prediction_id = data['prediction_id']

        # Test Fetching History
        response = self.client.get('/api/history')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(len(data['predictions']), 1)
        self.assertEqual(data['predictions'][0]['disease'], 'Lung Disease')

        # Test Generating PDF Report
        pdf_response = self.client.get(f'/api/report/{prediction_id}')
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.mimetype, 'application/pdf')
        self.assertIn(b'%PDF', pdf_response.data[:4])

if __name__ == '__main__':
    unittest.main()
