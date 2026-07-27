import os
import tensorflow as tf
from tensorflow.keras.layers import Dense

class CustomDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

try:
    model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={'Dense': CustomDense})
    
    print("\n" + "="*50)
    print("SUCCESS: Model loaded properly.")
    print("Model expects input shape:", model.input_shape)
    print("="*50 + "\n")
    
except Exception as e:
    print(f"Error inspecting model: {e}")
