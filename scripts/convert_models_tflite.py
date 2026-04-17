import tensorflow as tf

def convert_to_tflite(model_path, tflite_path, optimize=True):
    print(f"Converting {model_path} to TFLite...")
    try:
        # Load the Keras model
        model = tf.keras.models.load_model(model_path)
        
        # Initialize the converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if optimize:
            # Apply dynamic range quantization to reduce size
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("  - Optimization (Quantization) enabled.")
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Saved: {tflite_path}")
    except Exception as e:
        print(f"Error converting {model_path}: {e}")

if __name__ == "__main__":
    # Convert Urea Detector (TinyML)
    convert_to_tflite('urea_detector_model.keras', 'urea_detector.tflite')
    
    # Convert Spectral Raw Milk Detector
    convert_to_tflite('rfid_vna_cnn_model.keras', 'raw_milk_detector.tflite')
    
    # Convert High-Fidelity Urea Detector (Production Target)
    convert_to_tflite('hf_urea_detector_model.keras', 'hf_urea_detector.tflite')
