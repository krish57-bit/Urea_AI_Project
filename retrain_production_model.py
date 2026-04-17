import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
import matplotlib.pyplot as plt
import os
import argparse

"""
Production Domain Adaptation Retraining
---------------------------------------
Role: Lead ML Data Scientist (Co-Founder)
Objective: Hardening the 1D-CNN against real-world hardware noise using 
           a mixture of High-Fidelity Physics data and Real Hardware scans.
"""

def prepare_data(synthetic_path, production_path):
    print("Loading datasets for Domain Adaptation...")
    
    # 1. Load Synthetic Physics Data
    df_sync = pd.read_csv(synthetic_path)
    X_sync = df_sync.iloc[:, 3:].values
    y_sync = df_sync['Label'].values
    
    # 2. Load Real Production Data (with fallback)
    if os.path.exists(production_path) and os.path.getsize(production_path) > 100:
        df_prod = pd.read_csv(production_path)
        # Drop Timestamp and Bottle_ID columns (0 and 1)
        X_prod = df_prod.iloc[:, 4:].values # production_vna_dataset has Timestamp, BottleID, Pct, Label
        y_prod = df_prod['Label'].values
        print(f"  - Production Data: {len(X_prod)} samples found.")
    else:
        print("  - [WARNING] No real production data found. Using 'Hardened Synthetic' mode.")
        # Create "Hardened Synthetic" by adding more noise to the physics dataset
        X_prod = X_sync + np.random.normal(0, 0.4, X_sync.shape) 
        y_prod = y_sync
        
    return X_sync, y_sync, X_prod, y_prod

def augment_spectral_data(X):
    """
    Applies Spectral Jitter and Gain Shift to harden the model.
    """
    jitter = np.random.normal(0, 0.05, X.shape)
    gain_shift = np.random.uniform(0.98, 1.02, (X.shape[0], 1))
    return (X * gain_shift) + jitter

def retrain_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_csv", default="vna_physics_dataset.csv")
    parser.add_argument("--prod_csv", default="production_vna_dataset.csv")
    args = parser.parse_args()

    # 1. Get Data
    X_sync, y_sync, X_prod, y_prod = prepare_data(args.sync_csv, args.prod_csv)
    
    # 2. Merge and Shuffle (Mixed Domain)
    X_combined = np.vstack([X_sync, X_prod])
    y_combined = np.hstack([y_sync, y_prod])
    
    # 3. Final Augmentation
    X_hardened = augment_spectral_data(X_combined)
    X_hardened_cnn = X_hardened.reshape(X_hardened.shape[0], X_hardened.shape[1], 1)
    
    # 4. Load High-Fidelity Brain
    print("\nLoading High-Fidelity Base Model...")
    try:
        model = load_model('hf_urea_detector_model.keras')
    except:
        print("Error: hf_urea_detector_model.keras not found. Base model must be trained first.")
        return

    # 5. Fine-Tune on Mixed Domain
    # We use a very low learning rate to prevent "Catastrophic Forgetting" of physics
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Commencing Domain Adaptation Retraining...")
    history = model.fit(
        X_hardened_cnn, y_combined,
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # 6. Export Production Artifacts
    print("\nExporting Production Artifacts...")
    model.save('production_urea_detector.keras')
    
    # TFLite Conversion (Int8 Weights)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('production_urea_detector.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("FINAL PRODUCTION MODEL SAVED: production_urea_detector.tflite")
    
    # 7. Visualization
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Adaptation Accuracy', color='purple', lw=3)
    plt.title('Domain Adaptation: Physics -> Real Hardware Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('production_adaptation_metrics.png')
    print("Adaptation metrics saved to production_adaptation_metrics.png")

if __name__ == "__main__":
    retrain_pipeline()
