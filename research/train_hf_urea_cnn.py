import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_hf_urea_detector():
    print("Starting High-Fidelity 1D-CNN Training for Urea detection...")
    
    # 1. Load the High-Fidelity Physics Dataset
    try:
        df = pd.read_csv('vna_physics_dataset.csv')
    except FileNotFoundError:
        print("Error: vna_physics_dataset.csv not found!")
        return
    
    # 2. Preprocessing
    # Features start from column 3 (Sample_ID, Urea_Percentage, Label are first)
    X = df.iloc[:, 3:].values
    y = df['Label'].values
    
    # Final check on input dimensions
    num_samples, num_freq_points = X.shape
    print(f"Dataset Loaded: {num_samples} samples, {num_freq_points} frequency points.")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Reshape for Conv1D (Samples, TimeSteps, Channels)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 5. Build Advanced 1D-CNN for Physics Features
    # This model is designed to detect both global shifts (f0) and local slopes (gamma).
    model = Sequential([
        Input(shape=(X_train_cnn.shape[1], 1)),
        
        # Layer 1: Detect Sharp Resonance Peaks
        Conv1D(filters=16, kernel_size=7, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Layer 2: Detect Line Broandening / Slopes
        Conv1D(filters=32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Layer 3: Dynamic Context
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        Flatten(),
        
        # Prediction Head
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 6. Training Phase
    print("\nTraining the High-Fidelity Model...")
    history = model.fit(
        X_train_cnn, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # 7. Comprehensive Evaluation
    print("\nEvaluating High-Fidelity Model Performance...")
    loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    
    # 8. Visual Analytics (Engineering Standard)
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy Plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#3498db', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#e67e22', lw=2)
    ax1.set_title('CNN Learning Convergence', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Confusion Matrix
    y_pred = (model.predict(X_test_cnn) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pure', 'Urea-Adulterated'])
    disp.plot(cmap='YlGnBu', ax=ax2)
    ax2.set_title('Detection Precision Matrix', fontsize=14)
    ax2.grid(False)
    
    plt.tight_layout()
    plt.savefig('hf_urea_cnn_results.png', dpi=300)
    print("Performance visualization saved to hf_urea_cnn_results.png")
    
    # 9. Model Export
    model.save('hf_urea_detector_model.keras')
    print("Model saved as hf_urea_detector_model.keras")

if __name__ == "__main__":
    train_hf_urea_detector()
