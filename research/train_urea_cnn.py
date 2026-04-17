import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_urea_detector():
    print("Starting TinyML 1D-CNN Training for Urea Detection...")
    
    # 1. Load the synthetic VNA dataset
    try:
        df = pd.read_csv('vna_adulteration_dataset.csv')
    except FileNotFoundError:
        print("Error: vna_adulteration_dataset.csv not found!")
        return
    
    # 2. Extract Features and Labels
    # S11 scans start from column index 3 (Sample_ID, Urea_Pct, Class_Label are first)
    X = df.iloc[:, 3:].values
    y = df['Class_Label'].values
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} frequency points per scan.")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Reshape for CNN (Samples, TimeSteps, Channels)
    # 1D-CNN expects 3D input
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # 5. Build TinyML-Optimized 1D-CNN
    # Keeping architecture extremely lightweight for microcontroller deployment
    model = Sequential([
        Input(shape=(X_train_cnn.shape[1], 1)),
        
        # Level 1: Feature Extraction (Resonance Peak Detection)
        Conv1D(filters=8, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Level 2: Feature Extraction (Dampening/Width Detection)
        Conv1D(filters=16, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Level 3: Decision Head
        Flatten(),
        Dense(16, activation='relu'),
        Dropout(0.2), # Prevent overfitting on synthetic noise
        Dense(1, activation='sigmoid') # Binary Output: Adulterated (1) or Pure (0)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 6. Training Phase
    print("Training the TinyML Model...")
    history = model.fit(
        X_train_cnn, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )
    
    # 7. Comprehensive Evaluation
    print("\nEvaluating Model on Test Set...")
    loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    
    # 8. Visual Analytics (High Aesthetics)
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Accuracy/Loss Curves
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='teal', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', lw=2)
    ax1.set_title('Learning Curves (Accuracy)', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot Confusion Matrix
    y_pred = (model.predict(X_test_cnn) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Pure', 'Adulterated'])
    disp.plot(cmap='Blues', ax=ax2)
    ax2.set_title('TinyML Detection Matrix', fontsize=14)
    ax2.grid(False)
    
    plt.tight_layout()
    plt.savefig('urea_cnn_performance.png', dpi=300)
    print("Saved performance metrics to urea_cnn_performance.png")
    
    # 9. Save Model
    model.save('urea_detector_model.keras')
    print("Model saved as urea_detector_model.keras")

if __name__ == "__main__":
    train_urea_detector()
