import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

def run_cnn_pipeline():
    print("Loading Spectral Data (VNA Simulator)...")
    
    # 1. Load the dataset
    df = pd.read_csv('milk quality.csv')
    
    # 2. Extract the 'Signal' (The SPC columns act like our VNA Frequencies)
    # We find all columns that start with 'SPC'
    spectral_columns = [col for col in df.columns if 'SPC' in col]
    
    # Let's predict if the milk is Raw or not (Binary Classification)
    # Dropping rows with missing values in our target or features
    df_clean = df.dropna(subset=spectral_columns + ['IsRawMilk'])
    
    X = df_clean[spectral_columns].values
    y = df_clean['IsRawMilk'].map({'yes': 1, 'no': 0}).astype(int).values
    
    print(f"Extracted {len(spectral_columns)} frequency data points per scan.")
    print(f"Total samples: {len(X)}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scale the Data (Neural Networks require scaled data to learn properly)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. RESHAPE FOR CNN
    # Standard ML wants 2D data: (Samples, Features)
    # CNNs demand 3D data: (Samples, TimeSteps/Frequencies, Channels)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # 6. Build the 1D Convolutional Neural Network
    print("\nBuilding the 1D-CNN Architecture...")
    model = Sequential([
        # The 'Eye' of the AI: Slides across the frequency curve looking for patterns
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2), # Compresses the data, keeping only the strongest signals
        
        # A second, deeper look
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Flatten the curve into a standard array
        Flatten(),
        
        # Dense decision layers
        Dense(64, activation='relu'),
        Dropout(0.5), # Prevents the AI from memorizing the data (Overfitting)
        Dense(1, activation='sigmoid') # Final output: 0 (Pure) or 1 (Raw/Adulterated)
    ])
    
    # Compile the brain
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 7. Train the Model
    print("\nTraining the CNN (Epochs)...")
    history = model.fit(
        X_train_cnn, y_train, 
        epochs=20,           # How many times it loops through the whole dataset
        batch_size=32,       # How many scans it looks at simultaneously
        validation_split=0.2, # Watches for overfitting during training
        verbose=1
    )
    
    # 8. Evaluate on the hidden test set
    loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"\n--- Final Test Accuracy: {accuracy * 100:.2f}% ---")
    
    # 9. Plot the Learning Curve
    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Learning Curve')
    plt.xlabel('Epoch (Training Cycles)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn_learning_curve.png')
    print("Saved learning curve graph to cnn_learning_curve.png")
    
    # 10. Save the Deep Learning Model
    model.save('rfid_vna_cnn_model.keras')
    print("CNN Model saved as rfid_vna_cnn_model.keras")

if __name__ == "__main__":
    run_cnn_pipeline()