import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

def run_composition_pipeline():
    print("Loading Dataset for Chemical Composition Prediction...")
    
    # 1. Load data
    df = pd.read_csv('milk quality.csv')
    
    # Define Targets and Features
    target_cols = ['Fat', 'Protein', 'Lactose', 'Solids', 'FFA', 'Citrate', 'FrzPoint', 'SNF', 'MUN', 'Casein']
    spectral_cols = [col for col in df.columns if 'SPC' in col]
    
    # Drop rows with missing values in targets or spectral features
    df_clean = df.dropna(subset=target_cols + spectral_cols)
    
    X = df_clean[spectral_cols].values
    y = df_clean[target_cols].values
    
    print(f"Features: {X.shape[1]} spectral bands")
    print(f"Samples: {len(X)}")
    print(f"Targets: {len(target_cols)} (Multi-output Regression)")
    
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler() # Scaling targets is essential for multi-output regression stability
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # 4. Reshape for CNN (Samples, Features, Channels)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # 5. Build Multi-Output CNN Regressor
    print("\nDesigning Multi-Output 1D-CNN Regressor...")
    model = Sequential([
        Input(shape=(X_train_cnn.shape[1], 1)),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(len(target_cols), activation='linear') # Linear activation for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # 6. Train
    print("\nTraining the model (this may take a minute)...")
    history = model.fit(
        X_train_cnn, y_train_scaled,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 7. Evaluate and Inverse Transform Predictions
    y_pred_scaled = model.predict(X_test_cnn)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # 8. Visualizations (Aesthetic Comparison)
    print("\nGenerating Correlation Plots...")
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # We'll plot the top 4 most common components
    main_components = ['Fat', 'Protein', 'Lactose', 'SNF']
    for i, component in enumerate(main_components):
        idx = target_cols.index(component)
        axes[i].scatter(y_test[:, idx], y_pred[:, idx], alpha=0.5, color='teal')
        axes[i].plot([y_test[:, idx].min(), y_test[:, idx].max()], 
                    [y_test[:, idx].min(), y_test[:, idx].max()], 
                    'r--', lw=2)
        axes[i].set_title(f'{component}: Actual vs Predicted')
        axes[i].set_xlabel('Actual Value')
        axes[i].set_ylabel('Predicted Value')
        
    plt.tight_layout()
    plt.savefig('composition_regression_results.png')
    print("Saved visualization to composition_regression_results.png")
    
    # 9. Save all artifacts
    model.save('milk_composition_model.keras')
    joblib.dump(scaler_x, 'comp_scaler_x.joblib')
    joblib.dump(scaler_y, 'comp_scaler_y.joblib')
    print("Model and scalers saved successfully.")

if __name__ == "__main__":
    run_composition_pipeline()
