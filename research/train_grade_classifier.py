import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

def train_grade_model():
    print("Starting Milk Grade Classification Pipeline...")
    
    # 1. Load the dataset
    try:
        df = pd.read_csv('milknew.csv')
    except FileNotFoundError:
        print("Error: milknew.csv not found!")
        return
    
    # 2. Preprocessing
    # Strip whitespace from column names (handles 'Fat ' column)
    df.columns = df.columns.str.strip()
    
    # Map Target Grade to Integers
    grade_map = {'low': 0, 'medium': 1, 'high': 2}
    df['Grade'] = df['Grade'].map(grade_map)
    
    # Drop rows with missing values
    df = df.dropna()
    
    X = df.drop('Grade', axis=1)
    y = df['Grade']
    
    print(f"Dataset loaded: {len(df)} samples, {len(X.columns)} features.")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Random Forest Classifier
    print("Training Random Forest Model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # 6. Evaluation
    y_pred = clf.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))
    
    # 7. Visualizations (High Aesthetics)
    plt.style.use('ggplot')
    
    # Plot 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test_scaled, y_test, 
        display_labels=['Low', 'Medium', 'High'],
        cmap='Blues', ax=ax
    )
    plt.title('Milk Grade Confusion Matrix', fontsize=15, pad=20)
    plt.tight_layout()
    plt.savefig('grade_confusion_matrix.png', dpi=300)
    print("Saved confusion matrix to grade_confusion_matrix.png")
    
    # Plot 2: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    plt.bar(range(X.shape[1]), importances[indices], color='skyblue', edgecolor='navy')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.title('Critical Factors for Milk Grade (Feature Importance)', fontsize=15, pad=20)
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('grade_feature_importance.png', dpi=300)
    print("Saved feature importance plot to grade_feature_importance.png")
    
    # 8. Save the Model
    joblib.dump(clf, 'milk_grade_model.joblib')
    joblib.dump(scaler, 'grade_scaler.joblib')
    print("Model and Scaler saved successfully.")

if __name__ == "__main__":
    train_grade_model()
