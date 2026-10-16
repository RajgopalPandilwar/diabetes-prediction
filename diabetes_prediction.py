# -*- coding: utf-8 -*-
"""
Diabetes Prediction System
===========================
A machine learning-based system for predicting diabetes using patient health metrics.
Uses multiple classifiers and compares their performance.

Author: Rajgopal Pandilwar
Based on original work by Vatshayan.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path='Training.csv', test_path='Testing.csv'):
    """
    Load training and testing datasets from CSV files.
    
    Args:
        train_path (str): Path to the training CSV file
        test_path (str): Path to the testing CSV file
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Features and labels split for ML
    """
    # Load the training data
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    print(f"Training data shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    # Load the testing data
    print("\nLoading testing data...")
    test_df = pd.read_csv(test_path)
    print(f"Testing data shape: {test_df.shape}")
    
    # Separate features and target variable
    # The 'Outcome' column is the target (0 = No diabetes, 1 = Diabetes)
    X_train = train_df.drop('Outcome', axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    """
    Scale features using StandardScaler for better model performance.
    
    Args:
        X_train: Training features
        X_test: Testing features
    
    Returns:
        tuple: Scaled (X_train_scaled, X_test_scaled)
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit on training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data preprocessing complete. Features scaled using StandardScaler.")
    return X_train_scaled, X_test_scaled


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models and evaluate their performance.
    
    Args:
        X_train: Scaled training features
        X_test: Scaled testing features
        y_train: Training labels
        y_test: Testing labels
    
    Returns:
        dict: Dictionary with model names as keys and accuracy scores as values
    """
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42),
    }
    
    results = {}
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc*100:.2f}%")
    
    return results


def predict_diabetes(model, scaler, patient_data):
    """
    Predict diabetes for a new patient.
    
    Args:
        model: Trained ML model
        scaler: Fitted StandardScaler
        patient_data (dict): Dictionary with patient health metrics
    
    Returns:
        int: Prediction (0 = No diabetes, 1 = Diabetes)
    """
    # Convert patient data to DataFrame
    patient_df = pd.DataFrame([patient_data])
    
    # Scale the data
    patient_scaled = scaler.transform(patient_df)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0] if hasattr(model, 'predict_proba') else None
    
    return prediction, probability


def main():
    """Main execution function."""
    print("=" * 60)
    print("DIABETES PREDICTION SYSTEM")
    print("Author: Rajgopal Pandilwar")
    print("=" * 60)
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Step 2: Preprocess data
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    # Step 3: Train and evaluate models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 4: Use the best model for a sample prediction
    best_model_name = max(results, key=results.get)
    print(f"\nBest performing model: {best_model_name} ({results[best_model_name]*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Prediction system ready for new patient data.")
    print("=" * 60)


if __name__ == '__main__':
    main()
