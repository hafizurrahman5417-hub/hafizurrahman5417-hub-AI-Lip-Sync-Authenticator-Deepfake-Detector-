import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_new_model():
    print("🚀 Starting XGBoost Training Pipeline...")
    
    # 1. LOAD YOUR DATASET
    print("Loading datasets...")
    import pandas as pd
    df = pd.read_csv("dataset/final_training_data.csv")
    
    # X is all columns except the last one, y is the 'label' column
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # 2. SPLIT THE DATA
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. INITIALIZE XGBOOST
    model = xgb.XGBClassifier(
        n_estimators=200,        # Number of trees
        learning_rate=0.05,      # Step size shrinkage to prevent overfitting
        max_depth=6,             # Maximum depth of a tree
        subsample=0.8,           # Fraction of samples used per tree
        colsample_bytree=0.8,    # Fraction of features used per tree
        objective='binary:logistic', 
        random_state=42,
        early_stopping_rounds=20 # Stops if accuracy doesn't improve for 20 rounds
    )
    
    # 4. TRAIN THE MODEL
    print("Training XGBoost Classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # 5. EVALUATE ACCURACY
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n✅ Training Complete! Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Deepfake", "Authentic"]))
    
    # 6. SAVE THE MODEL
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"\n💾 Model successfully saved to: {model_path}")

if __name__ == "__main__":
    train_new_model()