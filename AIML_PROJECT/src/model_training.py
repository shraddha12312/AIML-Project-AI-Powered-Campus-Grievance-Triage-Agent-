"""
CO4: Machine Learning Basics - Supervised Learning & Classification.
Trains a Naive Bayes model based on the dataset.
"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'complaints_dataset.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'department_classifier.pkl')

def train_and_save_model():
    print(f"Loading dataset from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Dataset not found! Please ensure data/complaints_dataset.csv exists.")
        return
        
    # X: Features (Raw Text), y: Labels (Department)
    X = df['text']
    y = df['department']
    
    print(f"Dataset Loaded. Total samples: {len(df)}")
    print("Training Supervised ML Classification Model (MultinomialNB)...")
    
    # We build a pipeline that first turns words into numbers (TF-IDF Feature Learning)
    # Then applies the Naive Bayes Classifier.
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # Train the model 
    # (Note: In a huge dataset, we'd split into train/test to check for Overfitting. But here we train on all for demonstration).
    pipeline.fit(X, y)
    
    # Save the pipeline object to disk using Pickle so our Triage Agent can use it later
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Model successfully trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
