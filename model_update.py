import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths to the existing model and vectorizer
model_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/models/best_model.pkl'
vectorizer_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/models/tfidf_vectorizer.pkl'
data_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/data/raw_data.csv'

# Load the existing model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Load the existing data
data = pd.read_csv(data_path)
data = data.where((pd.notnull(data)), '')

# Label encoding
data['Category'] = data['Category'].map({'legitimate': 0, 'fraudulent': 1})

# Function to update the model with new data
def update_model(new_messages, new_labels):
    global data

    # Convert new data into a DataFrame
    new_data = pd.DataFrame({'Message': new_messages, 'Category': new_labels})
    
    # Update the existing data with new data
    data = pd.concat([data, new_data], ignore_index=True)

    # Split the updated data
    X = data['Message']
    Y = data['Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create a pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', LogisticRegression())
    ])

    # Hyperparameter tuning (optional but recommended)
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluation
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    print('Accuracy on training data:', accuracy_score(Y_train, train_preds))
    print('Accuracy on test data:', accuracy_score(Y_test, test_preds))
    print('Classification report on test data:\n', classification_report(Y_test, test_preds))

    # Save the updated model and vectorizer
    joblib.dump(best_model, model_path)
    joblib.dump(best_model.named_steps['tfidf'], vectorizer_path)

# Example usage
new_messages = ['new fraudulent message example', 'another new legitimate message']
new_labels = [1, 0]  # 1 for fraudulent, 0 for legitimate

update_model(new_messages, new_labels)
