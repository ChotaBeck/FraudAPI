# import os
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import cross_val_score
# import joblib
# import time
# import nltk
# from nltk.tokenize import word_tokenize

# # Specify the path to your venv's nltk_data folder
# nltk_data_path = os.path.join(os.path.dirname(__file__), 'venv','lib', 'nltk_data')

# # Create the nltk_data directory if it doesn't exist
# if not os.path.exists(nltk_data_path):
#     os.makedirs(nltk_data_path)

# # Add the nltk_data path to the nltk data paths
# nltk.data.path.append(nltk_data_path)

# # Download required NLTK resources
# nltk.download('punkt', download_dir=nltk_data_path)
# nltk.download('punkt_tab', download_dir=nltk_data_path)

# # Tokenization function
# def tokenize_text(text):
#     tokens = word_tokenize(text)
#     return [word for word in tokens if word.isalpha()]

# # Load and preprocess data
# df = pd.read_csv('/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/data/raw_data.csv')
# df['Category'] = df['Category'].map({'legitimate': 0, 'fraudulent': 1})

# X = df['Message']
# Y = df['Category']

# # Split the data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Feature extraction
# vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, tokenizer=tokenize_text)
# X_train_features = vectorizer.fit_transform(X_train)
# X_test_features = vectorizer.transform(X_test)

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(random_state=42),
#     "Naive Bayes": MultinomialNB(),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "SVM": SVC(kernel='rbf', random_state=42, probability=True)
# }

# # Function to evaluate model
# def evaluate_model(model, X_train, X_test, y_train, y_test):
#     start_time = time.time()
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
    
#     # Perform cross-validation
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
#     end_time = time.time()
#     training_time = end_time - start_time
    
#     return accuracy, precision, recall, f1, cv_scores.mean(), training_time

# # Evaluate all models
# results = []

# for name, model in models.items():
#     accuracy, precision, recall, f1, cv_score, training_time = evaluate_model(model, X_train_features, X_test_features, Y_train, Y_test)
#     results.append({
#         "Model": name,
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1,
#         "Cross-Validation Score": cv_score,
#         "Training Time": training_time
#     })

# # Convert results to DataFrame and sort by accuracy
# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values("Accuracy", ascending=False)

# print(results_df)

# # Save the best model
# best_model_name = results_df.iloc[0]["Model"]
# best_model = models[best_model_name]

# folder_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/models/'
# model_file_path = f"{folder_path}best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
# joblib.dump(best_model, model_file_path)

# vectorizer_file_path = f"{folder_path}best_vectorizer.pkl"
# joblib.dump(vectorizer, vectorizer_file_path)

# print(f"\nBest model ({best_model_name}) saved as: {model_file_path}")
# print(f"Vectorizer saved as: {vectorizer_file_path}")
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import time
import nltk
from nltk.tokenize import word_tokenize

# Specify the path to your venv's nltk_data folder
nltk_data_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib', 'nltk_data')

# Create the nltk_data directory if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add the nltk_data path to the nltk data paths
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)

# Tokenization function
def tokenize_text(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word.isalpha()]

# Load and preprocess data
df = pd.read_csv('/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/data/raw_data.csv')
df['Category'] = df['Category'].map({'legitimate': 0, 'fraudulent': 1})

X = df['Message']
Y = df['Category']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, tokenizer=tokenize_text)
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42, probability=True)
}

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return accuracy, precision, recall, f1, cv_scores.mean(), training_time

# Evaluate all models
results = []

for name, model in models.items():
    accuracy, precision, recall, f1, cv_score, training_time = evaluate_model(model, X_train_features, X_test_features, Y_train, Y_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Cross-Validation Score": cv_score,
        "Training Time": training_time
    })

# Convert results to DataFrame and sort by accuracy
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Accuracy", ascending=False)

print(results_df)

# Save the best model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

# Create folder if it doesn't exist
folder_path = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

model_file_path = os.path.join(folder_path, f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
vectorizer_file_path = os.path.join(folder_path, "best_vectorizer.pkl")

# Save model
try:
    joblib.dump(best_model, model_file_path)
    print(f"Best model ({best_model_name}) saved as: {model_file_path}")
except Exception as e:
    print(f"Error saving model: {str(e)}")

# Save vectorizer
try:
    joblib.dump(vectorizer, vectorizer_file_path)
    print(f"Vectorizer saved as: {vectorizer_file_path}")
except Exception as e:
    print(f"Error saving vectorizer: {str(e)}")

# If joblib fails, try using pickle
if not os.path.exists(model_file_path) or not os.path.exists(vectorizer_file_path):
    import pickle
    
    try:
        with open(model_file_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Best model ({best_model_name}) saved using pickle as: {model_file_path}")
    except Exception as e:
        print(f"Error saving model with pickle: {str(e)}")
    
    try:
        with open(vectorizer_file_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Vectorizer saved using pickle as: {vectorizer_file_path}")
    except Exception as e:
        print(f"Error saving vectorizer with pickle: {str(e)}")

print("\nScript completed.")