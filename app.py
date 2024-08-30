# import os
# import csv
# from flask import Flask, jsonify, request
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC  # Import SVM instead of Logistic Regression
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import logging

# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Paths
# base_path = os.path.dirname(os.path.abspath(__file__))
# #base_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/'
# model_path = os.path.join(base_path, 'models/best_model_svm.pkl')  # Update model filename
# vectorizer_path = os.path.join(base_path, 'models/best_vectorizer.pkl')
# data_path = os.path.join(base_path, 'data/raw_data.csv')

# # Load model and vectorizer
# def load_model_and_vectorizer():
#     global model, vectorizer
#     try:
#         model = joblib.load(model_path)
#         vectorizer = joblib.load(vectorizer_path)
#         logger.info("Model and vectorizer loaded successfully")
#     except Exception as e:
#         logger.error(f"Error loading model or vectorizer: {e}")
#         raise

# load_model_and_vectorizer()

# @app.route('/')
# def index():
#     return 'Fraud Detection API is running!'

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         text = request.get_json()
#         message = text.get('message', '')
        
#         if not message:
#             return jsonify({"error": 'No message found in request'}), 400
        
#         message_feature = vectorizer.transform([message])
        
#         # Get probability scores
#         probabilities = model.predict_proba(message_feature)[0]
#         fraud_probability = probabilities[1]  # Probability of being fraudulent
        
#         # Determine the result based on which probability is higher
#         result = 'fraudulent' if fraud_probability > 0.5 else 'legitimate'
        
#         return jsonify({
#             'message': message,
#             'prediction': result,
#             'fraud_probability': float(fraud_probability),
#             'fraud_percentage': f"{fraud_probability * 100:.2f}%"
#         })
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/newFraud', methods=['POST'])
# def new_fraud():
#     try:
#         data = request.get_json()
#         message = data.get('message', '')
#         category = data.get('category', '')
        
#         if not message or category not in ['legitimate', 'fraudulent']:
#             return jsonify({"error": 'Invalid input'}), 400
        
#         # Append new data to CSV
#         with open(data_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([message, category])
        
#         # Retrain model
#         retrain_model()
        
#         return jsonify({"message": "New data added and model retrained successfully"}), 200
#     except Exception as e:
#         logger.error(f"Error in adding new fraud data: {e}")
#         return jsonify({"error": str(e)}), 500

# def retrain_model():
#     global model, vectorizer
    
#     # Load data
#     df = pd.read_csv(data_path)
#     df['Category'] = df['Category'].map({'legitimate': 0, 'fraudulent': 1})
    
#     X = df['Message']
#     Y = df['Category']
    
#     # Split data
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
#     # Vectorize
#     vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
#     X_train_features = vectorizer.fit_transform(X_train)
#     X_test_features = vectorizer.transform(X_test)
    
#     # Train model
#     model = SVC(kernel='rbf', probability=True, random_state=42)  # Initialize SVM model
#     model.fit(X_train_features, Y_train)
    
#     # Evaluate
#     train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
#     test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))
    
#     logger.info(f"Model retrained. Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")
    
#     # Save model and vectorizer
#     joblib.dump(model, model_path)
#     joblib.dump(vectorizer, vectorizer_path)

# if __name__ == '__main__':
#     app.run()

#####################################################
# import os
# import csv
# from flask import Flask, jsonify, request
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import logging

# app = Flask(__name__)

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Paths
# base_path = os.environ.get('BASE_PATH', os.path.dirname(os.path.abspath(__file__)))
# model_path = os.path.join(base_path, 'models', 'best_model_svm.pkl')
# vectorizer_path = os.path.join(base_path, 'models', 'best_vectorizer.pkl')
# data_path = os.path.join(base_path, 'data', 'raw_data.csv')

# # Load model and vectorizer
# def load_model_and_vectorizer():
#     global model, vectorizer
#     try:
#         model = joblib.load(model_path)
#         vectorizer = joblib.load(vectorizer_path)
#         logger.info("Model and vectorizer loaded successfully")
#     except Exception as e:
#         logger.error(f"Error loading model or vectorizer: {e}")
#         raise

# # Only load the model if it's not a module import
# if __name__ != '__main__':
#     load_model_and_vectorizer()

# @app.route('/')
# def index():
#     return 'Fraud Detection API is running!'

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         text = request.get_json()
#         message = text.get('message', '')

#         if not message:
#             return jsonify({"error": 'No message found in request'}), 400

#         message_feature = vectorizer.transform([message])

#         # Get probability scores
#         probabilities = model.predict_proba(message_feature)[0]
#         fraud_probability = probabilities[1]  # Probability of being fraudulent

#         # Determine the result based on which probability is higher
#         result = 'fraudulent' if fraud_probability > 0.5 else 'legitimate'

#         return jsonify({
#             'message': message,
#             'prediction': result,
#             'fraud_probability': float(fraud_probability),
#             'fraud_percentage': f"{fraud_probability * 100:.2f}%"
#         })
#     except Exception as e:
#         logger.error(f"Error in prediction: {e}")
#         return jsonify({"error": str(e)}), 500

# @app.route('/newFraud', methods=['POST'])
# def new_fraud():
#     try:
#         data = request.get_json()
#         message = data.get('message', '')
#         category = data.get('category', '')

#         if not message or category not in ['legitimate', 'fraudulent']:
#             return jsonify({"error": 'Invalid input'}), 400

#         # Append new data to CSV
#         with open(data_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([message, category])

#         # Retrain model
#         retrain_model()

#         return jsonify({"message": "New data added and model retrained successfully"}), 200
#     except Exception as e:
#         logger.error(f"Error in adding new fraud data: {e}")
#         return jsonify({"error": str(e)}), 500

# def retrain_model():
#     global model, vectorizer

#     # Load data
#     df = pd.read_csv(data_path)
#     df['Category'] = df['Category'].map({'legitimate': 0, 'fraudulent': 1})

#     X = df['Message']
#     Y = df['Category']

#     # Split data
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#     # Vectorize
#     vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
#     X_train_features = vectorizer.fit_transform(X_train)
#     X_test_features = vectorizer.transform(X_test)

#     # Train model
#     model = SVC(kernel='rbf', probability=True, random_state=42)  # Initialize SVM model
#     model.fit(X_train_features, Y_train)

#     # Evaluate
#     train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
#     test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))

#     logger.info(f"Model retrained. Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")

#     # Save model and vectorizer
#     joblib.dump(model, model_path)
#     joblib.dump(vectorizer, vectorizer_path)

# if __name__ == '__main__':
#     load_model_and_vectorizer()
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)



import os
import csv
from flask import Flask, jsonify, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
base_path = os.environ.get('BASE_PATH', os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, 'models', 'best_model_svm.pkl')
vectorizer_path = os.path.join(base_path, 'models', 'best_vectorizer.pkl')
data_path = os.path.join(base_path, 'data', 'raw_data.csv')

# Global variables for model and vectorizer
model = None
vectorizer = None

def create_app():
    app = Flask(__name__)

    # Load model and vectorizer
    global model, vectorizer
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Model and vectorizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {e}")
        raise

    @app.route('/')
    def index():
        return 'Fraud Detection API is running!'

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            text = request.get_json()
            message = text.get('message', '')

            if not message:
                return jsonify({"error": 'No message found in request'}), 400

            message_feature = vectorizer.transform([message])

            # Get probability scores
            probabilities = model.predict_proba(message_feature)[0]
            fraud_probability = probabilities[1]  # Probability of being fraudulent

            # Determine the result based on which probability is higher
            result = 'fraudulent' if fraud_probability > 0.5 else 'legitimate'

            return jsonify({
                'message': message,
                'prediction': result,
                'fraud_probability': float(fraud_probability),
                'fraud_percentage': f"{fraud_probability * 100:.2f}%"
            })
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/newFraud', methods=['POST'])
    def new_fraud():
        try:
            data = request.get_json()
            message = data.get('message', '')
            category = data.get('category', '')

            if not message or category not in ['legitimate', 'fraudulent']:
                return jsonify({"error": 'Invalid input'}), 400

            # Append new data to CSV
            with open(data_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([message, category])

            # Retrain model
            retrain_model()

            return jsonify({"message": "New data added and model retrained successfully"}), 200
        except Exception as e:
            logger.error(f"Error in adding new fraud data: {e}")
            return jsonify({"error": str(e)}), 500

    return app

def retrain_model():
    global model, vectorizer

    # Load data
    df = pd.read_csv(data_path)
    df['Category'] = df['Category'].map({'legitimate': 0, 'fraudulent': 1})

    X = df['Message']
    Y = df['Category']

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Vectorize
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    # Train model
    model = SVC(kernel='rbf', probability=True, random_state=42)  # Initialize SVM model
    model.fit(X_train_features, Y_train)

    # Evaluate
    train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))

    logger.info(f"Model retrained. Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")

    # Save model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)