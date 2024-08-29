import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#read the data
df = pd.read_csv('/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/data/raw_data.csv')
data = df.where((pd.notnull(df)), '')

#
data.loc[data['Category'] == 'legitimate', 'Category'] = 0
data.loc[data['Category'] == 'fraudulent', 'Category'] = 1
X = data['Message']
Y = data['Category']

#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#convert the text data into numerical data
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression()

model.fit(X_train_features, Y_train)
prediction = model.predict(X_train_features)
accuracy_on_train_data = accuracy_score(Y_train, prediction)
print('Accuracy on training data : ', accuracy_on_train_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)
input_mail = ['hi how are you doing',]

input_mail_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_mail_features)
print(prediction)
if prediction[0] == 0:
    print('legitimate MAIL')
else:
    print('fraudulent MAIL')
    
import os

# Specify the folder path where you want to save the model
folder_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/models/'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save the model
model_file_path = os.path.join(folder_path, 'test_model.pkl')
joblib.dump(model, model_file_path)


# Save the feature extraction model     
# Specify the folder path where you want to save the vectorizer
vectorizer_folder_path = '/home/chota/Documents/School/4th Year/CS400/Code/SMSFinancialFraudDetectionSystem/DetectionModel/models/'
# Create the folder if it doesn't exist
if not os.path.exists(vectorizer_folder_path):
    os.makedirs(vectorizer_folder_path)

# Save the vectorizer
vectorizer_file_path = os.path.join(vectorizer_folder_path, 'test_tfidf_vectorizer.pkl')
joblib.dump(feature_extraction, vectorizer_file_path)
