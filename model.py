import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify

app = Flask(__name__)



# load data
df = pd.read_csv('dataset/Student_Performance.csv')

# convert categorical variable 'Extracurricular Activities' to numerical
label = LabelEncoder()
df['Extracurricular Activities'] = label.fit_transform(df['Extracurricular Activities'])
# create 'final outcome' column based on 'Previous Scores' column. A score of 50 or above is considered a 1 (Pass) and below 50 is considered a 0 (Fail)
df['Final Outcome'] = np.where(df['Previous Scores'] >= 50, 1, 0)
print(df[['Previous Scores', 'Final Outcome']].head(20))


# define features and target variable
X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Final Outcome']

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=0.2)

# evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Risk Score and Classification
def categorize_risk(probability):
    if probability >= 0.999909:
        return 'Low Risk'
    elif probability > 0.7:
        return 'Moderate Risk'
    else:
        return 'High Risk'

df['Risk Score'] = model.predict(scaler.transform(X))[:, 0]
df['Risk Category'] = df['Risk Score'].apply(lambda x: categorize_risk(x))

# Provide recommendations based on the risk category
def provide_recommendations(risk_category):
    recommendations = {
        'Low Risk': [
            'Continue your current study routine.',
            'Explore additional extracurricular activities to enhance your skills.'
        ],
        'Moderate Risk': [
            'Increase your study hours by 2 hours per week.',
            'Practice more sample question papers to improve your problem-solving skills.'
        ],
        'High Risk': [
            'Increase study hours and practice more sample question papers.'
            'Seek help from teachers and peers to improve understanding of the subjects.',
            'Participate in study groups',
            'Get more sleep and maintain a healthy lifestyle.'
        ]
    }

    selected_recommendations = recommendations[risk_category]
    return selected_recommendations
    

df['Recommendations'] = df['Risk Category'].apply(provide_recommendations)

def calc_required_score(cum_weighted_marks, target_cwa, total_credit_hours_obtained, current_semester_credit_hours):
  total_credit_hours = total_credit_hours_obtained + current_semester_credit_hours
  required_cum_weighted_marks = (target_cwa * total_credit_hours) - (cum_weighted_marks)
  required_score = required_cum_weighted_marks / current_semester_credit_hours
  return required_score


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    hours_studied = data['hours_studied']
    previous_scores = data['previous_scores']
    extracurricular_activities = data['extracurricular_activities']
    sleep_hours = data['sleep_hours']
    sample_question_papers_practiced = data['sample_question_papers_practiced']

    prediction = model.predict(scaler.transform([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers_practiced]]))[0][0]
    risk_category = categorize_risk(prediction)
    recommendations = provide_recommendations(risk_category)

    return jsonify({
        'risk_score': prediction,
        'risk_category': risk_category,
        'recommendations': recommendations
    })

@app.route('/required_score', methods=['POST'])
def required_score():
    data = request.get_json()
    cum_weighted_marks = data['cum_weighted_marks']
    target_cwa = data['target_cwa']
    total_credit_hours_obtained = data['total_credit_hours_obtained']
    current_semester_credit_hours = data['current_semester_credit_hours']

    required_score = calc_required_score(cum_weighted_marks, target_cwa, total_credit_hours_obtained, current_semester_credit_hours)

    return jsonify({
        'required_score': required_score
    })

if __name__ == '__main__':
    app.run(debug=True)