import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from flask import Flask, request, jsonify

app = Flask(__name__)

# load data
df = pd.read_csv('dataset/Student_Performance.csv')

# convert categorical variable 'Extracurricular Activities' to numerical
label = LabelEncoder()
df['Extracurricular Activities'] = label.fit_transform(df['Extracurricular Activities'])
# create 'final outcome' column based on 'Previous Scores' column. A score of 50 or above is considered a 1 (Pass) and below 50 is considered a 0 (Fail)
df['Final Outcome'] = np.where(df['Previous Scores'] >= 50, 1, 0)

# define features and target variable
X = df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Final Outcome']

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model training using Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# model evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

probabilities = y_pred_proba[:10]

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

df['Risk Score'] = model.predict_proba(scaler.transform(X))[:, 1]
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

@app.route('/predict', methods=['GET' ,'POST'])
def predict():
    data = request.get_json()
    hours_studied = data['Hours Studied']
    previous_scores = data['Previous Scores']
    extracurricular_activities = data['Extracurricular Activities']
    sleep_hours = data['Sleep Hours']
    sample_question_papers_practiced = data['Sample Question Papers Practiced']

    probability = model.predict_proba(scaler.transform([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers_practiced]]))[0][1]
    risk_category = categorize_risk(probability)
    recommendations = provide_recommendations(risk_category)

    response = {
        'risk_score': probability,
        'risk_category': risk_category,
        'recommendations': recommendations
    }

    return jsonify(response)

@app.route('/required_score', methods=['GET' ,'POST'])
def required_score():
    data = request.get_json()
    cum_weighted_marks = data['Cumulative Weighted Marks']
    target_cwa = data['Target CWA']
    total_credit_hours_obtained = data['Total Credit Hours Obtained']
    current_semester_credit_hours = data['Current Semester Credit Hours']

    required_score = calc_required_score(cum_weighted_marks, target_cwa, total_credit_hours_obtained, current_semester_credit_hours)

    response = {
        'required_score': required_score
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)