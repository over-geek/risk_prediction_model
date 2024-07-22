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
def provide_recommendations(risk_category, hours_studied, extracurricular_activities, sleep_hours, sample_question_papers_practiced):
    recommendations = []

    if risk_category == 'Low Risk':
        recommendations.append('Continue your current study routine.')
        if extracurricular_activities < 5:
            recommendations.append('Explore additional extracurricular activities to enhance your skills.')
        if sleep_hours < 7:
            recommendations.append('Try to get at least 7 hours of sleep each night to maintain a healthy lifestyle.')

    elif risk_category == 'Moderate Risk':
        if hours_studied < 20:
            recommendations.append('Increase your study hours by 2 hours per week.')
        if sample_question_papers_practiced < 5:
            recommendations.append('Practice more sample question papers to improve your problem-solving skills.')
        if sleep_hours < 7:
            recommendations.append('Ensure you are getting enough sleep (at least 7 hours).')
        if extracurricular_activities < 3:
            recommendations.append('Consider balancing your study with some extracurricular activities for overall development.')

    elif risk_category == 'High Risk':
        if hours_studied < 25:
            recommendations.append('Significantly increase your study hours.')
        if sample_question_papers_practiced < 7:
            recommendations.append('Practice more sample question papers.')
        if sleep_hours < 7:
            recommendations.append('Ensure you get more sleep (at least 7-8 hours) to help with retention and focus.')
        recommendations.append('Seek help from teachers and peers to improve understanding of the subjects.')
        recommendations.append('Participate in study groups.')
        if extracurricular_activities < 2:
            recommendations.append('Consider adding some extracurricular activities to reduce stress.')

    return recommendations

def risk_description(risk_category):
    if risk_category == 'Low Risk':
        return 'You are currently performing well in your studies and have a low risk of academic failure. Your study habits, engagement in extracurricular activities, sleep patterns, and practice with sample question papers are contributing positively to your academic success.'
    elif risk_category == 'Moderate Risk':
        return 'You are on the verge of meeting your academic goals, but need to take extra steps to ensure you do not fall behind. With a few adjustments to your study habits and practice, you can overcome this moderate risk and achieve your desired outcomes.'
    else:
        return 'You are at a high risk of academic failure. Significant improvements are needed in your study habits, sleep patterns, and engagement in extracurricular activities to enhance your academic performance.'


df['Recommendations'] = df.apply(
    lambda row: provide_recommendations(
        row['Risk Category'],
        row['Hours Studied'],
        row['Extracurricular Activities'],
        row['Sleep Hours'],
        row['Sample Question Papers Practiced']
    ), axis=1
)

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
    risk_info = risk_description(risk_category)
    recommendations = provide_recommendations(risk_category, hours_studied, extracurricular_activities, sleep_hours, sample_question_papers_practiced)

    response = {
        'risk_score': probability,
        'risk_category': risk_category,
        'risk_info': risk_info,
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