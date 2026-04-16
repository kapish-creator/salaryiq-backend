from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import os
import joblib 

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'salary_model.pkl'))
mlb = joblib.load(os.path.join(BASE_DIR, 'skills_encoder.pkl'))
columns = joblib.load(os.path.join(BASE_DIR, 'columns.pkl'))

@app.route('/')
def home():
    return "Salary Prediction API Running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    name = data.get('name', 'User')
    age = data.get('age', None)
    experience = data['experience']
    skills = data['skills']
    job_role = data['job_role']
    work_mode = data.get('work_mode', '')
    target_location = data.get('target_location', '')

    # Create empty input
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # Add experience
    if 'experience_years' in input_df.columns:
        input_df['experience_years'] = experience

    # Encode skills
    for skill in skills:
        skill = skill.strip()
        if skill in input_df.columns:
            input_df[skill] = 1
        elif f" {skill}" in input_df.columns:
            input_df[f" {skill}"] = 1
        else:
            for c in input_df.columns:
                if c.lower() == skill.lower() or c.lower() == f" {skill}".lower():
                    input_df[c] = 1

    # Encode job role
    role_col = f"job_role_{job_role}"
    if role_col in input_df.columns:
        input_df[role_col] = 1

    # Encode work mode
    wm_col = f"work_mode_{work_mode}"
    if wm_col in input_df.columns:
        input_df[wm_col] = 1
        
    # Encode Target Location
    loc_col = f"location_{target_location}"
    if loc_col in input_df.columns:
        input_df[loc_col] = 1

    # Prediction
    prediction = model.predict(input_df)[0]

    return jsonify({
        "predicted_salary": round(prediction, 2),
        "name": name
    })

if __name__ == "__main__":
    app.run(debug=True)