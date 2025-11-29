from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "secret_key_123"

# Load ML model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "heartdisease_predict.pkl")

loaded_obj = joblib.load(model_path)
if isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
    scaler, model = loaded_obj
else:
    scaler, model = None, loaded_obj

# Simple login credentials
USER_CREDENTIALS = {"user": "password123"}

# -------------------------------
# LOGIN ROUTES
# -------------------------------
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if USER_CREDENTIALS.get(username) == password:
        session['logged_in'] = True
        return redirect(url_for('dashboard'))   # Changed to redirect to dashboard
    else:
        return render_template('login.html', error="Invalid username or password")

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dash.html')  # Changed from dashboard.html to dash.html

@app.route('/flask_page')
def flask_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('flask_page.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# -------------------------------
# MAIN HEART RISK PREDICTOR PAGE
# -------------------------------
@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')  # your main predictor HTML

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    try:
        # Collect form data
        male = float(request.form['male'])
        age = float(request.form['age'])
        education = float(request.form['education'])
        currentSmoker = float(request.form['currentSmoker'])
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = float(request.form['BPMeds'])
        prevalentStroke = float(request.form['prevalentStroke'])
        prevalentHyp = float(request.form['prevalentHyp'])
        diabetes = float(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])
        emotion = int(float(request.form.get('emotion', 1)))

        features = np.array([[male, age, education, currentSmoker, cigsPerDay,
                              BPMeds, prevalentStroke, prevalentHyp, diabetes,
                              totChol, sysBP, diaBP, BMI, heartRate, glucose]])

        if scaler:
            features = scaler.transform(features)

        # Make prediction
        if hasattr(model, "predict_proba"):
            base_risk = model.predict_proba(features)[0, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(features)[0]
            base_risk = 1 / (1 + np.exp(-score))
        else:
            base_risk = float(model.predict(features)[0])

        if base_risk > 1:
            base_risk /= 100

        # Adjust based on emotion
        emotion_factors = {0: 0.95, 1: 1.0, 2: 1.05, 3: 1.1}
        adjusted_risk = np.clip(base_risk * emotion_factors.get(emotion, 1.0), 0, 1)

        # Risk category
        if adjusted_risk < 0.3:
            category = "✅ Low Risk (No Immediate Concern)"
            color = "green"
        elif adjusted_risk < 0.7:
            category = "⚠️ Moderate Risk (Monitor & Improve Lifestyle)"
            color = "orange"
        else:
            category = "🚨 High Risk (Consult a Doctor)"
            color = "red"

        return render_template(
            'index.html',
            prediction_text=f"Predicted 10-Year Risk: {adjusted_risk * 100:.1f}%",
            result=category,
            color=color,
            adjusted_risk=adjusted_risk
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# Add this route with your other routes
@app.route('/details')
def details():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('details.html')

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
