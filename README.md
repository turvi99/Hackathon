# 🩺 Digital Twin — Heart Risk Predictor

A full-stack AI-powered health informatics web application that predicts an individual's **10-year risk of developing Coronary Heart Disease (CHD)** using the Framingham Heart Study dataset and a trained Random Forest model, augmented with a psychosocial "Digital Twin" emotion modifier.

Built during a competitive hackathon — end-to-end from ML training to Flask deployment.

---

## 🔬 How It Works

1. **User authenticates** via a styled login page with session management
2. **Dashboard hub** provides navigation to the prediction tool, feature glossary, and logout
3. **Prediction form** collects 15 clinical biomarkers + 1 emotional state input
4. **Random Forest model** computes base CHD probability from clinical features
5. **Emotion modifier** adjusts risk based on psychosocial state (Happy: ×0.95, Neutral: ×1.00, Sad: ×1.05, Stressed: ×1.10)
6. **Risk stratification** classifies into Low (<30%), Moderate (30–70%), or High (≥70%) with color-coded animated gauge and personalized health tips

---

## 🧠 Machine Learning

| Component | Details |
|---|---|
| **Model** | `RandomForestClassifier` (Scikit-Learn) |
| **Dataset** | Framingham Heart Study — 4,240 patient records |
| **Features** | 15 clinical biomarkers (age, sex, cholesterol, BP, BMI, glucose, smoking, etc.) |
| **Target** | `TenYearCHD` (binary: 10-year coronary heart disease risk) |
| **Serialization** | Joblib `.pkl` (~7.9 MB, includes optional scaler) |

---

## 📁 Project Structure

```
Hackathon/
├── app.py                    # Flask backend — routing, authentication, ML inference
├── heartdisease_predict.pkl  # Trained RandomForest model (serialized via Joblib)
├── framingham.csv            # Framingham Heart Study dataset (4,240 records × 16 cols)
├── index.html                # Main prediction form + animated risk meter + health tips
├── dash.html                 # Dashboard navigation hub (glassmorphic card design)
├── details.html              # Clinical feature glossary with reference ranges
├── login.html                # User authentication page
├── flask_page.html           # Auxiliary Flask route template
└── style.css                 # Shared UI styling
```

---

## 🖥️ Tech Stack

- **Backend:** Python, Flask, Jinja2, Joblib, NumPy
- **ML:** Scikit-Learn (RandomForestClassifier)
- **Frontend:** HTML5, CSS3 (Flexbox, gradients, animations), Vanilla JavaScript
- **Dataset:** Framingham Heart Study (NIH/NHLBI)
- **UI Design:** Responsive cards, animated risk gauge, Poppins & Roboto fonts

---

## 🩻 Input Features (15 Biomarkers)

| # | Feature | Description |
|---|---|---|
| 1 | `male` | Sex (1 = Male, 0 = Female) |
| 2 | `age` | Age in years |
| 3 | `education` | Educational attainment (1–4 ordinal) |
| 4 | `currentSmoker` | Smoking status (1/0) |
| 5 | `cigsPerDay` | Cigarettes per day |
| 6 | `BPMeds` | On blood pressure medication (1/0) |
| 7 | `prevalentStroke` | History of stroke (1/0) |
| 8 | `prevalentHyp` | Prevalent hypertension (1/0) |
| 9 | `diabetes` | Diagnosed diabetes (1/0) |
| 10 | `totChol` | Total cholesterol (mg/dL) |
| 11 | `sysBP` | Systolic blood pressure (mmHg) |
| 12 | `diaBP` | Diastolic blood pressure (mmHg) |
| 13 | `BMI` | Body Mass Index (kg/m²) |
| 14 | `heartRate` | Resting heart rate (bpm) |
| 15 | `glucose` | Fasting blood glucose (mg/dL) |

---

## 🚦 Risk Stratification

| Risk Level | Threshold | Color | Action |
|---|---|---|---|
| ✅ Low | < 30% | Green | Lifestyle maintenance, positive reinforcement |
| ⚠️ Moderate | 30–70% | Orange | Diet modification, BP/glucose monitoring, stress reduction |
| 🚨 High | ≥ 70% | Red | Smoking cessation, lipid control, cardiology referral |

---

## 🚀 How to Run

```bash
pip install flask joblib numpy scikit-learn
python app.py
```

Open `http://127.0.0.1:5000` → Login with `user` / `password123`

---

## 👤 Author

**Turvi Devineni**  
B.Tech CSE (AI & ML) — SRM University AP  
📧 turvi992005@gmail.com
