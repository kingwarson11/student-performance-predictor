<div align="center">

<br/>

```
███████╗██████╗ ██╗   ██╗██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗
██╔════╝██╔══██╗██║   ██║██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝
█████╗  ██║  ██║██║   ██║██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   
██╔══╝  ██║  ██║██║   ██║██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   
███████╗██████╔╝╚██████╔╝██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   
╚══════╝╚═════╝  ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝  
```

### Track · Calculate · Predict

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20EduPredict-2563eb?style=for-the-badge)](https://student-performance-predictor-1-m1u2.onrender.com)
&nbsp;
![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)
&nbsp;
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
&nbsp;
![ML](https://img.shields.io/badge/ML-RF%20+%20GB%20Ensemble-16a34a?style=for-the-badge&logo=scikit-learn&logoColor=white)

<br/>

> **EduPredict** is a smart academic companion for university students — enter your scores, track your CWA or GPA in real time, and let a trained ML model tell you where you're headed before results drop.

<br/>

</div>

---

## ✨ What It Does

<table>
<tr>
<td width="50%">

### 📝 Grade Entry
Enter your **CA and exam scores** for every course across all levels and semesters. EduPredict instantly computes your total score, letter grade, weighted average, semester SWA, and cumulative CWA or GPA — no spreadsheet needed.

</td>
<td width="50%">

### 🔁 GPA / CWA Toggle
One switch. Two systems. **CWA mode** follows the weighted percentage formula used across Ghanaian universities. **GPA mode** maps your grades to the standard 4.0 scale. Flip between them anytime.

</td>
</tr>
<tr>
<td width="50%">

### 📊 Full Transcript View
See every semester at a glance — SWA per semester, cumulative CWA/GPA, a **degree classification banner**, and a visual progress tracker showing exactly how far you've come.

</td>
<td width="50%">

### 🎯 First Class Target
Find out exactly what score you need in your remaining courses to hit **First Class, Second Upper, Second Lower, or Third Class**. Includes a what-if simulator — adjust a hypothetical score and watch your projected CWA update live.

</td>
</tr>
<tr>
<td colspan="2">

### 🤖 AI-Powered Prediction
Your academic profile gets fed into a **Random Forest + Gradient Boosting ensemble model** trained on real student data. It predicts your expected final grade, shows the probability of each outcome, flags your risk level, and gives you personalised recommendations to improve.

</td>
</tr>
</table>

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5 · CSS3 · Vanilla JavaScript |
| **Backend** | Python 3.11 · Flask · Flask-CORS |
| **ML Model** | scikit-learn — Random Forest + Gradient Boosting Ensemble |
| **Data** | Student performance dataset (395 samples, 19 features) |
| **Hosting** | Render (Web Service) |

---

## 🚀 Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/kingwarson11/codequest.git
cd codequest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python backend/train.py

# 4. Start the server
python backend/app.py
```

Then open `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
├── backend/
│   ├── app.py          # Flask API + static file serving
│   └── train.py        # Model training script
├── frontend/
│   └── index.html      # Full single-page app
├── models/
│   ├── model.pkl       # Trained ensemble model
│   ├── scaler.pkl      # Feature scaler
│   ├── label_encoder.pkl
│   ├── feature_names.json
│   └── stats.json      # Model accuracy stats
├── data/
│   └── student_data.csv
├── requirements.txt
└── render.yaml         # Render deployment config
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Run ML prediction on student profile |
| `GET` | `/api/stats` | Return model accuracy and training stats |
| `GET` | `/api/health` | Health check + model load status |

---

<div align="center">

<br/>

**Built for university students. Powered by machine learning.**

<br/>

[![Try EduPredict](https://img.shields.io/badge/Try%20EduPredict%20Now-→-2563eb?style=for-the-badge)](https://student-performance-predictor-1-m1u2.onrender.com)

<br/><br/>

</div>
