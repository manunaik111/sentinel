<div align="center">

# Sentinel
### End-to-End Customer Retention Engine

*Predict churn. Retain customers. Ship ML that works.*

![Python](https://img.shields.io/badge/Python-3.11-000000?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-000000?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-000000?style=flat-square&logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-000000?style=flat-square&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-29.3-000000?style=flat-square&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-000000?style=flat-square&logo=xgboost&logoColor=white)
![AUC](https://img.shields.io/badge/AUC-0.846-000000?style=flat-square)
![Neon](https://img.shields.io/badge/Neon-DB-000000?style=flat-square)

<br>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-sentinel002.streamlit.app-000000?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentinel002.streamlit.app/)

</div>

---

## What is Sentinel?

Sentinel is a production-grade **customer churn prediction system** built on the IBM Telco dataset (7,043 customers). It combines a full ML pipeline — from raw data ingestion to a live prediction API — with a real-time analytics dashboard, containerized and ready to deploy.

**Target outcome:** Reduce churn by 15% through early intervention on high-risk customers.

---

## Key Numbers

| Metric | Value |
|---|---|
| Dataset | 7,043 customers · 21 features |
| Churn Rate | 26.5% (1,869 churned) |
| Best Model AUC | **0.846** — Logistic Regression |
| Churn Recall | **78%** on test set |
| DB | Neon PostgreSQL (cloud) |
| Deployment | Streamlit Cloud + Docker |

---

## Architecture

```
CSV Data
   │
   ▼
PostgreSQL (Neon Cloud)
   │
   ▼
Python Pipeline ──────────────► Trained Model (.pkl)
   │                                    │
   ▼                                    ▼
EDA Notebook                      FastAPI API
(01_EDA.ipynb)                   /predict endpoint
                                        │
                                        ▼
                              Streamlit Dashboard
                          ┌───────────────────────┐
                          │  Analytics   Predictor │
                          └───────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Database | PostgreSQL (Neon Cloud) | 16 |
| Data Processing | Python · Pandas · NumPy · SQLAlchemy | 2.2.2 |
| Modeling | Scikit-Learn · XGBoost · Joblib | 1.4.2 / 2.0.3 |
| API | FastAPI · Uvicorn · Pydantic | 0.111.0 |
| Dashboard | Streamlit · Plotly | 1.35.0 |
| Containerization | Docker | 29.3 |
| Testing | Pytest · HTTPX | 8.2.0 |

---

## Model Performance

| Model | AUC | Churn Recall | Precision | Status |
|---|---|---|---|---|
| **Logistic Regression** | **0.846** | **78%** | 51% | ✅ Selected |
| XGBoost | 0.821 | 63% | 57% | — |
| Random Forest | 0.820 | 47% | 65% | — |

**Why Logistic Regression?** For a retention engine, churn recall matters most. It is better to flag a loyal customer than miss an at-risk one. LR's 78% recall on churners outperforms tree-based models on this dataset.

---

## Top Churn Drivers

| Feature | Correlation | Insight |
|---|---|---|
| `tenure` | −0.352 | New customers churn fast |
| `InternetService_Fiber optic` | +0.308 | Fiber users pay more, churn more |
| `Contract_Two year` | −0.302 | Long-term contracts strongly retain |
| `PaymentMethod_Electronic check` | +0.302 | Top payment risk signal |
| `TechSupport / OnlineSecurity` | −0.228 | Add-on services retain customers |

---

## Project Structure

```
Sentinel/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── load_data.py          ← CSV → PostgreSQL
│   └── clean_data.py         ← clean + encode → cleaned_customers
│
├── notebooks/
│   └── 01_EDA.ipynb          ← full exploratory analysis (6 sections)
│
├── src/
│   ├── model.py              ← feature engineering + 3-model training
│   ├── app.py                ← FastAPI prediction API
│   ├── best_model.pkl        ← saved Logistic Regression
│   └── scaler.pkl            ← fitted StandardScaler
│
├── dashboard/
│   └── streamlit_app.py      ← B&W analytics dashboard
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .env                      ← not committed
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone & Setup

```bash
git clone https://github.com/manunaik111/sentinel.git
cd sentinel

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```env
DB_HOST=your-neon-host.neon.tech
DB_PORT=5432
DB_NAME=neondb
DB_USER=neondb_owner
DB_PASSWORD=your_password
```

### 3. Load & Clean Data

```bash
python data/load_data.py
python data/clean_data.py
```

### 4. Train the Model

```bash
python src/model.py
```

Outputs `src/best_model.pkl` and `src/scaler.pkl`.

### 5. Run the API

```bash
uvicorn src.app:app --reload
```

Swagger UI → `http://127.0.0.1:8000/docs`

### 6. Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard → `http://localhost:8501`

---

## API Usage

### POST `/predict`

```json
{
  "gender": "Male",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.5,
  "TotalCharges": 191.0
}
```

**Response:**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8369,
  "risk_level": "High",
  "message": "⚠️ Customer is at risk of churning."
}
```

**Risk Levels:**

| Probability | Risk |
|---|---|
| ≥ 70% | High |
| 40–70% | Medium |
| < 40% | Low |

---

## Docker

```bash
docker build -t sentinel-api .
docker run -p 8000:8000 --env-file .env sentinel-api
```

---

## Dashboard Features

### Analytics Page
- KPI strip — Total · Churned · Retained · Avg churn bill
- Churn donut chart
- Churn rate by contract type
- Tenure distribution by churn
- Monthly charges box plot
- Internet service risk bar
- Payment method risk bar
- Tenure vs charges scatter plot

### Predict Page
- 3-column input form — Demographics · Services · Billing
- Real-time churn probability
- Risk gauge (Plotly)
- Auto-detected risk factor breakdown

---

## Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `tenure_group` | Binned | 0–1yr · 1–2yr · 2–4yr · 4–6yr |
| `charges_per_tenure` | Ratio | TotalCharges / (tenure + 1) |
| `has_support_services` | Count | TechSupport + OnlineSecurity + OnlineBackup |
| `is_echeque` | Binary | Electronic check payment flag |

---

## EDA Highlights

- Churners average **18 months tenure** vs 38 months for retained
- Churners pay **$13/month more** on average
- **Month-to-month** contracts churn at 43% vs 3% for two-year
- **Fiber optic** churn rate is 42% vs 19% for DSL
- Class imbalance handled with `class_weight='balanced'`

---

## Streamlit Secrets

```toml
DB_HOST = "your-host.neon.tech"
DB_PORT = "5432"
DB_NAME = "neondb"
DB_USER = "neondb_owner"
DB_PASSWORD = "your_password"
```

---

## Requirements

```
pandas==2.2.2          numpy==1.26.4
sqlalchemy==2.0.30     psycopg2-binary==2.9.9
scikit-learn==1.4.2    xgboost==2.0.3
joblib==1.4.2          fastapi==0.111.0
uvicorn==0.29.0        pydantic==2.7.1
streamlit==1.35.0      plotly==5.22.0
pytest==8.2.0          httpx==0.27.0
python-dotenv==1.0.1   matplotlib==3.9.0
seaborn==0.13.2
```

---

## Roadmap

- [ ] SHAP explainability for individual predictions
- [ ] Batch prediction endpoint `POST /predict/batch`
- [ ] Automated retraining pipeline
- [ ] Email alerts for high-risk customers

---

<div align="center">

**Manu Naik** 

*Sentinel — Know who's leaving before they do.*

</div>
