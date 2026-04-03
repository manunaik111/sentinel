# Sentinel — End-to-End Customer Retention Engine

> *Predict churn. Retain customers. Ship ML that works.*

![Python](https://img.shields.io/badge/Python-3.11-white?style=for-the-badge&logo=python&logoColor=white&labelColor=000000&color=111111)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-white?style=for-the-badge&logo=fastapi&logoColor=white&labelColor=000000&color=111111)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-white?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=000000&color=111111)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-white?style=for-the-badge&logo=postgresql&logoColor=white&labelColor=000000&color=111111)
![Docker](https://img.shields.io/badge/Docker-29.3-white?style=for-the-badge&logo=docker&logoColor=white&labelColor=000000&color=111111)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-white?style=for-the-badge&logo=xgboost&logoColor=white&labelColor=000000&color=111111)
![AUC](https://img.shields.io/badge/AUC-0.846-white?style=for-the-badge&labelColor=000000&color=111111)

---

## What is Sentinel?

Sentinel is a production-grade **customer churn prediction system** built on the IBM Telco dataset (7,043 customers). It combines a full ML pipeline — from raw data ingestion to a live prediction API — with a real-time analytics dashboard, containerized and ready to deploy.

**Target outcome:** Reduce churn by 15% through early intervention on high-risk customers.

---

## Live Demo

| Interface | URL |
|---|---|
| Streamlit Dashboard | `http://localhost:8501` |
| FastAPI Swagger UI | `http://localhost:8000/docs` |
| Health Check | `http://localhost:8000/health` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Sentinel                             │
│                                                             │
│   CSV Data                                                  │
│      │                                                      │
│      ▼                                                      │
│   PostgreSQL ──► Python Pipeline ──► Trained Model (.pkl)   │
│      │                │                      │              │
│      │           EDA Notebook           FastAPI API         │
│      │                                        │              │
│      └──────────────────────────► Streamlit Dashboard       │
│                                                             │
│   [ All containerized via Docker ]                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL 16 |
| Data Processing | Python · Pandas · NumPy · SQLAlchemy |
| Modeling | Scikit-Learn · XGBoost · Joblib |
| API | FastAPI · Uvicorn · Pydantic |
| Dashboard | Streamlit · Plotly |
| Containerization | Docker |
| Testing | Pytest · HTTPX |
| Notebook | Jupyter · IPyKernel |

---

## Model Performance

Three models were trained and evaluated on a stratified 80/20 split with class imbalance handled via `class_weight='balanced'` and `scale_pos_weight`.

| Model | AUC | Churn Recall | Precision |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.846** | **78%** | 51% |
| XGBoost | 0.821 | 63% | 57% |
| Random Forest | 0.820 | 47% | 65% |

**Logistic Regression** was selected as the best model. For a retention engine, **churn recall matters most** — it's better to flag a loyal customer than miss an at-risk one. LR's 78% recall on churners outperforms tree-based models on this dataset.

### Key Churn Drivers (from EDA)

| Feature | Correlation | Insight |
|---|---|---|
| `tenure` | −0.352 | Newer customers churn more |
| `InternetService_Fiber optic` | +0.308 | Fiber users are high risk |
| `Contract_Two year` | −0.302 | Long-term contracts retain |
| `PaymentMethod_Electronic check` | +0.302 | E-cheque = high risk signal |
| `TechSupport / OnlineSecurity` | −0.228 | Add-ons improve retention |

---

## Project Structure

```
Sentinel/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← raw dataset
│   ├── load_data.py                             ← loads CSV → PostgreSQL
│   └── clean_data.py                            ← cleans + encodes data
│
├── notebooks/
│   └── 01_EDA.ipynb                             ← full exploratory analysis
│
├── src/
│   ├── model.py                                 ← feature eng + training
│   ├── app.py                                   ← FastAPI prediction API
│   ├── best_model.pkl                           ← saved Logistic Regression
│   └── scaler.pkl                               ← fitted StandardScaler
│
├── dashboard/
│   └── streamlit_app.py                         ← analytics dashboard
│
├── tests/
│   └── test_api.py                              ← API tests
│
├── Dockerfile
├── .env                                         ← DB credentials (not committed)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 16 running locally
- Docker (optional, for containerized deployment)

### 2. Clone & Setup

```bash
git clone https://github.com/yourusername/sentinel.git
cd sentinel

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install matplotlib seaborn
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sentinel_db
DB_USER=postgres
DB_PASSWORD=your_password
```

> ⚠️ If your password contains `@`, it is automatically URL-encoded in all scripts. No manual changes needed.

### 4. Load & Clean Data

```bash
# Load raw CSV into PostgreSQL
python data/load_data.py

# Clean and transform data
python data/clean_data.py
```

This creates two tables in PostgreSQL:
- `raw_customers` — 7,043 rows, 21 columns
- `cleaned_customers` — 7,043 rows, 20 columns (encoded, ready for modeling)

### 5. Train the Model

```bash
python src/model.py
```

Outputs:
- `src/best_model.pkl` — trained Logistic Regression
- `src/scaler.pkl` — fitted StandardScaler
- `src/confusion_matrices.png`
- `src/roc_curves.png`

### 6. Run the API

```bash
uvicorn src.app:app --reload
```

API is live at `http://127.0.0.1:8000`
Swagger UI at `http://127.0.0.1:8000/docs`

### 7. Run the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard at `http://localhost:8501`

---

## API Usage

### `POST /predict`

Send a customer profile and receive a churn prediction with probability and risk level.

**Request body:**

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

**Risk levels:**

| Probability | Risk Level |
|---|---|
| ≥ 70% | High |
| 40–70% | Medium |
| < 40% | Low |

---

## Docker Deployment

```bash
# Build image
docker build -t sentinel-api .

# Run container
docker run -p 8000:8000 --env-file .env sentinel-api
```

API available at `http://localhost:8000`

---

## Dashboard Features

### Analytics Page
- **KPI strip** — Total customers · Churned · Retained · Avg churn bill
- **Churn split** — Donut chart with live percentage
- **Contract type** — Horizontal bar chart showing churn rate by contract
- **Tenure distribution** — Overlaid histogram by churn status
- **Monthly charges** — Box plot comparison (retained vs churned)
- **Internet service** — Bar chart of churn rate by service type
- **Payment method** — Horizontal bar sorted by risk
- **Scatter plot** — Tenure vs MonthlyCharges colored by churn

### Predict Page
- **3-column input form** — Demographics · Services · Billing
- **Live inference** — Runs model on every prediction click
- **Churn probability card** — Large % display with risk level badge
- **Risk gauge** — Plotly gauge chart with B&W gradient
- **Risk factor breakdown** — Automatic detection of high-risk signals

---

## Feature Engineering

Four engineered features were added before modeling:

| Feature | Description |
|---|---|
| `tenure_group` | Binned tenure: 0–1yr, 1–2yr, 2–4yr, 4–6yr |
| `charges_per_tenure` | `TotalCharges / (tenure + 1)` — spend efficiency |
| `has_support_services` | Count of TechSupport + OnlineSecurity + OnlineBackup |
| `is_echeque` | Binary flag for Electronic check payment |

---

## EDA Highlights

From `notebooks/01_EDA.ipynb`:

- Churners have **avg tenure of 18 months** vs 38 months for retained customers
- Churners pay **$13/month more** on average (74 vs 61)
- **Month-to-month contracts** have a 43% churn rate vs 3% for two-year contracts
- **Fiber optic** customers churn at 42% vs 19% for DSL users
- **Electronic check** payment method has the highest churn rate of all payment types
- Dataset has a **26.5% churn rate** — handled with `class_weight='balanced'`

---

## Requirements

```
pandas==2.2.2
numpy==1.26.4
sqlalchemy==2.0.30
psycopg2-binary==2.9.9
scikit-learn==1.4.2
xgboost==2.0.3
joblib==1.4.2
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
streamlit==1.35.0
plotly==5.22.0
pytest==8.2.0
httpx==0.27.0
python-dotenv==1.0.1
jupyter==1.0.0
ipykernel==6.29.4
matplotlib
seaborn
```

---

## Dataset

**IBM Telco Customer Churn Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers · 21 features · Binary churn label
- Features include demographics, services subscribed, contract type, billing info

---

## Roadmap

- [ ] SHAP explainability for individual predictions
- [ ] Batch prediction endpoint (`POST /predict/batch`)
- [ ] Automated retraining pipeline
- [ ] Email alerts for high-risk customers
- [ ] Streamlit Cloud deployment

---

## Author

**Manu Naik**
Built as a portfolio project targeting ₹350–500k CTC roles · June 2026

---

*Sentinel — Know who's leaving before they do.*