# src/model.py
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD").replace("@", "%40")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_sql("SELECT * FROM cleaned_customers", engine)
    return df

# ── 2. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    df['tenure_group'] = pd.cut(df['tenure'],
                                 bins=[0, 12, 24, 48, 72],
                                 labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
    df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['has_support_services'] = (
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int)
    )
    df['is_echeque'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    return df

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, 'src/scaler.pkl')
    print("Scaler saved to src/scaler.pkl")

    return X_train, X_test, y_train, y_test, X.columns.tolist()

# ── 4. Train Models ──────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(
            scale_pos_weight=int((y_train == 0).sum() / (y_train == 1).sum()),
            use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"{name} — trained ✅")
    return trained

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
def evaluate_models(trained_models, X_test, y_test, feature_names):
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        results[name] = auc

        print(f"\n{'='*40}")
        print(f"{name} — AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                    cmap='Blues', cbar=False)
        axes[i].set_title(f'{name}\nAUC: {auc:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('src/confusion_matrices.png', dpi=150)
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = results[name]
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — All Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig('src/roc_curves.png', dpi=150)
    plt.show()

    best = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best} — AUC: {results[best]:.4f}")
    return best, results

# ── 6. Save Best Model ────────────────────────────────────────────────────────
def save_best_model(trained_models, best_name):
    model = trained_models[best_name]
    joblib.dump(model, 'src/best_model.pkl')
    print(f"Best model saved to src/best_model.pkl")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Engineering features...")
    df = engineer_features(df)

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    print("Training models...")
    trained_models = train_models(X_train, y_train)

    print("Evaluating...")
    best_name, results = evaluate_models(trained_models, X_test, y_test, feature_names)

    print("Saving best model...")
    save_best_model(trained_models, best_name)

    print("\n✅ Phase III Complete!")