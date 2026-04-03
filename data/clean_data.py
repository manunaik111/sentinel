import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD").replace("@", "%40")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def clean_data():
    df = pd.read_sql("SELECT * FROM raw_customers", engine)
    print(f"Loaded from DB: {df.shape}")

    # Fix TotalCharges — it's a string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Median imputation for TotalCharges nulls
    median_val = df["TotalCharges"].median()
    df["TotalCharges"].fillna(median_val, inplace=True)
    print(f"TotalCharges nulls filled with median: {median_val:.2f}")

    # Encode target variable
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    # IQR outlier detection on MonthlyCharges
    Q1 = df["MonthlyCharges"].quantile(0.25)
    Q3 = df["MonthlyCharges"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df["MonthlyCharges"] < lower) | (df["MonthlyCharges"] > upper)]
    print(f"MonthlyCharges outliers found: {len(outliers)}")
    df = df[(df["MonthlyCharges"] >= lower) & (df["MonthlyCharges"] <= upper)]

    # Save cleaned data back to DB
    df.to_sql("cleaned_customers", engine, if_exists="replace", index=False)
    print(f"\nCleaned data saved to DB: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")

if __name__ == "__main__":
    clean_data()