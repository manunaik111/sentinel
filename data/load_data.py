import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

password = DB_PASSWORD.replace("@", "%40")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def load_raw_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")

    df.to_sql("raw_customers", engine, if_exists="replace", index=False)
    print("\nData loaded into PostgreSQL table: raw_customers")

if __name__ == "__main__":
    load_raw_data()