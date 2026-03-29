from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

load_dotenv()
DATABASE_URL = os.getenv("DB_URL")
if not DATABASE_URL:
    raise ValueError("Database URL not found in environment variables.")
engine = create_engine(DATABASE_URL)
sessionLocal = sessionmaker(bind=engine, autocommit=False,autoflush=False)
Base = declarative_base()

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()