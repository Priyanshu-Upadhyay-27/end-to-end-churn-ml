import os
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv
load_dotenv()

# 1. Grab the URL from Render's environment variables
raw_url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://myuser:mypassword@localhost:5432/churn_db")

# 2. Bulletproof conversion: Catch BOTH 'postgres://' and 'postgresql://'
if raw_url.startswith("postgres://"):
    db_url = raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_url.startswith("postgresql://"):
    # This block prevents SQLAlchemy from falling back to psycopg2
    db_url = raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    db_url = raw_url

# 3. Create the Engine using the corrected async URL
engine = create_async_engine(db_url, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()
class ChurnLog(Base):
    __tablename__ = "churn_predictions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    prediction_mode = Column(String, default="single")
    batch_id = Column(String, nullable=True, index=True)
    input_features = Column(JSONB, nullable=False)
    churn_probability = Column(Float, nullable=False)
    churn_prediction = Column(Integer, nullable=False)
    model_version = Column(String, default="v1.0")
    model_configs = Column(JSONB, nullable=True)