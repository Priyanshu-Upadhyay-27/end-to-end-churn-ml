import os
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from dotenv import load_dotenv
load_dotenv()

raw_db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./local_fallback.db")

# --- DYNAMIC CLOUD DRIVER PARSING ---
#raw_db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./local_fallback.db")

if raw_db_url.startswith("postgres://"):
    db_url = raw_db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif raw_db_url.startswith("postgresql://") and "asyncpg" not in raw_db_url:
    db_url = raw_db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    db_url = raw_db_url

# --- DB ENGINE SETUP ---
engine = create_async_engine(db_url, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# --- SCHEMA ---
class ChurnLog(Base):
    __tablename__ = "churn_predictions"

    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    prediction_mode = Column(String, default="single")
    batch_id = Column(String, index=True, nullable=True)
    input_features = Column(JSONB, nullable=False)
    churn_probability = Column(Float, nullable=False)
    churn_prediction = Column(Integer, nullable=False)

# --- INDEPENDENT ABSTRACTION FUNCTIONS ---
async def init_db():
    """Called by api.py on startup to create tables without exposing the engine."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def save_predictions_to_db(predictions_list: list, mode: str, batch_id: str = None):
    """Handles saving both single and batch predictions strictly inside the DB layer."""
    async with AsyncSessionLocal() as session:
        try:
            db_logs = []
            for item in predictions_list:
                log = ChurnLog(
                    id=str(uuid.uuid4()),
                    prediction_mode=mode,
                    batch_id=batch_id,
                    input_features=item["features"],
                    churn_probability=item["probability"],
                    churn_prediction=item["prediction"]
                )
                db_logs.append(log)
            session.add_all(db_logs)
            await session.commit()
        except Exception as e:
            await session.rollback()
            print(f"DB ERROR: Failed to save logs - {e}")