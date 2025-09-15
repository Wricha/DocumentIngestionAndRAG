from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
Base = declarative_base()

class Documents(Base):
    __tablename__="documents"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String, index=True)
    metadata = Column(JSON)
    created_at = Column(DateTime, default = datetime.datetime.utcnow)

class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    date = Column(String, nullable=False)
    time = Column(String, nullable=False)
    metadata = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)