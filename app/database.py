"""
Database models and configuration for Resume Insight application.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Optional, Generator
import redis
from app.config import settings

# Database setup
engine = create_engine(settings.database.url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.from_url(settings.database.redis_url)


class User(Base):
    """User model for storing user information."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Resume(Base):
    """Resume model for storing resume information and analysis results."""
    
    __tablename__ = "resumes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Extracted content
    raw_text = Column(Text, nullable=True)
    sections = Column(JSON, nullable=True)  # Store extracted sections as JSON
    
    # Analysis results
    ats_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)
    word_count = Column(Integer, nullable=True)
    section_count = Column(Integer, nullable=True)
    
    # Skills analysis
    technical_skills = Column(JSON, nullable=True)
    soft_skills = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    analyzed_at = Column(DateTime, nullable=True)


class Analysis(Base):
    """Analysis model for storing detailed analysis results."""
    
    __tablename__ = "analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resume_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    job_role = Column(String(100), nullable=False)
    
    # Scoring results
    ats_score = Column(Float, nullable=False)
    role_match_percentage = Column(Float, nullable=False)
    overall_score = Column(Float, nullable=False)
    
    # Detailed analysis
    found_keywords = Column(JSON, nullable=True)
    missing_keywords = Column(JSON, nullable=True)
    grammar_issues = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # LLM-generated content
    ai_recommendations = Column(Text, nullable=True)
    personalized_suggestions = Column(Text, nullable=True)
    
    # RAG results
    similar_resumes = Column(JSON, nullable=True)
    context_embeddings = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class JobDescription(Base):
    """Job description model for storing job postings."""
    
    __tablename__ = "job_descriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    raw_text = Column(Text, nullable=False)
    
    # Extracted information
    required_skills = Column(JSON, nullable=True)
    preferred_skills = Column(JSON, nullable=True)
    experience_level = Column(String(50), nullable=True)
    salary_range = Column(String(100), nullable=True)
    
    # Embeddings for RAG
    embeddings = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatSession(Base):
    """Chat session model for storing AI assistant conversations."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    resume_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Session metadata
    session_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(Base):
    """Chat message model for storing individual chat messages."""
    
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    
    # Metadata
    tokens_used = Column(Integer, nullable=True)
    model_used = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Redis dependency
def get_redis() -> redis.Redis:
    """Get Redis client dependency."""
    return redis_client


# Database initialization
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def create_tables():
    """Create all database tables."""
    init_db()


def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)


# Utility functions
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email address."""
    return db.query(User).filter(User.email == email).first()


def get_resume_by_id(db: Session, resume_id: str) -> Optional[Resume]:
    """Get resume by ID."""
    return db.query(Resume).filter(Resume.id == resume_id).first()


def get_analysis_by_resume_id(db: Session, resume_id: str) -> Optional[Analysis]:
    """Get analysis by resume ID."""
    return db.query(Analysis).filter(Analysis.resume_id == resume_id).first()


def get_chat_session_by_id(db: Session, session_id: str) -> Optional[ChatSession]:
    """Get chat session by ID."""
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def get_chat_messages_by_session(db: Session, session_id: str) -> list[ChatMessage]:
    """Get chat messages for a session."""
    return db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()
