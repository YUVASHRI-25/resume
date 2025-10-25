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
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # Null for Google Auth users
    name = Column(String(255), nullable=True)
    role = Column(String(20), nullable=False, default='user')  # 'user' or 'admin'
    auth_type = Column(String(20), nullable=False, default='email')  # 'email' or 'google'
    is_verified = Column(Boolean, default=False)
    google_id = Column(String(255), nullable=True, unique=True)
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


# Verification token model
class VerificationToken(Base):
    """Verification token model for email verification."""
    
    __tablename__ = "verification_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    token = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)


# Authentication and verification functions
def create_user(db: Session, email: str, username: str, password: str = None, auth_type: str = 'email') -> User:
    """Create a new user."""
    from werkzeug.security import generate_password_hash
    
    user = User(
        email=email,
        username=username,
        password_hash=generate_password_hash(password) if password else None,
        auth_type=auth_type,
        is_verified=auth_type == 'google'  # Google users are pre-verified
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user with username and password."""
    from werkzeug.security import check_password_hash
    
    user = db.query(User).filter(
        User.username == username,
        User.auth_type == 'email',
        User.role == 'user'
    ).first()
    
    if user and check_password_hash(user.password_hash, password):
        return user
    return None


def authenticate_admin(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate an admin user."""
    from werkzeug.security import check_password_hash
    
    admin = db.query(User).filter(
        User.username == username,
        User.role == 'admin'
    ).first()
    
    if admin and check_password_hash(admin.password_hash, password):
        return admin
    return None


def create_verification_token(db: Session, user_id: uuid.UUID) -> str:
    """Create and store a verification token for a user."""
    import secrets
    from datetime import timedelta
    
    token = secrets.token_urlsafe(32)
    verification_token = VerificationToken(
        user_id=user_id,
        token=token,
        expires_at=datetime.utcnow() + timedelta(hours=24)
    )
    
    db.add(verification_token)
    db.commit()
    return token


def verify_email(db: Session, token: str) -> bool:
    """Verify a user's email using a verification token."""
    verification = db.query(VerificationToken).filter(
        VerificationToken.token == token,
        VerificationToken.expires_at > datetime.utcnow()
    ).first()
    
    if verification:
        user = db.query(User).filter(User.id == verification.user_id).first()
        if user:
            user.is_verified = True
            db.commit()
            return True
    return False


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
