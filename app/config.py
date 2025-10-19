"""
Configuration management for Resume Insight application.
Handles environment variables, settings, and application configuration.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(default="postgresql://username:password@localhost:5432/resume_insight", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    class Config:
        env_prefix = "DB_"


class LLMSettings(BaseSettings):
    """LLM API configuration settings."""
    
    mistral_api_key: Optional[str] = Field(default=None, env="MISTRAL_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Model configurations
    mistral_model: str = Field(default="mistral-7b-instruct", env="MISTRAL_MODEL")
    openai_model: str = Field(default="gpt-3.5-turbo", env="OPENAI_MODEL")
    
    # API settings
    max_tokens: int = Field(default=1000, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    class Config:
        env_prefix = "LLM_"


class StorageSettings(BaseSettings):
    """File storage configuration settings."""
    
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket_name: Optional[str] = Field(default=None, env="S3_BUCKET_NAME")
    
    # Local storage settings
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=20971520, env="MAX_FILE_SIZE")  # 20MB
    allowed_extensions: List[str] = Field(default=["pdf", "docx", "txt", "png", "jpg", "jpeg"], env="ALLOWED_EXTENSIONS")
    
    class Config:
        env_prefix = "STORAGE_"


class RAGSettings(BaseSettings):
    """RAG system configuration settings."""
    
    chroma_persist_directory: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Vector search settings
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_results: int = Field(default=5, env="MAX_RAG_RESULTS")
    
    class Config:
        env_prefix = "RAG_"


class APISettings(BaseSettings):
    """API server configuration settings."""
    
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(default=["http://localhost:8501", "http://127.0.0.1:8501"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_prefix = "API_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your-jwt-secret-key", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        env_prefix = "SECURITY_"


class AppSettings(BaseSettings):
    """Main application configuration settings."""
    
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    app_name: str = Field(default="Resume Insight", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    
    # Database settings
    database: DatabaseSettings = DatabaseSettings()
    
    # LLM settings
    llm: LLMSettings = LLMSettings()
    
    # Storage settings
    storage: StorageSettings = StorageSettings()
    
    # RAG settings
    rag: RAGSettings = RAGSettings()
    
    # API settings
    api: APISettings = APISettings()
    
    # Security settings
    security: SecuritySettings = SecuritySettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = AppSettings()


def get_settings() -> AppSettings:
    """Get the application settings instance."""
    return settings


def validate_settings() -> bool:
    """Validate that all required settings are configured."""
    errors = []
    
    # Check required API keys
    if not settings.llm.mistral_api_key and not settings.llm.openai_api_key:
        errors.append("At least one LLM API key (Mistral or OpenAI) must be configured")
    
    # Check database URL
    if not settings.database.url or "postgresql://username:password" in settings.database.url:
        errors.append("Valid DATABASE_URL must be configured")
    
    # Check S3 settings if using cloud storage
    if settings.storage.s3_bucket_name:
        if not settings.storage.aws_access_key_id or not settings.storage.aws_secret_access_key:
            errors.append("AWS credentials must be configured when using S3 storage")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


# Job role keywords configuration
JOB_KEYWORDS = {
    "Data Scientist": [
        "python", "machine learning", "statistics", "pandas", "numpy", "scikit-learn",
        "tensorflow", "pytorch", "sql", "data analysis", "visualization", "jupyter", 
        "r", "statistics", "deep learning", "neural networks", "regression", 
        "classification", "clustering", "feature engineering", "model deployment"
    ],
    "Software Engineer": [
        "programming", "java", "python", "javascript", "react", "node.js", "database",
        "git", "agile", "testing", "debugging", "api", "frontend", "backend", 
        "algorithms", "data structures", "microservices", "docker", "kubernetes",
        "rest api", "graphql", "version control", "ci/cd"
    ],
    "Product Manager": [
        "product", "strategy", "roadmap", "stakeholder", "analytics", "user experience",
        "market research", "agile", "scrum", "requirements", "metrics", "wireframes", 
        "user stories", "product development", "customer research", "competitive analysis",
        "go-to-market", "product launch"
    ],
    "Marketing Manager": [
        "marketing", "digital marketing", "seo", "social media", "analytics", "campaigns",
        "brand", "content", "advertising", "growth", "conversion", "roi", "crm",
        "email marketing", "ppc", "content marketing", "brand management", "lead generation"
    ],
    "Data Analyst": [
        "sql", "excel", "python", "tableau", "power bi", "statistics", "reporting",
        "data visualization", "business intelligence", "analytics", "dashboards", "kpi",
        "data mining", "etl", "data modeling", "statistical analysis", "forecasting"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "ci/cd", "terraform",
        "ansible", "monitoring", "linux", "bash", "infrastructure", "deployment",
        "cloud computing", "automation", "configuration management", "security"
    ],
    "UI/UX Designer": [
        "figma", "sketch", "adobe xd", "prototyping", "wireframes", "user research",
        "usability testing", "design systems", "responsive design", "accessibility", 
        "typography", "visual design", "interaction design", "user interface", "user experience"
    ],
    "Cybersecurity Analyst": [
        "security", "penetration testing", "vulnerability assessment", "siem", "firewall",
        "incident response", "compliance", "risk assessment", "cryptography", "network security",
        "security monitoring", "threat analysis", "security policies", "audit"
    ],
    "Business Analyst": [
        "requirements gathering", "process improvement", "stakeholder management", 
        "documentation", "business process", "gap analysis", "user stories", "workflow",
        "project management", "data analysis", "business intelligence", "process mapping"
    ],
    "Full Stack Developer": [
        "html", "css", "javascript", "react", "angular", "vue", "node.js", "express",
        "mongodb", "postgresql", "rest api", "graphql", "version control", "responsive design",
        "frontend", "backend", "database", "api development", "web development"
    ],
    "Machine Learning Engineer": [
        "python", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "machine learning",
        "deep learning", "neural networks", "computer vision", "nlp", "data science", 
        "algorithms", "statistics", "linear algebra", "calculus", "regression", "classification",
        "clustering", "feature engineering", "model deployment", "mlops", "docker", "kubernetes"
    ],
    "AI Engineer": [
        "artificial intelligence", "machine learning", "deep learning", "neural networks", 
        "python", "tensorflow", "pytorch", "computer vision", "nlp", "natural language processing",
        "opencv", "transformers", "bert", "gpt", "reinforcement learning", "generative ai", 
        "llm", "chatbot", "model optimization", "ai ethics", "edge ai", "quantization"
    ]
}

# Technical and soft skills databases
TECHNICAL_SKILLS = [
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust", "swift",
    "sql", "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask",
    "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "jenkins", "ci/cd", "mongodb", 
    "postgresql", "redis", "elasticsearch", "spark", "hadoop", "tableau", "power bi", 
    "excel", "figma", "sketch", "linux", "bash", "terraform", "ansible", "selenium", 
    "junit", "jira", "confluence", "github", "gitlab", "bitbucket"
]

SOFT_SKILLS = [
    "leadership", "communication", "teamwork", "problem solving", "critical thinking",
    "project management", "time management", "adaptability", "creativity", "analytical",
    "collaboration", "innovation", "strategic thinking", "customer service", "negotiation",
    "presentation", "mentoring", "conflict resolution", "decision making", "emotional intelligence",
    "interpersonal skills", "active listening", "empathy", "flexibility", "resilience"
]
