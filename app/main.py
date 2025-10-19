"""
Main FastAPI application for Resume Insight.
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
import logging

from app.config import settings, validate_settings
from app.database import get_db, init_db
from app.api.resume import router as resume_router
from app.api.llm import router as llm_router
from app.api.rag import router as rag_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered resume analysis and optimization system",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(resume_router, prefix="/api/resume", tags=["resume"])
app.include_router(llm_router, prefix="/api/llm", tags=["llm"])
app.include_router(rag_router, prefix="/api/rag", tags=["rag"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Resume Insight application...")
    
    # Validate settings
    if not validate_settings():
        logger.error("Invalid configuration. Please check your environment variables.")
        raise RuntimeError("Invalid configuration")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    logger.info("Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Resume Insight application...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Resume Insight API",
        "version": settings.version,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": settings.version
    }


@app.get("/api/status")
async def api_status(db: Session = Depends(get_db)):
    """API status with database connectivity check."""
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_status = "disconnected"
    
    return {
        "api": "running",
        "database": db_status,
        "version": settings.version,
        "debug": settings.debug
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
