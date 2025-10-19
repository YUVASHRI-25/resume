"""
RAG (Retrieval-Augmented Generation) API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import logging

from app.database import get_db, Resume, Analysis
from core.rag_service import RAGService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


@router.post("/similar-resumes")
async def find_similar_resumes(
    resume_id: str = Form(...),
    job_role: Optional[str] = Form(None),
    limit: int = Form(5),
    db: Session = Depends(get_db)
):
    """Find similar resumes using RAG system."""
    try:
        # Get resume
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.raw_text:
            raise HTTPException(status_code=400, detail="No text content found in resume")
        
        # Find similar resumes
        similar_resumes = rag_service.get_similar_resumes(
            resume.raw_text, job_role, limit
        )
        
        return {
            "resume_id": resume_id,
            "job_role": job_role,
            "similar_resumes": similar_resumes,
            "service_status": rag_service.get_service_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error finding similar resumes: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/context")
async def get_contextual_recommendations(
    resume_id: str = Form(...),
    job_role: str = Form(...),
    db: Session = Depends(get_db)
):
    """Get contextual recommendations based on similar resumes."""
    try:
        # Get resume
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.raw_text:
            raise HTTPException(status_code=400, detail="No text content found in resume")
        
        # Get contextual recommendations
        recommendations = rag_service.get_contextual_recommendations(
            resume.raw_text, job_role
        )
        
        return {
            "resume_id": resume_id,
            "job_role": job_role,
            "recommendations": recommendations,
            "service_status": rag_service.get_service_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting contextual recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@router.post("/index-resume")
async def index_resume_for_rag(
    resume_id: str = Form(...),
    job_role: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Add resume to RAG index for similarity search."""
    try:
        # Get resume
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.raw_text:
            raise HTTPException(status_code=400, detail="No text content found in resume")
        
        # Prepare metadata
        metadata = {
            "user_id": str(resume.user_id),
            "filename": resume.filename,
            "file_type": resume.file_type,
            "created_at": resume.created_at.isoformat()
        }
        
        if job_role:
            metadata["job_role"] = job_role
        
        # Add to RAG index
        success = rag_service.add_resume_to_index(
            str(resume_id), resume.raw_text, metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to index resume")
        
        return {
            "resume_id": resume_id,
            "job_role": job_role,
            "indexed": True,
            "service_status": rag_service.get_service_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error indexing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.delete("/index/{resume_id}")
async def remove_resume_from_index(
    resume_id: str,
    db: Session = Depends(get_db)
):
    """Remove resume from RAG index."""
    try:
        # Check if resume exists
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Remove from RAG index
        success = rag_service.remove_resume_from_index(resume_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove resume from index")
        
        return {
            "resume_id": resume_id,
            "removed": True,
            "service_status": rag_service.get_service_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error removing resume from index: {e}")
        raise HTTPException(status_code=500, detail=f"Removal failed: {str(e)}")


@router.get("/stats")
async def get_rag_stats():
    """Get RAG system statistics."""
    try:
        stats = rag_service.get_collection_stats()
        return {
            "collection_stats": stats,
            "service_status": rag_service.get_service_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/bulk-index")
async def bulk_index_resumes(
    resume_ids: str = Form(...),
    job_role: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Bulk index multiple resumes."""
    try:
        # Parse resume IDs
        resume_id_list = [rid.strip() for rid in resume_ids.split(',') if rid.strip()]
        
        if not resume_id_list:
            raise HTTPException(status_code=400, detail="No resume IDs provided")
        
        # Get resumes
        resumes = db.query(Resume).filter(Resume.id.in_(resume_id_list)).all()
        
        if len(resumes) != len(resume_id_list):
            raise HTTPException(status_code=404, detail="Some resumes not found")
        
        # Prepare data for bulk indexing
        resumes_data = []
        for resume in resumes:
            if resume.raw_text:
                metadata = {
                    "user_id": str(resume.user_id),
                    "filename": resume.filename,
                    "file_type": resume.file_type,
                    "created_at": resume.created_at.isoformat()
                }
                
                if job_role:
                    metadata["job_role"] = job_role
                
                resumes_data.append({
                    "id": str(resume.id),
                    "text": resume.raw_text,
                    "metadata": metadata
                })
        
        # Perform bulk indexing
        results = rag_service.bulk_index_resumes(resumes_data)
        
        return {
            "bulk_indexing_results": results,
            "service_status": rag_service.get_service_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in bulk indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk indexing failed: {str(e)}")


@router.get("/status")
async def get_rag_status():
    """Get RAG service status and configuration."""
    try:
        status = rag_service.get_service_status()
        return {
            "status": "operational" if status["initialized"] else "not_initialized",
            "configuration": status,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
