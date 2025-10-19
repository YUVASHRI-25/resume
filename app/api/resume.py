"""
Resume analysis API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import uuid
import logging

from app.database import get_db, Resume, Analysis, User
from core.analyzer import ResumeAnalyzer
from services.file_processor import FileProcessor
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
analyzer = ResumeAnalyzer()
file_processor = FileProcessor()


@router.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user_email: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload and process a resume file."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        if file.size > settings.storage.max_file_size:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {settings.storage.max_file_size} bytes"
            )
        
        # Check file extension
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.storage.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.storage.allowed_extensions)}"
            )
        
        # Get or create user
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            user = User(email=user_email)
            db.add(user)
            db.commit()
            db.refresh(user)
        
        # Process file
        file_path = await file_processor.save_uploaded_file(file)
        
        # Extract text
        extracted_text = await file_processor.extract_text(file_path, file_extension)
        
        # Create resume record
        resume = Resume(
            user_id=user.id,
            filename=file.filename,
            file_path=file_path,
            file_type=file_extension,
            file_size=file.size,
            raw_text=extracted_text
        )
        
        db.add(resume)
        db.commit()
        db.refresh(resume)
        
        return {
            "resume_id": str(resume.id),
            "filename": resume.filename,
            "status": "uploaded",
            "message": "Resume uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading resume: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/analyze/{resume_id}")
async def analyze_resume(
    resume_id: str,
    job_role: str = Form(...),
    db: Session = Depends(get_db)
):
    """Analyze a resume for a specific job role."""
    try:
        # Get resume
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.raw_text:
            raise HTTPException(status_code=400, detail="No text content found in resume")
        
        # Perform analysis
        analysis_result = analyzer.analyze_resume(resume.raw_text, job_role)
        
        # Update resume with analysis results
        resume.sections = analysis_result.get("sections")
        resume.ats_score = analysis_result.get("ats_score")
        resume.overall_score = analysis_result.get("overall_score")
        resume.word_count = analysis_result.get("word_count")
        resume.section_count = analysis_result.get("section_count")
        resume.technical_skills = analysis_result.get("technical_skills")
        resume.soft_skills = analysis_result.get("soft_skills")
        
        # Create detailed analysis record
        analysis = Analysis(
            resume_id=resume.id,
            job_role=job_role,
            ats_score=analysis_result.get("ats_score", 0),
            role_match_percentage=analysis_result.get("role_match_percentage", 0),
            overall_score=analysis_result.get("overall_score", 0),
            found_keywords=analysis_result.get("found_keywords"),
            missing_keywords=analysis_result.get("missing_keywords"),
            grammar_issues=analysis_result.get("grammar_issues"),
            recommendations=analysis_result.get("recommendations")
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        return {
            "analysis_id": str(analysis.id),
            "resume_id": resume_id,
            "job_role": job_role,
            "results": analysis_result,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{resume_id}")
async def get_resume(
    resume_id: str,
    db: Session = Depends(get_db)
):
    """Get resume information and analysis results."""
    try:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get latest analysis
        analysis = db.query(Analysis).filter(Analysis.resume_id == resume_id).order_by(Analysis.created_at.desc()).first()
        
        return {
            "resume": {
                "id": str(resume.id),
                "filename": resume.filename,
                "file_type": resume.file_type,
                "file_size": resume.file_size,
                "created_at": resume.created_at.isoformat(),
                "analyzed_at": resume.analyzed_at.isoformat() if resume.analyzed_at else None
            },
            "analysis": {
                "ats_score": resume.ats_score,
                "overall_score": resume.overall_score,
                "word_count": resume.word_count,
                "section_count": resume.section_count,
                "technical_skills": resume.technical_skills,
                "soft_skills": resume.soft_skills,
                "sections": resume.sections
            } if resume.ats_score else None,
            "detailed_analysis": {
                "job_role": analysis.job_role,
                "role_match_percentage": analysis.role_match_percentage,
                "found_keywords": analysis.found_keywords,
                "missing_keywords": analysis.missing_keywords,
                "grammar_issues": analysis.grammar_issues,
                "recommendations": analysis.recommendations,
                "ai_recommendations": analysis.ai_recommendations,
                "created_at": analysis.created_at.isoformat()
            } if analysis else None
        }
        
    except Exception as e:
        logger.error(f"Error getting resume: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resume: {str(e)}")


@router.post("/compare")
async def compare_resume_with_job(
    resume_id: str = Form(...),
    job_description: str = Form(...),
    job_title: str = Form(...),
    db: Session = Depends(get_db)
):
    """Compare resume with a job description."""
    try:
        # Get resume
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.raw_text:
            raise HTTPException(status_code=400, detail="No text content found in resume")
        
        # Perform comparison analysis
        comparison_result = analyzer.compare_with_job_description(
            resume.raw_text, 
            job_description, 
            job_title
        )
        
        return {
            "resume_id": resume_id,
            "job_title": job_title,
            "comparison_results": comparison_result,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error comparing resume with job: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/user/{user_email}")
async def get_user_resumes(
    user_email: str,
    db: Session = Depends(get_db)
):
    """Get all resumes for a user."""
    try:
        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        resumes = db.query(Resume).filter(Resume.user_id == user.id).order_by(Resume.created_at.desc()).all()
        
        resume_list = []
        for resume in resumes:
            resume_data = {
                "id": str(resume.id),
                "filename": resume.filename,
                "file_type": resume.file_type,
                "file_size": resume.file_size,
                "created_at": resume.created_at.isoformat(),
                "analyzed_at": resume.analyzed_at.isoformat() if resume.analyzed_at else None,
                "ats_score": resume.ats_score,
                "overall_score": resume.overall_score
            }
            resume_list.append(resume_data)
        
        return {
            "user_email": user_email,
            "resumes": resume_list,
            "total_count": len(resume_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting user resumes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resumes: {str(e)}")


@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: str,
    db: Session = Depends(get_db)
):
    """Delete a resume and its associated data."""
    try:
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Delete associated analyses
        db.query(Analysis).filter(Analysis.resume_id == resume_id).delete()
        
        # Delete resume
        db.delete(resume)
        db.commit()
        
        # Delete file from storage
        await file_processor.delete_file(resume.file_path)
        
        return {
            "resume_id": resume_id,
            "status": "deleted",
            "message": "Resume and associated data deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting resume: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
