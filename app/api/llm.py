"""
LLM integration API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import logging

from app.database import get_db, Resume, Analysis
from core.llm_service import LLMService
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize LLM service
llm_service = LLMService()


@router.post("/recommendations")
async def get_ai_recommendations(
    resume_id: str = Form(...),
    job_role: str = Form(...),
    db: Session = Depends(get_db)
):
    """Get AI-powered recommendations for resume improvement."""
    try:
        # Get resume
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.raw_text:
            raise HTTPException(status_code=400, detail="No text content found in resume")
        
        # Get analysis results
        analysis = db.query(Analysis).filter(Analysis.resume_id == resume_id).first()
        
        # Prepare analysis data
        analysis_data = {
            "ats_score": analysis.ats_score if analysis else 0,
            "role_match_percentage": analysis.role_match_percentage if analysis else 0,
            "found_keywords": analysis.found_keywords if analysis else [],
            "missing_keywords": analysis.missing_keywords if analysis else []
        }
        
        # Get AI recommendations
        recommendations = llm_service.get_resume_recommendations(
            resume.raw_text, job_role, analysis_data
        )
        
        # Update analysis record with AI recommendations
        if analysis:
            analysis.ai_recommendations = recommendations
            db.commit()
        
        return {
            "resume_id": resume_id,
            "job_role": job_role,
            "ai_recommendations": recommendations,
            "model_status": llm_service.get_model_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting AI recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.post("/chat")
async def chat_with_ai(
    message: str = Form(...),
    resume_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Chat with AI assistant about resume improvement."""
    try:
        # Get resume context if provided
        resume_context = ""
        if resume_id:
            resume = db.query(Resume).filter(Resume.id == resume_id).first()
            if resume and resume.raw_text:
                resume_context = resume.raw_text[:1000]  # Limit context length
        
        # Generate AI response
        response = llm_service.chat_with_resume_context(message, resume_context)
        
        return {
            "user_message": message,
            "ai_response": response,
            "resume_id": resume_id,
            "session_id": session_id,
            "model_status": llm_service.get_model_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/skill-suggestions")
async def get_skill_suggestions(
    job_role: str = Form(...),
    current_skills: str = Form(...),
    db: Session = Depends(get_db)
):
    """Get skill suggestions for a specific job role."""
    try:
        # Parse current skills
        skills_list = [skill.strip() for skill in current_skills.split(',') if skill.strip()]
        
        # Get skill suggestions
        suggestions = llm_service.get_skill_suggestions(job_role, skills_list)
        
        return {
            "job_role": job_role,
            "current_skills": skills_list,
            "suggestions": suggestions,
            "model_status": llm_service.get_model_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting skill suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.post("/ats-optimization")
async def get_ats_optimization_tips(
    resume_id: str = Form(...),
    db: Session = Depends(get_db)
):
    """Get ATS optimization tips for a resume."""
    try:
        # Get resume and analysis
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        analysis = db.query(Analysis).filter(Analysis.resume_id == resume_id).first()
        
        # Get ATS score and issues
        ats_score = analysis.ats_score if analysis else 0
        issues = analysis.grammar_issues if analysis else []
        
        # Get optimization tips
        tips = llm_service.get_ats_optimization_tips(ats_score, issues)
        
        return {
            "resume_id": resume_id,
            "ats_score": ats_score,
            "optimization_tips": tips,
            "model_status": llm_service.get_model_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting ATS optimization tips: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tips: {str(e)}")


@router.get("/status")
async def get_llm_status():
    """Get LLM service status and configuration."""
    try:
        status = llm_service.get_model_status()
        return {
            "status": "operational" if status["initialized"] else "not_initialized",
            "configuration": status,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/generate-content")
async def generate_resume_content(
    content_type: str = Form(...),
    job_role: str = Form(...),
    context: str = Form(...),
    db: Session = Depends(get_db)
):
    """Generate resume content using AI."""
    try:
        # Validate content type
        valid_types = ["summary", "experience", "skills", "achievements"]
        if content_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid content type. Allowed: {', '.join(valid_types)}"
            )
        
        # Create prompt based on content type
        prompt = f"""
        Generate a {content_type} section for a resume targeting a {job_role} position.
        
        Context: {context}
        
        Requirements:
        - Professional and compelling tone
        - Specific and quantifiable where possible
        - Relevant to {job_role} role
        - Concise but impactful
        - Use action verbs
        """
        
        # Generate content
        generated_content = llm_service.generate_response(prompt)
        
        return {
            "content_type": content_type,
            "job_role": job_role,
            "generated_content": generated_content,
            "model_status": llm_service.get_model_status(),
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")
