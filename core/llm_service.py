"""
LLM Service for integrating with Mistral 7B and other language models.
"""

import logging
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for integrating with various LLM providers.
    Supports Mistral 7B, OpenAI GPT models, and fallback responses.
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self.mistral_client = None
        self.openai_client = None
        self.is_initialized = False
        
        # Model configurations
        self.mistral_model = settings.llm.mistral_model
        self.openai_model = settings.llm.openai_model
        self.max_tokens = settings.llm.max_tokens
        self.temperature = settings.llm.temperature
        
        # Fallback responses for common scenarios
        self.fallback_responses = {
            "experience": "Focus on quantifiable achievements, use action verbs, and tailor content to job requirements.",
            "skills": "Organize skills by category, match job requirements, and provide proficiency levels.",
            "ats": "Use standard headings, include keywords naturally, avoid complex formatting, and save as PDF.",
            "keywords": "Study job descriptions, use industry terms, include both acronyms and full terms.",
            "format": "Use clear headings, consistent bullet points, readable fonts, and professional layout."
        }
    
    def initialize(self):
        """Initialize LLM clients."""
        try:
            # Initialize Mistral client
            if MISTRAL_AVAILABLE and settings.llm.mistral_api_key:
                self.mistral_client = Mistral(api_key=settings.llm.mistral_api_key)
                logger.info("Mistral client initialized successfully")
            
            # Initialize OpenAI client
            if OPENAI_AVAILABLE and settings.llm.openai_api_key:
                openai.api_key = settings.llm.openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            self.is_initialized = False
    
    def generate_response(self, prompt: str, context: str = "", 
                        max_tokens: Optional[int] = None) -> str:
        """
        Generate a response using available LLM providers.
        
        Args:
            prompt: The input prompt
            context: Additional context for the prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if not self.is_initialized:
            return self._get_fallback_response(prompt)
        
        # Try Mistral first
        if self.mistral_client:
            try:
                response = self._generate_mistral_response(prompt, context, max_tokens)
                if response:
                    return response
            except Exception as e:
                logger.error(f"Mistral generation failed: {e}")
        
        # Try OpenAI as fallback
        if self.openai_client:
            try:
                response = self._generate_openai_response(prompt, context, max_tokens)
                if response:
                    return response
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
        
        # Return fallback response
        return self._get_fallback_response(prompt)
    
    def _generate_mistral_response(self, prompt: str, context: str = "", 
                                 max_tokens: Optional[int] = None) -> str:
        """Generate response using Mistral API."""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            response = self.mistral_client.chat.completions.create(
                model=self.mistral_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert resume consultant and career advisor. Provide specific, actionable advice for resume improvement."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return ""
    
    def _generate_openai_response(self, prompt: str, context: str = "", 
                                max_tokens: Optional[int] = None) -> str:
        """Generate response using OpenAI API."""
        try:
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            response = self.openai_client.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert resume consultant and career advisor. Provide specific, actionable advice for resume improvement."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""
    
    def get_resume_recommendations(self, resume_text: str, job_role: str, 
                                 analysis_results: Dict[str, Any]) -> str:
        """
        Get personalized resume recommendations based on analysis.
        
        Args:
            resume_text: Resume content
            job_role: Target job role
            analysis_results: Results from resume analysis
            
        Returns:
            Personalized recommendations
        """
        try:
            prompt = self._create_recommendation_prompt(resume_text, job_role, analysis_results)
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_fallback_response("recommendations")
    
    def get_skill_suggestions(self, job_role: str, current_skills: List[str]) -> str:
        """
        Get skill suggestions for a specific job role.
        
        Args:
            job_role: Target job role
            current_skills: Current skills from resume
            
        Returns:
            Skill suggestions
        """
        try:
            prompt = f"""
            For a {job_role} position, suggest additional skills that would strengthen this resume.
            
            Current skills: {', '.join(current_skills[:20])}
            
            Provide 5-7 specific skills that are highly relevant to {job_role} roles and would improve the candidate's competitiveness.
            Focus on both technical and soft skills.
            """
            
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generating skill suggestions: {e}")
            return self._get_fallback_response("skills")
    
    def get_ats_optimization_tips(self, ats_score: float, issues: List[str]) -> str:
        """
        Get ATS optimization tips based on current score and issues.
        
        Args:
            ats_score: Current ATS score
            issues: List of identified issues
            
        Returns:
            ATS optimization tips
        """
        try:
            prompt = f"""
            This resume has an ATS score of {ats_score}/100.
            
            Identified issues: {', '.join(issues[:10])}
            
            Provide specific, actionable tips to improve ATS compatibility.
            Focus on formatting, keywords, and structure improvements.
            """
            
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generating ATS tips: {e}")
            return self._get_fallback_response("ats")
    
    def chat_with_resume_context(self, user_message: str, resume_context: str = "") -> str:
        """
        Chat with AI assistant about resume improvement.
        
        Args:
            user_message: User's question or request
            resume_context: Resume content for context
            
        Returns:
            AI response
        """
        try:
            # Check for common patterns first
            fallback_response = self._check_common_patterns(user_message)
            if fallback_response:
                return fallback_response
            
            # Create contextual prompt
            context = f"Resume context: {resume_context[:1000]}..." if resume_context else ""
            prompt = f"User question: {user_message}\n\nProvide helpful resume advice."
            
            return self.generate_response(prompt, context)
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I'm here to help with your resume questions. Could you please rephrase your question?"
    
    def _create_recommendation_prompt(self, resume_text: str, job_role: str, 
                                    analysis_results: Dict[str, Any]) -> str:
        """Create a detailed prompt for resume recommendations."""
        ats_score = analysis_results.get("ats_score", 0)
        match_percentage = analysis_results.get("role_match_percentage", 0)
        found_keywords = analysis_results.get("found_keywords", [])
        missing_keywords = analysis_results.get("missing_keywords", [])
        
        prompt = f"""
        Analyze this resume for a {job_role} position and provide specific improvement recommendations.
        
        Resume Summary:
        - ATS Score: {ats_score}/100
        - Role Match: {match_percentage:.1f}%
        - Found Keywords: {', '.join(found_keywords[:10])}
        - Missing Keywords: {', '.join(missing_keywords[:10])}
        
        Resume Content (first 2000 characters):
        {resume_text[:2000]}...
        
        Provide 5-7 specific, actionable recommendations to improve this resume for the {job_role} role.
        Focus on:
        1. Content improvements
        2. Keyword optimization
        3. Formatting enhancements
        4. Skills alignment
        5. ATS optimization
        
        Make each recommendation specific and actionable.
        """
        
        return prompt
    
    def _check_common_patterns(self, user_message: str) -> Optional[str]:
        """Check for common patterns and return appropriate responses."""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['experience', 'work history', 'job history']):
            return self.fallback_responses["experience"]
        elif any(word in message_lower for word in ['skills', 'technical skills', 'abilities']):
            return self.fallback_responses["skills"]
        elif any(word in message_lower for word in ['ats', 'applicant tracking', 'ats-friendly']):
            return self.fallback_responses["ats"]
        elif any(word in message_lower for word in ['keywords', 'keyword', 'terms']):
            return self.fallback_responses["keywords"]
        elif any(word in message_lower for word in ['format', 'formatting', 'layout', 'design']):
            return self.fallback_responses["format"]
        
        return None
    
    def _get_fallback_response(self, prompt_type: str) -> str:
        """Get fallback response when LLM services are unavailable."""
        if prompt_type in self.fallback_responses:
            return self.fallback_responses[prompt_type]
        
        return """
        I'm here to help improve your resume! Here are some general tips:
        
        1. Use clear, professional formatting with standard headings
        2. Include relevant keywords from job descriptions
        3. Quantify achievements with specific numbers
        4. Use bullet points for easy scanning
        5. Keep it concise (1-2 pages)
        6. Proofread carefully for errors
        7. Tailor content to each job application
        
        For more specific advice, please describe what aspect of your resume you'd like to improve.
        """
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of available LLM models."""
        return {
            "mistral_available": self.mistral_client is not None,
            "openai_available": self.openai_client is not None,
            "initialized": self.is_initialized,
            "mistral_model": self.mistral_model,
            "openai_model": self.openai_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
