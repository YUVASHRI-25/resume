"""
Core ResumeAnalyzer class with LLM integration for comprehensive resume analysis.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings, JOB_KEYWORDS, TECHNICAL_SKILLS, SOFT_SKILLS
from core.llm_service import LLMService
from core.rag_service import RAGService
from services.nlp_processor import NLPProcessor
from services.file_processor import FileProcessor

logger = logging.getLogger(__name__)


class ResumeAnalyzer:
    """
    Main resume analysis class that orchestrates all analysis components.
    Integrates LLM, RAG, NLP processing, and scoring algorithms.
    """
    
    def __init__(self):
        """Initialize the ResumeAnalyzer with all required services."""
        self.nlp_processor = NLPProcessor()
        self.llm_service = LLMService()
        self.rag_service = RAGService()
        self.file_processor = FileProcessor()
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all analysis services."""
        try:
            self.nlp_processor.initialize()
            logger.info("NLP processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NLP processor: {e}")
        
        try:
            self.llm_service.initialize()
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
        
        try:
            self.rag_service.initialize()
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
    
    def analyze_resume(self, resume_text: str, job_role: str) -> Dict[str, Any]:
        """
        Perform comprehensive resume analysis for a specific job role.
        
        Args:
            resume_text: Raw text content of the resume
            job_role: Target job role for analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            logger.info(f"Starting comprehensive analysis for job role: {job_role}")
            
            # Basic text processing
            cleaned_text = self.nlp_processor.clean_text(resume_text)
            
            # Extract sections
            sections = self._extract_sections(cleaned_text)
            
            # Extract skills
            technical_skills, soft_skills = self._extract_skills(cleaned_text)
            
            # Keyword matching
            found_keywords, missing_keywords, match_percentage = self._analyze_keywords(
                cleaned_text, job_role
            )
            
            # Calculate ATS score
            ats_score = self._calculate_ats_score(cleaned_text, sections)
            
            # Grammar and language analysis
            grammar_issues = self._analyze_grammar(cleaned_text)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                ats_score, match_percentage, len(technical_skills), len(soft_skills)
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                ats_score, match_percentage, missing_keywords, grammar_issues, sections
            )
            
            # Get AI-powered recommendations using LLM
            ai_recommendations = self._get_ai_recommendations(
                resume_text, job_role, found_keywords, missing_keywords
            )
            
            # Get RAG-based suggestions
            rag_suggestions = self._get_rag_suggestions(resume_text, job_role)
            
            # Compile results
            results = {
                "sections": sections,
                "technical_skills": technical_skills,
                "soft_skills": soft_skills,
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords,
                "role_match_percentage": match_percentage,
                "ats_score": ats_score,
                "overall_score": overall_score,
                "word_count": len(cleaned_text.split()),
                "section_count": len([s for s in sections.values() if s]),
                "grammar_issues": grammar_issues,
                "recommendations": recommendations,
                "ai_recommendations": ai_recommendations,
                "rag_suggestions": rag_suggestions,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Analysis completed successfully. Overall score: {overall_score:.1f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in resume analysis: {e}")
            raise
    
    def compare_with_job_description(self, resume_text: str, job_description: str, job_title: str) -> Dict[str, Any]:
        """
        Compare resume with a specific job description.
        
        Args:
            resume_text: Resume content
            job_description: Job description text
            job_title: Job title
            
        Returns:
            Comparison analysis results
        """
        try:
            logger.info(f"Comparing resume with job: {job_title}")
            
            # Extract keywords from job description
            job_keywords = self.nlp_processor.extract_keywords(job_description)
            
            # Calculate similarity
            similarity_score = self._calculate_text_similarity(resume_text, job_description)
            
            # Match skills
            resume_skills = self._extract_skills(resume_text)[0] + self._extract_skills(resume_text)[1]
            matched_skills = [skill for skill in resume_skills if skill.lower() in job_description.lower()]
            
            # Generate comparison recommendations
            comparison_recommendations = self._generate_comparison_recommendations(
                resume_text, job_description, matched_skills, job_keywords
            )
            
            return {
                "similarity_score": similarity_score,
                "job_keywords": job_keywords,
                "matched_skills": matched_skills,
                "missing_skills": [kw for kw in job_keywords if kw not in matched_skills],
                "recommendations": comparison_recommendations,
                "comparison_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in job comparison: {e}")
            raise
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from resume text."""
        sections = {}
        
        section_patterns = {
            'summary': r'(summary|objective|profile|about|overview)',
            'experience': r'(experience|employment|work|career|professional|job|position)',
            'education': r'(education|academic|qualification|degree|university|college)',
            'skills': r'(skills|technical|competencies|expertise|abilities|technologies)',
            'projects': r'(projects|portfolio|work samples|personal projects)',
            'certifications': r'(certifications?|certificates?|licensed?|credentials)',
            'achievements': r'(achievements|awards|honors|recognition)'
        }
        
        text_lower = text.lower()
        lines = text.split('\n')
        
        for section_name, pattern in section_patterns.items():
            section_content = []
            capturing = False
            
            for i, line in enumerate(lines):
                if re.search(pattern, line.lower()):
                    capturing = True
                    continue
                
                if capturing:
                    # Stop capturing if we hit another section
                    if any(re.search(p, line.lower()) for p in section_patterns.values() if p != pattern):
                        break
                    if line.strip():
                        section_content.append(line.strip())
            
            sections[section_name] = '\n'.join(section_content)
        
        return sections
    
    def _extract_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract technical and soft skills from resume text."""
        text_lower = text.lower()
        
        found_technical = []
        found_soft = []
        
        # Find technical skills
        for skill in TECHNICAL_SKILLS:
            if skill.lower() in text_lower:
                found_technical.append(skill)
        
        # Find soft skills
        for skill in SOFT_SKILLS:
            skill_words = skill.lower().split()
            if all(word in text_lower for word in skill_words):
                found_soft.append(skill)
        
        return found_technical, found_soft
    
    def _analyze_keywords(self, text: str, job_role: str) -> Tuple[List[str], List[str], float]:
        """Analyze keyword matching for specific job role."""
        if job_role not in JOB_KEYWORDS:
            return [], [], 0.0
        
        keywords = JOB_KEYWORDS[job_role]
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        missing_keywords = [kw for kw in keywords if kw not in found_keywords]
        match_percentage = (len(found_keywords) / len(keywords)) * 100
        
        return found_keywords, missing_keywords, match_percentage
    
    def _calculate_ats_score(self, text: str, sections: Dict[str, str]) -> float:
        """Calculate ATS compatibility score."""
        score = 0.0
        
        # Check for key sections (40 points)
        required_sections = ['experience', 'education', 'skills']
        for section in required_sections:
            if sections.get(section) and len(sections[section]) > 50:
                score += 13.33
        
        # Check text length (20 points)
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            score += 20
        elif word_count > 200:
            score += 10
        
        # Check for contact information (20 points)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        if re.search(email_pattern, text):
            score += 10
        if re.search(phone_pattern, text):
            score += 10
        
        # Check for bullet points (20 points)
        bullet_patterns = [r'•', r'◦', r'\*', r'-\s', r'→']
        bullet_count = sum(len(re.findall(pattern, text)) for pattern in bullet_patterns)
        if bullet_count >= 5:
            score += 20
        elif bullet_count >= 2:
            score += 10
        
        return min(score, 100.0)
    
    def _analyze_grammar(self, text: str) -> List[str]:
        """Analyze grammar and language quality."""
        issues = []
        
        # Basic grammar checks
        sentences = self.nlp_processor.sent_tokenize(text)
        
        for i, sentence in enumerate(sentences):
            # Check for long sentences
            if len(sentence.split()) > 30:
                issues.append(f"Sentence {i+1} might be too long ({len(sentence.split())} words)")
            
            # Check for repeated words
            words = sentence.lower().split()
            for j in range(len(words) - 1):
                if words[j] == words[j + 1] and len(words[j]) > 3:
                    issues.append(f"Repeated word '{words[j]}' in sentence {i+1}")
        
        return issues
    
    def _calculate_overall_score(self, ats_score: float, match_percentage: float, 
                                tech_skills_count: int, soft_skills_count: int) -> float:
        """Calculate overall resume score."""
        # Weighted combination of different factors
        ats_weight = 0.4
        match_weight = 0.4
        skills_weight = 0.2
        
        # Skills score (normalized)
        skills_score = min((tech_skills_count + soft_skills_count) * 2, 100)
        
        overall_score = (
            ats_score * ats_weight +
            match_percentage * match_weight +
            skills_score * skills_weight
        )
        
        return min(overall_score, 100.0)
    
    def _generate_recommendations(self, ats_score: float, match_percentage: float,
                                missing_keywords: List[str], grammar_issues: List[str],
                                sections: Dict[str, str]) -> List[str]:
        """Generate rule-based recommendations."""
        recommendations = []
        
        # ATS recommendations
        if ats_score < 70:
            recommendations.extend([
                "Improve ATS compatibility by adding more bullet points",
                "Ensure contact information is clearly visible",
                "Use standard section headings"
            ])
        
        # Keyword recommendations
        if match_percentage < 60:
            recommendations.append("Add more role-specific keywords to improve match")
        
        # Skills recommendations
        if not sections.get('skills'):
            recommendations.append("Add a dedicated skills section")
        
        # Grammar recommendations
        if grammar_issues:
            recommendations.append("Review and fix grammar issues")
        
        return recommendations
    
    def _get_ai_recommendations(self, resume_text: str, job_role: str,
                              found_keywords: List[str], missing_keywords: List[str]) -> str:
        """Get AI-powered recommendations using LLM."""
        try:
            prompt = f"""
            Analyze this resume for a {job_role} position and provide specific improvement recommendations.
            
            Resume content: {resume_text[:2000]}...
            
            Found keywords: {', '.join(found_keywords[:10])}
            Missing keywords: {', '.join(missing_keywords[:10])}
            
            Provide 3-5 specific, actionable recommendations to improve this resume for the {job_role} role.
            Focus on content, keywords, and formatting improvements.
            """
            
            response = self.llm_service.generate_response(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return "AI recommendations temporarily unavailable. Please try again later."
    
    def _get_rag_suggestions(self, resume_text: str, job_role: str) -> Dict[str, Any]:
        """Get RAG-based suggestions using similar resumes."""
        try:
            suggestions = self.rag_service.get_similar_resumes(resume_text, job_role)
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting RAG suggestions: {e}")
            return {"error": "RAG suggestions temporarily unavailable"}
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity * 100)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _generate_comparison_recommendations(self, resume_text: str, job_description: str,
                                           matched_skills: List[str], job_keywords: List[str]) -> List[str]:
        """Generate recommendations for job comparison."""
        recommendations = []
        
        if len(matched_skills) < len(job_keywords) * 0.5:
            recommendations.append("Add more skills that match the job requirements")
        
        if "experience" not in resume_text.lower():
            recommendations.append("Highlight relevant work experience")
        
        if "education" not in resume_text.lower():
            recommendations.append("Include educational background")
        
        return recommendations
