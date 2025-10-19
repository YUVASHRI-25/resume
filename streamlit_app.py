"""
Standalone Streamlit application for Resume Insight.
This version works independently without requiring the backend modules.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional
import io
import base64
import re
import nltk
from collections import Counter
import PyPDF2
import pdfplumber
from docx import Document

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Insight - AI-Powered Resume Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Colorful CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.main {
    padding-top: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stContainer {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    color: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.metric-label {
    font-size: 1rem;
    color: rgba(255, 255, 255, 0.9);
    margin-top: 0.5rem;
    font-weight: 500;
}

.warning-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border: none;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 30px rgba(240, 147, 251, 0.3);
}

.success-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    border: none;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
}

.error-box {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    border: none;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 30px rgba(250, 112, 154, 0.3);
}

.info-box {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border: none;
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: 15px;
    color: #333;
    box-shadow: 0 10px 30px rgba(168, 237, 234, 0.3);
}

.skill-tag {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 25px;
    padding: 0.5rem 1rem;
    margin: 0.25rem;
    font-size: 0.875rem;
    color: white;
    font-weight: 500;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    transition: transform 0.2s ease;
}

.skill-tag:hover {
    transform: scale(1.05);
}

.section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
}

.chat-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    backdrop-filter: blur(10px);
}

.chat-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 20px;
    margin: 0.5rem 0;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

.chat-user {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    margin-left: 2rem;
}

.chat-ai {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    margin-right: 2rem;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.stFileUploader > div {
    background: rgba(255, 255, 255, 0.9);
    border: 3px dashed #667eea;
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: #764ba2;
    background: rgba(255, 255, 255, 0.95);
    transform: scale(1.02);
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 15px 15px 0 0;
    padding: 1rem 2rem;
    margin: 0 0.25rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}

.stProgress > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    border: 2px solid #667eea;
}

.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    border: 2px solid #667eea;
}

.sidebar .sidebar-content {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.sidebar .sidebar-content .stMarkdown {
    color: white;
}

.gradient-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

.animated-card {
    animation: slideInUp 0.6s ease-out;
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
    100% {
        transform: scale(1);
    }
}
</style>
""", unsafe_allow_html=True)

# Job keywords database
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


class AIAssistant:
    """Simple AI Assistant for resume guidance."""
    
    def __init__(self):
        self.responses = {
            "experience": [
                "💡 **Experience Section Tips:**\n\n• Use bullet points with action verbs (Led, Managed, Developed)\n• Quantify achievements with numbers (Increased sales by 25%)\n• Focus on results, not just duties\n• Tailor content to match job requirements\n• Use reverse chronological order",
                "🚀 **Make Your Experience Stand Out:**\n\n• Start each bullet with a strong action verb\n• Include metrics and measurable results\n• Show progression and growth in your roles\n• Highlight relevant projects and achievements\n• Keep descriptions concise but impactful"
            ],
            "skills": [
                "🎯 **Skills Section Optimization:**\n\n• Organize technical and soft skills separately\n• Match skills to job description requirements\n• Include both hard and soft skills\n• Provide proficiency levels when appropriate\n• Update skills regularly to stay current",
                "⚡ **Skills That Get You Hired:**\n\n• Technical skills relevant to the role\n• Soft skills like leadership and communication\n• Industry-specific tools and technologies\n• Certifications and training\n• Languages (if relevant to the position)"
            ],
            "ats": [
                "🤖 **ATS-Friendly Resume Tips:**\n\n• Use standard section headings (Experience, Education, Skills)\n• Include relevant keywords naturally\n• Avoid images, graphics, and complex formatting\n• Use common fonts like Arial or Calibri\n• Save as PDF to preserve formatting",
                "📋 **ATS Optimization Checklist:**\n\n✅ Standard section headings\n✅ Contact information at the top\n✅ Bullet points for easy scanning\n✅ Relevant keywords throughout\n✅ Simple, clean formatting\n✅ PDF format preferred"
            ],
            "keywords": [
                "🔍 **Keyword Strategy:**\n\n• Study job descriptions in your field\n• Use industry-specific terminology\n• Include both acronyms and full terms\n• Incorporate keywords naturally\n• Don't overstuff - keep it readable",
                "🎯 **Finding the Right Keywords:**\n\n• Look at job postings for your target role\n• Use industry forums and websites\n• Check company websites for terminology\n• Include technical skills and tools\n• Add soft skills relevant to the role"
            ],
            "format": [
                "📄 **Resume Formatting Best Practices:**\n\n• Use clear, professional headings\n• Consistent bullet points and spacing\n• Readable fonts (10-12pt)\n• Appropriate white space\n• Clean, professional layout",
                "✨ **Visual Appeal Tips:**\n\n• Use consistent formatting throughout\n• Choose professional colors (black text on white)\n• Include adequate white space\n• Use bullet points for easy scanning\n• Keep it to 1-2 pages maximum"
            ],
            "general": [
                "🌟 **General Resume Tips:**\n\n• Customize for each job application\n• Proofread carefully for errors\n• Use a professional email address\n• Include a compelling summary\n• Show quantifiable achievements",
                "💼 **Career Success Tips:**\n\n• Research the company and role\n• Match your experience to job requirements\n• Highlight transferable skills\n• Show career progression\n• Include relevant certifications"
            ]
        }
    
    def get_response(self, query, context=""):
        """Get AI response based on query."""
        query_lower = query.lower()
        
        # Determine response category
        if any(word in query_lower for word in ['experience', 'work history', 'job history']):
            category = "experience"
        elif any(word in query_lower for word in ['skills', 'technical skills', 'abilities']):
            category = "skills"
        elif any(word in query_lower for word in ['ats', 'applicant tracking', 'ats-friendly']):
            category = "ats"
        elif any(word in query_lower for word in ['keywords', 'keyword', 'terms']):
            category = "keywords"
        elif any(word in query_lower for word in ['format', 'formatting', 'layout', 'design']):
            category = "format"
        else:
            category = "general"
        
        # Return a random response from the category
        import random
        response = random.choice(self.responses[category])
        
        # Add context if available
        if context:
            response += f"\n\n📊 **Based on your analysis:**\n{context}"
        
        return response


class ResumeAnalyzer:
    """Simple resume analyzer for standalone operation."""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.ai_assistant = AIAssistant()
    
    def extract_text_from_pdf(self, file):
        """Extract text from PDF file."""
        try:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            except:
                return "Error extracting PDF text"
    
    def extract_text_from_docx(self, file):
        """Extract text from DOCX file."""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except:
            return "Error extracting DOCX text"
    
    def extract_text_from_txt(self, file):
        """Extract text from TXT file."""
        try:
            return str(file.read(), "utf-8")
        except:
            return "Error extracting TXT text"
    
    def extract_sections(self, text):
        """Extract different sections from resume."""
        sections = {}
        
        section_patterns = {
            'education': r'(education|academic|qualification|degree|university|college)',
            'experience': r'(experience|employment|work|career|professional|job|position)',
            'skills': r'(skills|technical|competencies|expertise|abilities|technologies)',
            'projects': r'(projects|portfolio|work samples|personal projects)',
            'certifications': r'(certifications?|certificates?|licensed?|credentials)',
            'summary': r'(summary|objective|profile|about|overview)'
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
                    if any(re.search(p, line.lower()) for p in section_patterns.values() if p != pattern):
                        break
                    if line.strip():
                        section_content.append(line.strip())
            
            sections[section_name] = '\n'.join(section_content)
        
        return sections
    
    def extract_skills(self, text):
        """Extract technical and soft skills."""
        text_lower = text.lower()
        
        found_technical = []
        found_soft = []
        
        for skill in TECHNICAL_SKILLS:
            if skill.lower() in text_lower:
                found_technical.append(skill)
        
        for skill in SOFT_SKILLS:
            skill_words = skill.lower().split()
            if all(word in text_lower for word in skill_words):
                found_soft.append(skill)
        
        return found_technical, found_soft
    
    def keyword_matching(self, text, job_role):
        """Match keywords for specific job role."""
        if job_role not in JOB_KEYWORDS:
            return [], 0
        
        keywords = JOB_KEYWORDS[job_role]
        text_lower = text.lower()
        
        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        match_percentage = (len(found_keywords) / len(keywords)) * 100
        return found_keywords, match_percentage
    
    def calculate_ats_score(self, text, sections):
        """Calculate ATS friendliness score."""
        score = 0
        
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
        
        return min(score, 100)
    
    def analyze_resume(self, text, job_role):
        """Perform comprehensive resume analysis."""
        sections = self.extract_sections(text)
        tech_skills, soft_skills = self.extract_skills(text)
        found_keywords, match_percentage = self.keyword_matching(text, job_role)
        ats_score = self.calculate_ats_score(text, sections)
        
        overall_score = (ats_score + match_percentage) / 2
        
        return {
            "sections": sections,
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in JOB_KEYWORDS[job_role] if kw not in found_keywords],
            "role_match_percentage": match_percentage,
            "ats_score": ats_score,
            "overall_score": overall_score,
            "word_count": len(text.split()),
            "section_count": len([s for s in sections.values() if s])
        }


def display_metric_card(title, value, description=""):
    """Display a metric in a card format."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f"<small>{description}</small>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def display_alert_box(message, alert_type="info"):
    """Display alert box with different types."""
    box_class = f"{alert_type}-box"
    st.markdown(f"""
    <div class="{box_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Colorful Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   background-clip: text; 
                   font-size: 3.5rem; 
                   font-weight: 800; 
                   margin-bottom: 0.5rem;
                   text-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);">
            📄 Resume Insight
        </h1>
        <p style="font-size: 1.3rem; color: #667eea; font-weight: 600; margin-bottom: 0;">
            ✨ AI-Powered Resume Analysis & Optimization System ✨
        </p>
        <p style="color: #666; font-size: 1rem; margin-top: 0.5rem;">
            🚀 Transform your resume into a job-winning masterpiece
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Colorful Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; font-size: 1.5rem; margin-bottom: 1rem;">
                ⚙️ Analysis Configuration
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        job_roles = list(JOB_KEYWORDS.keys())
        selected_role = st.selectbox(
            "🎯 Target Job Role:",
            job_roles,
            help="Select the job role you're targeting to get relevant keyword analysis"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h3 style="color: white; font-size: 1.2rem; margin-bottom: 1rem;">
                🚀 Analysis Features
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="color: white; padding: 0.5rem;">
            <h4 style="color: #4facfe;">📊 Comprehensive Analysis:</h4>
            <ul style="color: rgba(255,255,255,0.9);">
                <li>🎯 ATS Compatibility Score</li>
                <li>🔍 Skills Detection & Matching</li>
                <li>📋 Section Structure Analysis</li>
                <li>🔑 Keyword Optimization</li>
            </ul>
            
            <h4 style="color: #fa709a;">🤖 AI Assistant:</h4>
            <ul style="color: rgba(255,255,255,0.9);">
                <li>💡 Personalized Resume Advice</li>
                <li>🚀 Quick Expert Recommendations</li>
                <li>💬 Interactive Q&A</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<h2 class="section-header">Upload Your Resume</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select your resume file",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT (Maximum size: 20MB)"
    )
    
    if uploaded_file is not None:
        # Show file information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        with st.expander("File Information", expanded=False):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        # Extract text based on file type
        analyzer = ResumeAnalyzer()
        
        with st.spinner("Processing your resume..."):
            try:
                if uploaded_file.type == "application/pdf":
                    text = analyzer.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = analyzer.extract_text_from_docx(uploaded_file)
                else:  # txt
                    text = analyzer.extract_text_from_txt(uploaded_file)
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
                return
        
        if "Error" not in text and text.strip():
            st.success("Resume processed successfully!")
            
            try:
                # Analyze the resume
                results = analyzer.analyze_resume(text, selected_role)
                
                # Create tabs for different analysis views
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Summary", "🔍 Skills Analysis", "📋 Section Review",
                    "🎯 ATS Analysis", "💡 Recommendations", "🤖 AI Assistant"
                ])
                
                with tab1:
                    st.markdown('<h3 class="section-header">Resume Summary</h3>', unsafe_allow_html=True)
                    
                    # Key metrics
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        display_metric_card("ATS Score", f"{results['ats_score']:.1f}/100")
                    
                    with metric_cols[1]:
                        display_metric_card("Role Match", f"{results['role_match_percentage']:.1f}%")
                    
                    with metric_cols[2]:
                        display_metric_card("Word Count", f"{results['word_count']}")
                    
                    with metric_cols[3]:
                        display_metric_card("Sections", f"{results['section_count']}/6")
                    
                    st.divider()
                    
                    # Overall assessment
                    overall_score = results['overall_score']
                    
                    if overall_score >= 80:
                        display_alert_box("Excellent resume! Your resume shows strong alignment with the target role and good ATS compatibility.", "success")
                    elif overall_score >= 60:
                        display_alert_box("Good foundation with room for improvement. Focus on adding more role-specific keywords and optimizing for ATS.", "warning")
                    else:
                        display_alert_box("Significant improvements needed. Consider restructuring sections, adding relevant keywords, and improving ATS compatibility.", "error")
                
                with tab2:
                    st.markdown('<h3 class="section-header">Skills Analysis</h3>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Technical Skills Detected")
                        tech_skills = results['technical_skills']
                        
                        if tech_skills:
                            tech_html = ""
                            for skill in tech_skills:
                                tech_html += f'<span class="skill-tag">{skill}</span>'
                            st.markdown(tech_html, unsafe_allow_html=True)
                            st.metric("Technical Skills Count", len(tech_skills))
                        else:
                            display_alert_box("No technical skills detected. Consider adding a dedicated skills section.", "warning")
                    
                    with col2:
                        st.subheader("Soft Skills Detected")
                        soft_skills = results['soft_skills']
                        
                        if soft_skills:
                            soft_html = ""
                            for skill in soft_skills:
                                soft_html += f'<span class="skill-tag">{skill}</span>'
                            st.markdown(soft_html, unsafe_allow_html=True)
                            st.metric("Soft Skills Count", len(soft_skills))
                        else:
                            display_alert_box("Limited soft skills detected. Consider highlighting leadership, communication, and teamwork skills.", "info")
                    
                    st.divider()
                    
                    # Role-specific analysis
                    st.subheader(f"Analysis for {selected_role}")
                    
                    progress_col, details_col = st.columns([1, 2])
                    
                    with progress_col:
                        match_percentage = results['role_match_percentage']
                        st.metric("Match Percentage", f"{match_percentage:.1f}%")
                        st.progress(match_percentage / 100)
                    
                    with details_col:
                        if match_percentage >= 70:
                            display_alert_box("Excellent match for this role! Your skills align well with industry expectations.", "success")
                        elif match_percentage >= 50:
                            display_alert_box("Good match with opportunities for improvement. Consider adding more role-specific skills.", "warning")
                        else:
                            display_alert_box("Limited match detected. Focus on adding more relevant skills and keywords for this role.", "error")
                    
                    # Missing keywords
                    missing_keywords = results['missing_keywords']
                    
                    if missing_keywords:
                        st.subheader("Suggested Keywords to Add")
                        missing_html = ""
                        for keyword in missing_keywords[:15]:
                            missing_html += f'<span class="skill-tag" style="background-color: #fff3cd;">{keyword}</span>'
                        st.markdown(missing_html, unsafe_allow_html=True)
                
                with tab3:
                    st.markdown('<h3 class="section-header">Section Structure Review</h3>', unsafe_allow_html=True)
                    
                    sections = results['sections']
                    
                    for section_name, section_content in sections.items():
                        with st.expander(f"{section_name.title()} Section", expanded=False):
                            if section_content:
                                st.text_area(
                                    f"{section_name.title()} content:",
                                    section_content,
                                    height=150,
                                    disabled=True,
                                    key=f"section_{section_name}"
                                )
                            else:
                                st.warning(f"No {section_name} section found or content is too brief")
                
                with tab4:
                    st.markdown('<h3 class="section-header">ATS Compatibility Analysis</h3>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        ats_score = results['ats_score']
                        st.metric("Overall ATS Score", f"{ats_score}/100")
                        st.progress(ats_score / 100)
                        
                        if ats_score >= 80:
                            st.success("Excellent ATS compatibility")
                        elif ats_score >= 60:
                            st.warning("Good ATS compatibility")
                        else:
                            st.error("Needs ATS optimization")
                    
                    with col2:
                        st.subheader("ATS Optimization Tips")
                        
                        tips = [
                            "Use standard section headings: Experience, Education, Skills, etc.",
                            "Include relevant keywords naturally throughout your resume",
                            "Use bullet points to improve readability and scanning",
                            "Avoid images, graphics, tables, and complex formatting",
                            "Use standard fonts like Arial, Calibri, or Times New Roman",
                            "Save your resume as a PDF to preserve formatting",
                            "Include your contact information prominently at the top",
                            "Use consistent formatting throughout the document"
                        ]
                        
                        for i, tip in enumerate(tips, 1):
                            st.write(f"{i}. {tip}")
                
                with tab5:
                    st.markdown('<h3 class="section-header">Personalized Recommendations</h3>', unsafe_allow_html=True)
                    
                    recommendations = []
                    
                    # Generate recommendations based on analysis
                    if results['ats_score'] < 70:
                        recommendations.extend([
                            "Improve ATS compatibility by adding more bullet points throughout your resume",
                            "Ensure your contact information (email and phone) is clearly visible at the top",
                            "Use standard section headings that ATS systems can easily recognize"
                        ])
                    
                    if results['role_match_percentage'] < 60:
                        recommendations.append(f"Increase your match for {selected_role} by incorporating more industry-specific keywords")
                    
                    if not results['technical_skills']:
                        recommendations.append("Add a dedicated Technical Skills section to highlight your capabilities")
                    
                    if not results['soft_skills']:
                        recommendations.append("Incorporate more soft skills like leadership, communication, and teamwork throughout your experience descriptions")
                    
                    if recommendations:
                        st.subheader("📋 Improvement Recommendations")
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"**{i}.** {rec}")
                    else:
                        display_alert_box("Your resume looks great! Minor tweaks based on specific job applications can further improve your success rate.", "success")
                    
                    st.divider()
                    
                    # Action plan
                    st.subheader("🎯 Priority Action Plan")
                    
                    action_items = []
                    
                    if results['ats_score'] < 60:
                        action_items.append("**High Priority:** Improve ATS compatibility - focus on formatting and standard sections")
                    
                    if results['role_match_percentage'] < 50:
                        action_items.append("**High Priority:** Add more role-specific keywords and skills")
                    
                    action_items.append("**Ongoing:** Customize your resume for each job application by matching keywords")
                    
                    for action in action_items:
                        st.markdown(action)
                
                with tab6:
                    st.markdown('<h3 class="section-header">🤖 AI Resume Assistant</h3>', unsafe_allow_html=True)
                    
                    # Initialize chat history in session state
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Quick questions
                    st.subheader("🚀 Quick Questions")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💼 How to improve experience section?", use_container_width=True):
                            response = analyzer.ai_assistant.get_response("How to improve experience section?")
                            st.session_state.chat_history.append({"role": "user", "message": "How to improve experience section?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                    
                    with col2:
                        if st.button("🤖 Make resume ATS-friendly?", use_container_width=True):
                            response = analyzer.ai_assistant.get_response("How to make resume ATS-friendly?")
                            st.session_state.chat_history.append({"role": "user", "message": "How to make resume ATS-friendly?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.button("🔍 Add better keywords?", use_container_width=True):
                            response = analyzer.ai_assistant.get_response("What keywords should I add?")
                            st.session_state.chat_history.append({"role": "user", "message": "What keywords should I add?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                    
                    with col4:
                        if st.button("⚡ Improve skills section?", use_container_width=True):
                            response = analyzer.ai_assistant.get_response("How can I improve my skills section?")
                            st.session_state.chat_history.append({"role": "user", "message": "How can I improve my skills section?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                    
                    st.divider()
                    
                    # Chat interface
                    st.subheader("💬 Ask Me Anything")
                    
                    # Chat input
                    user_question = st.text_input(
                        "Ask about your resume:",
                        placeholder="How can I improve my resume?",
                        key="chat_input"
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        send_clicked = st.button("Send 💬", type="primary", use_container_width=True)
                    with col2:
                        if st.button("Clear Chat 🗑️", use_container_width=True):
                            st.session_state.chat_history = []
                            st.rerun()
                    
                    # Process user input
                    if send_clicked and user_question.strip():
                        # Generate context from analysis results
                        context = f"ATS Score: {results['ats_score']:.1f}/100, Role Match: {results['role_match_percentage']:.1f}%, Technical Skills: {len(results['technical_skills'])}, Soft Skills: {len(results['soft_skills'])}"
                        
                        response = analyzer.ai_assistant.get_response(user_question, context)
                        
                        st.session_state.chat_history.append({"role": "user", "message": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "message": response})
                        st.rerun()
                    
                    # Display chat history
                    if st.session_state.chat_history:
                        st.subheader("📱 Conversation History")
                        
                        for i, chat in enumerate(reversed(st.session_state.chat_history[-6:])):  # Show last 6 messages
                            if chat["role"] == "user":
                                st.markdown(f"""
                                <div class="chat-message chat-user">
                                    <strong>👤 You:</strong> {chat["message"]}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="chat-message chat-ai">
                                    <strong>🤖 AI Assistant:</strong><br>
                                    {chat["message"]}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # AI Assistant info
                    st.divider()
                    st.markdown("""
                    <div class="info-box">
                        <h4>🎯 AI Assistant Features:</h4>
                        <ul>
                            <li>💡 Personalized resume advice</li>
                            <li>🚀 Quick expert recommendations</li>
                            <li>📊 Context-aware suggestions</li>
                            <li>🎨 Formatting and design tips</li>
                            <li>🔍 Keyword optimization guidance</li>
                            <li>⚡ ATS compatibility tips</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.error("Please check your resume format and try again.")
    
    else:
        # Instructions when no file is uploaded
        st.markdown('<h3 class="section-header">Getting Started</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("How It Works")
            st.markdown("""
            **1. Upload Your Resume**
            Upload your resume in PDF, DOCX, or TXT format using the file uploader above.
            
            **2. Select Target Role**
            Choose your target job role from the sidebar to get relevant keyword analysis.
            
            **3. Get Comprehensive Analysis**
            Review detailed analysis across multiple categories including ATS compatibility, skills matching, and section structure.
            
            **4. Get Recommendations**
            Receive personalized suggestions for improving your resume.
            """)
        
        with col2:
            st.subheader("Analysis Features")
            
            features = [
                "ATS Compatibility Scoring",
                "Skills Detection & Matching",
                "Keyword Optimization",
                "Section Structure Analysis",
                "Role-Specific Recommendations"
            ]
            
            for feature in features:
                st.write(f"✓ {feature}")


if __name__ == "__main__":
    main()
