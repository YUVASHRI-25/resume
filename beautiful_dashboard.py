"""
Beautiful Modern Resume Insight Dashboard (Soft Pastel Rainbow Edition)
-- WITH:
 - Numbered AI Assistant responses (1., 2., 3., ...)
 - Clean pastel AI chat cards
 - Topic-colored AI cards (Skills, ATS, Experience, Keywords)
 - Properly aligned, rounded Suggested Keyword chips
 - Matched Pro Tip pastel box
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
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import PyPDF2
import pdfplumber
from docx import Document
import os
import json
from dotenv import load_dotenv
import random
from openai import OpenAI


# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RESUME INSIGHT 🌈",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Soft Pastel Rainbow CSS (updated)
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* App background */
.stApp {
    background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 50%, #f6f3ff 100%);
    min-height: 100vh;
}

/* Header */
.modern-header {
    background: linear-gradient(120deg, #ffdee9 0%, #b5fffc 50%, #d4fc79 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 15px 40px rgba(255, 182, 193, 0.3);
    position: relative;
}
.header-title { font-size: 3rem; font-weight: 800; color: #334155; text-shadow: 0 2px 8px rgba(255,255,255,0.8); }
.header-subtitle { font-size: 1.2rem; color: #475569; font-weight: 600; }

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #fef9ef 0%, #fde2e4 100%);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    transition: all 0.4s ease;
}
.metric-card:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
.metric-value { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 50%, #ffdde1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { font-size: 1rem; font-weight: 600; color: #64748b; }

/* Alerts */
.alert-modern { border-radius: 12px; padding: 1.5rem 2rem; margin: 1rem 0; font-weight: 500; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
.alert-success { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); color: #14532d; }
.alert-warning { background: linear-gradient(135deg, #fff6bd 0%, #fef3c7 100%); color: #78350f; }
.alert-error { background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%); color: #7f1d1d; }

/* Cards */
.modern-card { background: linear-gradient(135deg, #ffffff 0%, #f9f9ff 100%); border-radius: 16px; padding: 2rem; margin: 1rem 0; box-shadow: 0 8px 25px rgba(0,0,0,0.08); border: 1px solid rgba(226, 232, 240, 0.8); }

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #ff9a9e 0%, #fad0c4 30%, #a1c4fd 100%);
    color: #334155; font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; border: none; transition: all 0.3s ease;
}
.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(255, 182, 193, 0.3); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: linear-gradient(90deg, #fceabb 0%, #f8b500 100%); border-radius: 16px; padding: 0.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); color: #334155 !important; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.1); }

/* Sidebar */
.sidebar-modern { background: linear-gradient(135deg, #fef9ef 0%, #e0c3fc 100%); border-radius: 16px; padding: 1.5rem; box-shadow: 0 6px 20px rgba(0,0,0,0.05); }

/* Skill Tags */
.skill-tag {
    display: inline-block; background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%); color: #334155; border-radius: 20px; padding: 0.5rem 1rem; margin: 0.3rem; font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.05); transition: all 0.3s ease;
}
.skill-tag:hover { transform: translateY(-2px); background: linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%); }

/* Suggested Keywords (chips) */
.keyword-suggested {
    display: inline-block; background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%); color: #0f172a; border-radius: 25px; padding: 0.6rem 1.2rem; margin: 0.4rem; font-weight: 600; font-size: 0.9rem; white-space: nowrap; box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s ease, background 0.3s ease;
}
.keyword-suggested:hover { transform: translateY(-3px); background: #bae6fd; }

/* Progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg, #ff9a9e 0%, #fad0c4 25%, #a1c4fd 50%, #c2e9fb 75%, #fbc2eb 100%); border-radius: 10px; }

/* Chat messages (base) */
.chat-message { border-radius: 16px; padding: 1.5rem; margin: 0.75rem 0; font-size: 0.95rem; line-height: 1.8; box-shadow: 0 6px 20px rgba(0,0,0,0.05); }

/* Chat - default AI */
.chat-ai { background: linear-gradient(135deg, #fff0f6 0%, #ffe5f0 50%, #ffecec 100%); color: #1e293b; border-left: 6px solid #f472b6; }

/* Chat - user */
.chat-user { background: linear-gradient(135deg, #e0f7ff 0%, #e6f0ff 50%, #e0f0ff 100%); color: #1e293b; border-left: 6px solid #60a5fa; text-align:left; }

/* Topic-specific AI colors */
.chat-ai-skills { background: linear-gradient(135deg, #fff0f6 0%, #ffe7f2 50%, #fff7fb 100%); border-left: 6px solid #f472b6; }
.chat-ai-ats { background: linear-gradient(135deg, #ecfeff 0%, #e6f7ff 50%, #f0fbff 100%); border-left: 6px solid #0ea5a0; }
.chat-ai-experience { background: linear-gradient(135deg, #f0fff4 0%, #e9ffef 50%, #f7fff9 100%); border-left: 6px solid #16a34a; }
.chat-ai-keywords { background: linear-gradient(135deg, #f3f0ff 0%, #efe9ff 50%, #fbf8ff 100%); border-left: 6px solid #7c3aed; }

/* Chat headings/emphasis */
.chat-message strong { font-weight: 700; color: #0f172a; display: block; margin-bottom: 0.5rem; }
.chat-ai div { line-height: 1.9; margin-left: 0.5rem; }

/* Section header style */
.section-header-modern { background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 30%, #ffdde1 100%); color: #334155; font-weight: 700; text-align: center; padding: 1rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: 0 8px 20px rgba(0,0,0,0.08); }

/* Pro Tip box (info) */
.alert-modern.alert-info { background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%); color: #0f172a; border-left: 6px solid #3b82f6; box-shadow: 0 6px 20px rgba(0,0,0,0.05); padding: 1rem 1.5rem; border-radius: 12px; font-size: 0.95rem; line-height: 1.7; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Data / DB / Keywords (unchanged logic)
# ---------------------------
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'resume_insight'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'port': os.getenv('DB_PORT', '5432')
}

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

# ---------------------------
# Database Manager (unchanged)
# ---------------------------
class DatabaseManager:
    def __init__(self):
        self.config = DB_CONFIG

    def get_connection(self):
        try:
            return psycopg2.connect(**self.config)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None

    def create_tables(self):
        conn = self.get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS resume_analyses (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        filename VARCHAR(255) NOT NULL,
                        job_role VARCHAR(100) NOT NULL,
                        ats_score DECIMAL(5,2),
                        role_match_percentage DECIMAL(5,2),
                        overall_score DECIMAL(5,2),
                        technical_skills TEXT[],
                        soft_skills TEXT[],
                        found_keywords TEXT[],
                        missing_keywords TEXT[],
                        analysis_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id),
                        session_id VARCHAR(100),
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error creating tables: {e}")
            return False
        finally:
            conn.close()

    def authenticate_user(self, username, password):
        conn = self.get_connection()
        if not conn:
            return None
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cur.execute(
                    "SELECT id, username, email FROM users WHERE username = %s AND password_hash = %s",
                    (username, password_hash)
                )
                user = cur.fetchone()
                return dict(user) if user else None
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return None
        finally:
            conn.close()

    def register_user(self, username, email, password):
        conn = self.get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cur.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                    (username, email, password_hash)
                )
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Registration error: {e}")
            return False
        finally:
            conn.close()

    def save_analysis(self, user_id, filename, job_role, analysis_data):
        conn = self.get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                analysis_json = json.dumps(analysis_data)
                cur.execute("""
                    INSERT INTO resume_analyses 
                    (user_id, filename, job_role, ats_score, role_match_percentage, overall_score,
                     technical_skills, soft_skills, found_keywords, missing_keywords, analysis_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, filename, job_role,
                    analysis_data.get('ats_score'),
                    analysis_data.get('role_match_percentage'),
                    analysis_data.get('overall_score'),
                    analysis_data.get('technical_skills', []),
                    analysis_data.get('soft_skills', []),
                    analysis_data.get('found_keywords', []),
                    analysis_data.get('missing_keywords', []),
                    analysis_json
                ))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error saving analysis: {e}")
            return False
        finally:
            conn.close()

    def get_user_analyses(self, user_id):
        conn = self.get_connection()
        if not conn:
            return []
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM resume_analyses WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Error fetching analyses: {e}")
            return []
        finally:
            conn.close()

    def save_chat_message(self, user_id, session_id, message, response):
        conn = self.get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO chat_history (user_id, session_id, message, response) VALUES (%s, %s, %s, %s)",
                            (user_id, session_id, message, response))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error saving chat message: {e}")
            return False
        finally:
            conn.close()

# ---------------------------
# Resume Analyzer (unchanged logic)
# ---------------------------
class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.ai_assistant = AIAssistant()

    def extract_text_from_pdf(self, file):
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
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except:
            return "Error extracting DOCX text"

    def extract_text_from_txt(self, file):
        try:
            return str(file.read(), "utf-8")
        except:
            return "Error extracting TXT text"

    def extract_sections(self, text):
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
        score = 0
        required_sections = ['experience', 'education', 'skills']
        for section in required_sections:
            if sections.get(section) and len(sections[section]) > 50:
                score += 13.33
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            score += 20
        elif word_count > 200:
            score += 10
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        if re.search(email_pattern, text):
            score += 10
        if re.search(phone_pattern, text):
            score += 10
        bullet_patterns = [r'•', r'◦', r'\*', r'-\s', r'→']
        bullet_count = sum(len(re.findall(pattern, text)) for pattern in bullet_patterns)
        if bullet_count >= 5:
            score += 20
        elif bullet_count >= 2:
            score += 10
        return min(score, 100)

    def analyze_resume(self, text, job_role):
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

# ---------------------------
# AI Assistant (unchanged responses)
# ---------------------------
class AIAssistant:
    def __init__(self):
        self.responses = {
            "experience": [
                "**🚀 Experience Section Optimization:**\n\nStart each bullet with a strong action verb\nInclude metrics and measurable results\nShow progression and growth in your roles\nHighlight relevant projects and achievements\nKeep descriptions concise but impactful\n\n**🎯 Focus:** Show how you added value, not just what you did!",
                "**⭐ Make Your Experience Stand Out:**\n\nStart each bullet with a strong action verb\nInclude metrics and measurable results\nShow progression and growth in your roles\nHighlight relevant projects and achievements\nKeep descriptions concise but impactful\n\n**🎯 Focus:** Show how you added value, not just what you did!"
            ],
            "skills": [
                "**🎯 Skills Section Optimization:**\n\nOrganize technical and soft skills separately\nMatch skills to job description requirements\nInclude both hard and soft skills\nProvide proficiency levels when appropriate\nUpdate skills regularly to stay current\n\n💪 Power Skills: Technical skills + Soft skills = Winning combination!",
                "**⚡ Skills That Get You Hired:**\n\nTechnical skills relevant to the role\nSoft skills like leadership and communication\nIndustry-specific tools and technologies\nCertifications and training\nLanguages (if relevant to the position)\n\n🔥 Hot Tip: Include both technical and soft skills for maximum impact!"
            ],
            "ats": [
                "**📋 ATS Optimization Checklist:**\n\nStandard section headings\nContact information at the top\nBullet points for easy scanning\nRelevant keywords throughout\nSimple, clean formatting\nPDF format preferred\n\n🎯 Goal: Make it easy for ATS systems to read and parse your resume!",
                "**🤖 ATS-Friendly Resume Tips:**\n\nUse standard section headings (Experience, Education, Skills)\nInclude relevant keywords naturally\nAvoid images, graphics, and complex formatting\nUse common fonts like Arial or Calibri\nSave as PDF to preserve formatting\n\n✅ ATS Checklist: Standard headings + Keywords + Simple format = ATS success!"
            ],
            "keywords": [
                "**🔍 Keyword Strategy:**\n\nLook at job postings for your target role\nUse industry forums and websites\nCheck company websites for terminology\nInclude technical skills and tools\nAdd soft skills relevant to the role\n\n🚀 Power Move: Use keywords from the actual job description you're applying for!",
                "**🎯 Finding the Right Keywords:**\n\nStudy job descriptions in your field\nUse industry-specific terminology\nInclude both acronyms and full terms\nIncorporate keywords naturally\nDon't overstuff - keep it readable\n\n💡 Pro Strategy: Keywords should flow naturally in your content!"
            ],
            "format": [
                "**📄 Resume Formatting Best Practices:**\n\nUse clear, professional headings\nConsistent bullet points and spacing\nReadable fonts (10-12pt)\nAppropriate white space\nClean, professional layout\n\n✨ Visual Appeal: Clean formatting = Professional impression!",
                "**🎨 Visual Appeal Tips:**\n\nUse consistent formatting throughout\nChoose professional colors (black text on white)\nInclude adequate white space\nUse bullet points for easy scanning\nKeep it to 1-2 pages maximum\n\n💎 Remember: Your resume is your first impression - make it count!"
            ],
            "general": [
                "**🌟 General Resume Tips:**\n\nCustomize for each job application\nProofread carefully for errors\nUse a professional email address\nInclude a compelling summary\nShow quantifiable achievements\n\n🎯 Success Formula: Customization + Quality + Relevance = Job offers!",
                "**💼 Career Success Tips:**\n\nResearch the company and role\nMatch your experience to job requirements\nHighlight transferable skills\nShow career progression\nInclude relevant certifications\n\n🚀 Pro Tip: Every resume should tell a story of growth and achievement!"
            ]
        }

    def get_response(self, query, context=""):
        query_lower = query.lower()
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
        response = random.choice(self.responses[category])
        if context:
            response += f"\n\n**📊 Based on your analysis:**\n{context}"
        return response

# ---------------------------
# Helper UI functions
# ---------------------------
def display_metric_card(title, value, description=""):
    st.markdown(f"""
    <div class="metric-card fade-in-up">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f"<small>{description}</small>" if description else ""}
    </div>
    """, unsafe_allow_html=True)

def display_alert_box(message, alert_type="info"):
    st.markdown(f"""
    <div class="alert-modern alert-{alert_type} fade-in-up">
        {message}
    </div>
    """, unsafe_allow_html=True)

# Render AI Assistant messages with numbering and topic-based color classes
def render_ai_message(message):
    """
    Formats AI assistant message:
    - Removes bullet dots
    - Converts content lines into numbered lines (except headings/emojis)
    - Chooses CSS class based on detected topic keywords
    - Renders a pastel card
    """
    # Clean bullets
    msg = re.sub(r"•\s*", "", message)

    # Normalize multiple blank lines
    lines = [ln.strip() for ln in msg.split("\n") if ln.strip()]

    numbered_lines = []
    count = 1
    for line in lines:
        # treat lines that are likely headings (start with emoji or are short)
        if line.startswith(("🎯", "💪", "⭐", "📋", "🚀", "🔥", "🤖")) or len(line.split()) <= 3 and line.endswith(":"):
            numbered_lines.append(line)
        else:
            # number regular content lines
            numbered_lines.append(f"{count}. {line}")
            count += 1

    formatted = "<br>".join(numbered_lines)

    # detect topic for CSS coloring
    lower = msg.lower()
    if 'skill' in lower or 'skills' in lower or 'skills section' in lower:
        cls = "chat-ai-skills"
    elif 'ats' in lower or 'applicant tracking' in lower or 'ats-friendly' in lower:
        cls = "chat-ai-ats"
    elif 'experience' in lower or 'work history' in lower or 'experience section' in lower:
        cls = "chat-ai-experience"
    elif 'keyword' in lower or 'keywords' in lower or 'keyword strategy' in lower:
        cls = "chat-ai-keywords"
    else:
        cls = "chat-ai"

    st.markdown(f"""
    <div class="chat-message {cls} fade-in-up">
        <strong>🤖 AI Assistant:</strong>
        <div style="margin-top:0.5rem;">{formatted}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Login / Dashboard UI (main app) - with updated chat rendering
# ---------------------------
def login_page():
    st.markdown("""
    <div class="login-container fade-in-up">
        <h2 style="text-align: center; color: #334155; margin-bottom: 2rem; font-weight: 700;">
            📊 RESUME INSIGHT
        </h2>
        <p style="text-align: center; color: #64748b; margin-bottom: 2rem;">
            Beautiful Modern AI-Powered Resume Analysis Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔐 Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            login_button = st.form_submit_button("🚀 Login", use_container_width=True)
            if login_button:
                db = DatabaseManager()
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.success("🎉 Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        st.markdown("---")
        st.markdown("### ✨ Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("👤 New Username")
            new_email = st.text_input("📧 Email")
            new_password = st.text_input("🔒 New Password", type="password")
            register_button = st.form_submit_button("🚀 Register", use_container_width=True)
            if register_button:
                db = DatabaseManager()
                if db.register_user(new_username, new_email, new_password):
                    st.success("🎉 Registration successful! Please login.")
                else:
                    st.error("❌ Registration failed. Username or email may already exist.")

def dashboard_page():
    user = st.session_state.user
    st.markdown(f"""
    <div class="modern-header fade-in-up">
        <div class="header-title">Welcome back, {user['username']}! 👋</div>
        <div class="header-subtitle">📊 Beautiful Modern Resume Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-modern fade-in-up">
            <h3 style="color: #334155; margin-bottom: 1rem;">⚙️ Analysis Configuration</h3>
        </div>
        """, unsafe_allow_html=True)

        job_roles = list(JOB_KEYWORDS.keys())
        selected_role = st.selectbox("🎯 Target Job Role:", job_roles, help="Select the job role you're targeting")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

        st.markdown("""
        <div class="sidebar-modern fade-in-up">
            <h3 style="color: #334155; margin-bottom: 1rem;">📊 Analysis History</h3>
        </div>
        """, unsafe_allow_html=True)

        db = DatabaseManager()
        analyses = db.get_user_analyses(user['id'])
        if analyses:
            for analysis in analyses[:5]:
                with st.expander(f"📄 {analysis['filename']} - {analysis['job_role']}"):
                    st.markdown(f"""
                    <div class="history-card">
                        <p><strong>📅 Date:</strong> {analysis['created_at'].strftime('%Y-%m-%d %H:%M')}</p>
                        <p><strong>🎯 ATS Score:</strong> {analysis['ats_score']}/100</p>
                        <p><strong>📈 Role Match:</strong> {analysis['role_match_percentage']:.1f}%</p>
                        <p><strong>⭐ Overall Score:</strong> {analysis['overall_score']:.1f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📝 No previous analyses found")

    st.markdown('<div class="section-header-modern fade-in-up">📄 Upload Your Resume</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Select your resume file", type=['pdf', 'docx', 'txt'], help="Supported formats: PDF, DOCX, TXT")

    if uploaded_file is not None:
        file_details = {"📄 Filename": uploaded_file.name, "📊 File size": f"{uploaded_file.size / 1024:.1f} KB", "🔧 File type": uploaded_file.type}
        with st.expander("📋 File Information", expanded=False):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

        analyzer = ResumeAnalyzer()
        with st.spinner("🔄 Processing your resume..."):
            try:
                if uploaded_file.type == "application/pdf":
                    text = analyzer.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = analyzer.extract_text_from_docx(uploaded_file)
                else:
                    text = analyzer.extract_text_from_txt(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error extracting text: {str(e)}")
                return

        if "Error" not in text and text.strip():
            st.success("✅ Resume processed successfully!")
            try:
                results = analyzer.analyze_resume(text, selected_role)
                db = DatabaseManager()
                if db.save_analysis(user['id'], uploaded_file.name, selected_role, results):
                    st.success("💾 Analysis saved to history!")
                else:
                    st.warning("⚠️ Could not save analysis to history")

                tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔍 Skills Analysis", "🎯 ATS Analysis", "🤖 AI Assistant"])

                # ----- TAB 1: Summary -----
                with tab1:
                    st.markdown('<div class="section-header-modern fade-in-up">📊 Resume Summary</div>', unsafe_allow_html=True)
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        display_metric_card("ATS Score", f"{results['ats_score']:.1f}/100")
                    with metric_cols[1]:
                        display_metric_card("Role Match", f"{results['role_match_percentage']:.1f}%")
                    with metric_cols[2]:
                        display_metric_card("Word Count", f"{results['word_count']}")

                    overall_score = results['overall_score']
                    if overall_score >= 80:
                        display_alert_box("🎉 Excellent resume! Your resume shows strong alignment with the target role and good ATS compatibility.", "success")
                    elif overall_score >= 60:
                        display_alert_box("⚠️ Good foundation with room for improvement. Focus on adding more role-specific keywords and optimizing for ATS.", "warning")
                    else:
                        display_alert_box("🚨 Significant improvements needed. Consider restructuring sections, adding relevant keywords, and improving ATS compatibility.", "error")

                # ----- TAB 2: Skills Analysis -----
                with tab2:
                    st.markdown('<div class="section-header-modern fade-in-up">🔍 Skills Analysis</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 💻 Technical Skills Detected")
                        tech_skills = results['technical_skills']
                        if tech_skills:
                            tech_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in tech_skills])
                            st.markdown(tech_html, unsafe_allow_html=True)
                            st.metric("Technical Skills Count", len(tech_skills))
                        else:
                            display_alert_box("⚠️ No technical skills detected. Consider adding a dedicated skills section.", "warning")
                    with col2:
                        st.markdown("### 🤝 Soft Skills Detected")
                        soft_skills = results['soft_skills']
                        if soft_skills:
                            soft_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in soft_skills])
                            st.markdown(soft_html, unsafe_allow_html=True)
                            st.metric("Soft Skills Count", len(soft_skills))
                        else:
                            display_alert_box("ℹ️ Limited soft skills detected. Consider highlighting leadership, communication, and teamwork skills.", "info")

                    st.divider()
                    st.markdown("### 🎯 Analysis for " + selected_role)
                    progress_col, details_col = st.columns([1, 2])
                    with progress_col:
                        match_percentage = results['role_match_percentage']
                        st.metric("Match Percentage", f"{match_percentage:.1f}%")
                        st.progress(match_percentage / 100)
                    with details_col:
                        if match_percentage >= 70:
                            display_alert_box("🎉 Excellent match for this role! Your skills align well with industry expectations.", "success")
                        elif match_percentage >= 50:
                            display_alert_box("⚠️ Good match with opportunities for improvement. Consider adding more role-specific skills.", "warning")
                        else:
                            display_alert_box("🚨 Limited match detected. Focus on adding more relevant skills and keywords for this role.", "error")

                    # Suggested Keywords (clean chips + Pro Tip)
                    missing_keywords = results['missing_keywords']
                    if missing_keywords:
                        st.markdown(f"""
                        <div class="section-header-modern fade-in-up" style="margin-top: 1.5rem;">🔑 Suggested Keywords to Add</div>
                        <p style="color: #475569; margin-bottom: 1rem; font-size: 1rem;">
                            💡 These keywords are commonly found in <strong>{selected_role}</strong> job descriptions:
                        </p>
                        """, unsafe_allow_html=True)
                        missing_html = "".join([f'<span class="keyword-suggested">{kw}</span>' for kw in missing_keywords[:30]])
                        st.markdown(missing_html, unsafe_allow_html=True)
                        st.markdown("""
                        <div class="alert-modern alert-info fade-in-up">
                            💡 <strong>Pro Tip:</strong> Try to naturally incorporate these keywords into your resume sections.
                            Don't just add them randomly — weave them into your experience descriptions and skills sections.
                        </div>
                        """, unsafe_allow_html=True)

                    # Quick AI questions (use render_ai_message)
                    st.divider()
                    st.markdown("### 🤖 AI Assistant for Skills")
                    st.markdown("""
                    <div class="ai-assistant-modern fade-in-up">
                        <h4 style="color: #334155; margin-bottom: 1rem;">💬 Quick AI Questions</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💡 How to improve my skills section?", key="btn_skill_improve"):
                            response = analyzer.ai_assistant.get_response("How to improve my skills section?")
                            # append to chat history and render
                            if 'chat_history' not in st.session_state:
                                st.session_state.chat_history = []
                            st.session_state.chat_history.append({"role": "user", "message": "How to improve my skills section?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                            render_ai_message(response)
                    with col2:
                        if st.button("🔍 What keywords should I add?", key="btn_skill_keywords"):
                            response = analyzer.ai_assistant.get_response("What keywords should I add?")
                            if 'chat_history' not in st.session_state:
                                st.session_state.chat_history = []
                            st.session_state.chat_history.append({"role": "user", "message": "What keywords should I add?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                            render_ai_message(response)

                # ----- TAB 3: ATS Analysis -----
                with tab3:
                    st.markdown('<div class="section-header-modern fade-in-up">🎯 ATS Compatibility Analysis</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        ats_score = results['ats_score']
                        st.metric("Overall ATS Score", f"{ats_score}/100")
                        st.progress(ats_score / 100)
                        if ats_score >= 80:
                            st.success("🎉 Excellent ATS compatibility")
                        elif ats_score >= 60:
                            st.warning("⚠️ Good ATS compatibility")
                        else:
                            st.error("🚨 Needs ATS optimization")
                    with col2:
                        st.markdown("### 💡 ATS Optimization Tips")
                        tips = [
                            "✅ Use standard section headings: Experience, Education, Skills, etc.",
                            "✅ Include relevant keywords naturally throughout your resume",
                            "✅ Use bullet points to improve readability and scanning",
                            "✅ Avoid images, graphics, tables, and complex formatting",
                            "✅ Use standard fonts like Arial, Calibri, or Times New Roman",
                            "✅ Save your resume as a PDF to preserve formatting",
                            "✅ Include your contact information prominently at the top",
                            "✅ Use consistent formatting throughout the document"
                        ]
                        for tip in tips:
                            st.markdown(tip)

                # ----- TAB 4: AI Assistant (full chat) -----
                with tab4:
                    st.markdown('<div class="section-header-modern fade-in-up">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []

                    st.markdown("### 🚀 Quick Questions")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💼 How to improve experience section?", key="btn_exp"):
                            response = analyzer.ai_assistant.get_response("How to improve experience section?")
                            st.session_state.chat_history.append({"role": "user", "message": "How to improve experience section?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                    with col2:
                        if st.button("🤖 Make resume ATS-friendly?", key="btn_ats"):
                            response = analyzer.ai_assistant.get_response("How to make resume ATS-friendly?")
                            st.session_state.chat_history.append({"role": "user", "message": "How to make resume ATS-friendly?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})

                    col3, col4 = st.columns(2)
                    with col3:
                        if st.button("🔍 Add better keywords?", key="btn_kw"):
                            response = analyzer.ai_assistant.get_response("What keywords should I add?")
                            st.session_state.chat_history.append({"role": "user", "message": "What keywords should I add?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})
                    with col4:
                        if st.button("⚡ Improve skills section?", key="btn_sk2"):
                            response = analyzer.ai_assistant.get_response("How can I improve my skills section?")
                            st.session_state.chat_history.append({"role": "user", "message": "How can I improve my skills section?"})
                            st.session_state.chat_history.append({"role": "assistant", "message": response})

                    st.divider()
                    st.markdown("### 💬 Ask Me Anything")
                    user_question = st.text_input("Ask about your resume:", placeholder="How can I improve my resume?", key="chat_input")
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        send_clicked = st.button("Send 💬", type="primary", use_container_width=True, key="send_chat")
                    with col2:
                        if st.button("Clear Chat 🗑️", use_container_width=True, key="clear_chat"):
                            st.session_state.chat_history = []
                            st.rerun()

                    if send_clicked and user_question.strip():
                        context = f"ATS Score: {results['ats_score']:.1f}/100, Role Match: {results['role_match_percentage']:.1f}%, Technical Skills: {len(results['technical_skills'])}, Soft Skills: {len(results['soft_skills'])}"
                        response = analyzer.ai_assistant.get_response(user_question, context)
                        st.session_state.chat_history.append({"role": "user", "message": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "message": response})
                        st.rerun()

                    # Display chat history (last 12 messages)
                    if st.session_state.chat_history:
                        st.markdown("### 📱 Conversation History")
                        # we will display last 12 entries reversed (most recent last)
                        history_to_show = st.session_state.chat_history[-12:]
                        for chat in history_to_show:
                            if chat["role"] == "user":
                                st.markdown(f"""
                                <div class="chat-message chat-user fade-in-up">
                                    <strong>👤 You:</strong>
                                    <div style="margin-top:0.5rem;">{chat["message"]}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                # render assistant message (numbered + colored)
                                render_ai_message(chat["message"])

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.error("Please check your resume format and try again.")
    else:
        st.markdown('<div class="section-header-modern fade-in-up">🚀 Getting Started</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="modern-card fade-in-up">
                <h3 style="color: #334155; margin-bottom: 1rem;">✨ Analysis Features</h3>
                <ul style="color: #64748b; line-height: 1.8;">
                    <li>🎯 ATS Compatibility Scoring</li>
                    <li>🔍 Skills Detection & Matching</li>
                    <li>🔑 Keyword Optimization</li>
                    <li>📊 Professional Visualizations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="modern-card fade-in-up">
                <h3 style="color: #334155; margin-bottom: 1rem;">📄 Supported Formats</h3>
                <ul style="color: #64748b; line-height: 1.8;">
                    <li>📄 PDF Documents</li>
                    <li>📄 DOCX Files</li>
                    <li>📄 TXT Files</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------
# Main
# ---------------------------
def main():
    db = DatabaseManager()
    db.create_tables()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None

    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()
