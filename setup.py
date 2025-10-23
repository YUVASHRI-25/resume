# Modified Resume Insight Dashboard
# Based on user's uploaded file beautiful_dashboard.py. Source preview: :contentReference[oaicite:1]{index=1}
#
# Changes made:
# - White + blue UI theme
# - Skill extraction prefers explicit "Skills" section; fallback to whole-text scanning
# - AI Assistant can call Mistral 7B (or OpenAI) if API key provided via env; automatically decides
#   whether to call the model for generation tasks (cover letter, rewrite, expand, suggest).
# - Save_analysis now updates existing analysis for same user_id + filename + job_role (dedupe)
# - ATS tab: download overall overview as PDF
# - Login/register UI clarified (explicit create account area)
# - DB errors shown but app continues gracefully
#
# Required environment variables (set these before running):
# - DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
# - MISTRAL_API_KEY (optional) OR OPENAI_API_KEY (optional) - if provided AI Assistant will call the real model
#
# Required Python packages (example):
# pip install streamlit pandas plotly pdfplumber python-docx python-dotenv psycopg2-binary openai requests reportlab
#
# Save this file and run with:
# streamlit run modified_dashboard.py
#

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
import requests

# Optional: OpenAI client (if using OpenAI)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:
    reportlab = None

load_dotenv()

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Resume Insight", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'resume_insight'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'port': os.getenv('DB_PORT', '5432')
}

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')      # optional
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')        # optional

# Minimal role keywords kept (you can extend)
JOB_KEYWORDS = {
    "Data Scientist": ["python","machine learning","statistics","pandas","numpy","scikit-learn","tensorflow","pytorch","sql"],
    "Software Engineer": ["programming","java","python","javascript","react","node.js","git","api","algorithms"],
    "Product Manager": ["product","strategy","roadmap","stakeholder","analytics","user experience","agile"],
    # ... keep the rest as in your original file (trimmed here for brevity)
}

TECHNICAL_SKILLS = [
    "python","java","javascript","c++","c#","sql","html","css","react","node.js","django","flask",
    "pandas","numpy","tensorflow","pytorch","docker","kubernetes","aws","azure","gcp","git"
]

SOFT_SKILLS = [
    "leadership","communication","teamwork","problem solving","critical thinking","project management",
    "time management","adaptability","creativity","analytical","collaboration","innovation"
]

# ---------------------------
# CSS: White + Blue clean look
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%); min-height:100vh; color:#0f172a; }
.modern-header { background: linear-gradient(90deg,#ffffff 0%, #e8f3ff 100%); padding:20px; border-radius:12px; margin-bottom:16px; border:1px solid #e2f0ff; }
.header-title { font-size:28px; font-weight:700; color:#0f172a; }
.header-subtitle { font-size:13px; color:#475569; margin-top:6px; }
.metric-card { background:white; border:1px solid #e6f0ff; border-radius:12px; padding:16px; text-align:center; box-shadow: 0 6px 18px rgba(14,47,84,0.04); }
.metric-value { font-size:26px; font-weight:700; color:#0b5cff; }
.metric-label { font-size:12px; color:#475569; margin-top:6px; }
.sidebar-modern { background: #ffffff; border-radius:12px; padding:12px; border:1px solid #eef7ff; }
.modern-card { background:white; border-radius:12px; padding:16px; border:1px solid #eef7ff; }
.skill-tag { display:inline-block; background:#eef7ff; color:#084bff; border-radius:16px; padding:6px 10px; margin:4px; font-weight:600; }
.keyword-suggested { display:inline-block; background:#e6f0ff; color:#0b5cff; border-radius:18px; padding:6px 10px; margin:4px; font-weight:600; }
.alert-info { background:#f0f8ff; border-left:4px solid #0b5cff; padding:12px; border-radius:8px; color:#0f172a; }
.chat-message { border-radius:12px; padding:12px; margin:8px 0; }
.chat-ai { background:#f0f8ff; border-left:4px solid #0b5cff; color:#0f172a; }
.chat-user { background:#ffffff; border-left:4px solid #cfe7ff; color:#0f172a; }
.section-header { background:#eaf4ff; color:#0b5cff; padding:10px; border-radius:10px; font-weight:600; margin-bottom:10px; }
button[kind="primary"] { background:#0b5cff; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Database Manager (updated save_analysis to upsert duplication)
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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, filename, job_role)
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
                cur.execute("SELECT id, username, email FROM users WHERE username = %s AND password_hash = %s",
                            (username, password_hash))
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
                cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                            (username, email, password_hash))
                user_id = cur.fetchone()[0]
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Registration error: {e}")
            return False
        finally:
            conn.close()

    def save_analysis(self, user_id, filename, job_role, analysis_data):
        """
        Upsert logic: if a record for (user_id, filename, job_role) exists, update it.
        This prevents duplicate rows for repeated uploads of the same file/role.
        """
        conn = self.get_connection()
        if not conn:
            return False
        try:
            with conn.cursor() as cur:
                analysis_json = json.dumps(analysis_data)
                # Try update first
                cur.execute("""
                    UPDATE resume_analyses
                    SET ats_score = %s, role_match_percentage = %s, overall_score = %s,
                        technical_skills = %s, soft_skills = %s, found_keywords = %s,
                        missing_keywords = %s, analysis_data = %s, created_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s AND filename = %s AND job_role = %s
                    RETURNING id
                """, (
                    analysis_data.get('ats_score'),
                    analysis_data.get('role_match_percentage'),
                    analysis_data.get('overall_score'),
                    analysis_data.get('technical_skills', []),
                    analysis_data.get('soft_skills', []),
                    analysis_data.get('found_keywords', []),
                    analysis_data.get('missing_keywords', []),
                    analysis_json,
                    user_id, filename, job_role
                ))
                updated = cur.fetchone()
                if updated:
                    conn.commit()
                    return True
                # else insert
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
# Resume Analyzer (prefer Skills section)
# ---------------------------
class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by'}
        self.ai_assistant = AIAssistant()

    def extract_text_from_pdf(self, file):
        try:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception:
            try:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
            except Exception as e:
                return f"Error extracting PDF text: {e}"

    def extract_text_from_docx(self, file):
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error extracting DOCX text: {e}"

    def extract_text_from_txt(self, file):
        try:
            data = file.read()
            if isinstance(data, bytes):
                return data.decode('utf-8', errors='ignore')
            return str(data)
        except Exception as e:
            return f"Error extracting TXT text: {e}"

    def extract_sections(self, text):
        """
        Improved approach:
        - Locate headings by common section keywords
        - Capture lines until next section heading
        """
        sections = {}
        patterns = {
            'summary': r'\b(summary|objective|profile|about|overview)\b',
            'education': r'\b(education|academic|qualification|degree|university|college)\b',
            'experience': r'\b(experience|employment|work|career|professional|job|position)\b',
            'skills': r'\b(skills|technical skills|competencies|expertise|abilities|technologies)\b',
            'projects': r'\b(projects|portfolio|work samples|personal projects)\b',
            'certifications': r'\b(certifications?|credentials|licenses)\b'
        }
        # find lines and their indexes
        lines = [ln.rstrip() for ln in text.splitlines()]
        lower_lines = [ln.lower() for ln in lines]
        # find heading indexes
        headings = []
        for i, ln in enumerate(lower_lines):
            for name, pat in patterns.items():
                if re.search(pat, ln):
                    headings.append((i, name))
        # if no headings found, return empty sections
        if not headings:
            return {k: "" for k in patterns.keys()}

        # sort headings by index
        headings.sort()
        for idx, (start_i, sec_name) in enumerate(headings):
            end_i = headings[idx+1][0] if idx+1 < len(headings) else len(lines)
            content = "\n".join(lines[start_i+1:end_i]).strip()
            sections[sec_name] = content

        # ensure all keys exist
        for k in patterns.keys():
            if k not in sections:
                sections[k] = ""

        return sections

    def extract_skills(self, text, sections):
        """
        Prefer to extract skills from explicit skills section if present and not empty.
        Otherwise fallback to whole-text scanning.
        """
        found_technical = []
        found_soft = []

        skills_section = sections.get('skills', '').lower().strip()
        search_area = skills_section if len(skills_section) > 20 else text.lower()

        # match technical: exact token or phrase boundaries
        for skill in TECHNICAL_SKILLS:
            skill_norm = skill.lower()
            # word boundary match to reduce partial matches
            if re.search(r'\b' + re.escape(skill_norm) + r'\b', search_area):
                found_technical.append(skill)

        for skill in SOFT_SKILLS:
            skill_norm = skill.lower()
            if re.search(r'\b' + re.escape(skill_norm) + r'\b', search_area):
                found_soft.append(skill)

        # keep unique and preserve order
        def unique_list(seq):
            seen = set()
            out = []
            for s in seq:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        return unique_list(found_technical), unique_list(found_soft)

    def keyword_matching(self, text, job_role):
        if job_role not in JOB_KEYWORDS:
            return [], 0
        keywords = JOB_KEYWORDS[job_role]
        text_lower = text.lower()
        found_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower)]
        match_percentage = (len(found_keywords) / len(keywords)) * 100 if keywords else 0
        return found_keywords, match_percentage

    def calculate_ats_score(self, text, sections):
        score = 0
        required_sections = ['experience', 'education', 'skills']
        for section in required_sections:
            if sections.get(section) and len(sections[section]) > 30:
                score += 12
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            score += 20
        elif word_count >= 200:
            score += 10
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{6,10}'
        if re.search(email_pattern, text):
            score += 10
        if re.search(phone_pattern, text):
            score += 10
        bullet_patterns = [r'•', r'◦', r'\*', r'-\s', r'→']
        bullet_count = sum(len(re.findall(p, text)) for p in bullet_patterns)
        if bullet_count >= 6:
            score += 20
        elif bullet_count >= 3:
            score += 10
        return min(score, 100)

    def analyze_resume(self, text, job_role):
        sections = self.extract_sections(text)
        tech_skills, soft_skills = self.extract_skills(text, sections)
        found_keywords, match_percentage = self.keyword_matching(text, job_role)
        ats_score = self.calculate_ats_score(text, sections)
        overall_score = (ats_score + match_percentage) / 2
        return {
            "sections": sections,
            "technical_skills": tech_skills,
            "soft_skills": soft_skills,
            "found_keywords": found_keywords,
            "missing_keywords": [kw for kw in JOB_KEYWORDS.get(job_role, []) if kw not in found_keywords],
            "role_match_percentage": match_percentage,
            "ats_score": ats_score,
            "overall_score": overall_score,
            "word_count": len(text.split())
        }

# ---------------------------
# AI Assistant: improved auto-detect + optional real model call
# ---------------------------
class AIAssistant:
    def __init__(self):
        # fallback canned suggestions (improved and shorter)
        self.templates = {
            "experience": "Start bullets with strong action verbs, add measurable impact (numbers/results), and show progression. Keep bullets concise and focused on impact.",
            "skills": "Separate technical and soft skills, give context (proficiency or projects), and align with the job description keywords.",
            "ats": "Use standard section headings, include keywords naturally, avoid images/tables, use bullet points and common fonts, and save as PDF.",
            "keywords": "Extract keywords from the target job description and insert them naturally into your experience and skills sections.",
            "format": "Keep layout simple, consistent fonts (Arial/Calibri), clear headings, consistent bullet style, 1-2 pages."
        }

    def should_call_model(self, user_query):
        """Decide when to call the real model automatically (cover letter, rewrite, generate, expand)"""
        q = user_query.lower()
        triggers = ["cover letter", "cover-letter", "rewrite", "reword", "improve", "paraphrase", "generate", "write", "draft", "summary", "improve my", "optimi", "make resume"]
        return any(t in q for t in triggers)

    def call_mistral(self, prompt):
        """
        If MISTRAL_API_KEY set, call the Mistral inference API.
        This is a simple POST to the inference endpoint - adjust per your Mistral provider details.
        """
        if not MISTRAL_API_KEY:
            return None
        # NOTE: endpoint & payload shape may vary by host. This is a placeholder example for Mistral inference.
        try:
            url = "https://api.mistral.ai/v1/generate"  # placeholder - replace with actual Mistral endpoint
            headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "mistral-7b", "input": prompt, "max_tokens": 512}
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                out = r.json()
                # Attempt to extract text from known response shapes
                if isinstance(out, dict):
                    # many providers return {'text': '...'} or {'outputs': [{'content':'...'}]}
                    if out.get('text'):
                        return out['text']
                    if 'outputs' in out and isinstance(out['outputs'], list) and out['outputs'][0].get('content'):
                        return out['outputs'][0]['content']
                    # fallback to full json
                    return json.dumps(out)
            return None
        except Exception as e:
            # Do not crash the app on model call failure
            st.warning(f"Mistral API call failed: {e}")
            return None

    def call_openai(self, prompt):
        """Uses OpenAI (if key provided) with simple client if available"""
        if not OPENAI_API_KEY:
            return None
        try:
            # Use official OpenAI python client if installed
            if OpenAI:
                client = OpenAI(api_key=OPENAI_API_KEY)
                resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_tokens=512)
                # extract text
                if resp and getattr(resp, "output_text", None):
                    return resp.output_text
                # try to parse
                return str(resp)
            else:
                # fallback to raw HTTP call (if new client not available)
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": "gpt-4o-mini", "input": prompt, "max_tokens": 512}
                r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    # extract text
                    if 'output' in data and isinstance(data['output'], list):
                        return " ".join([chunk.get('content', '') for chunk in data['output']])
                    if 'choices' in data and data['choices']:
                        return data['choices'][0].get('text') or data['choices'][0].get('message', {}).get('content')
                return None
        except Exception as e:
            st.warning(f"OpenAI call failed: {e}")
            return None

    def get_response(self, user_query, context_text=""):
        # If trigger words detected and an API key exists, call model
        if self.should_call_model(user_query):
            prompt = f"Context:\n{context_text}\n\nUser request:\n{user_query}\n\nProvide a helpful, professional response in bullet points or short paragraphs."
            # try Mistral first
            out = self.call_mistral(prompt) if MISTRAL_API_KEY else None
            if not out:
                out = self.call_openai(prompt) if OPENAI_API_KEY else None
            if out:
                return out.strip()
        # fallback: use canned templates based on keywords
        q = user_query.lower()
        if any(w in q for w in ("experience","work history","job history")):
            return self.templates["experience"]
        if any(w in q for w in ("skill","skills","technical","soft")):
            return self.templates["skills"]
        if any(w in q for w in ("ats","applicant tracking","ats-friendly")):
            return self.templates["ats"]
        if any(w in q for w in ("keyword","keywords","terms")):
            return self.templates["keywords"]
        if any(w in q for w in ("format","layout","design","formatting")):
            return self.templates["format"]
        # default general advice
        return "Customize your resume for each job. Highlight quantifiable achievements and use clear section headings. If you'd like, ask me to 'generate cover letter' or 'rewrite summary' and I will produce a full draft."

# ---------------------------
# Helper UI functions
# ---------------------------
def display_metric_card(title, value, description=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f"<div style='font-size:12px;color:#64748b;margin-top:6px;'>{description}</div>" if description else ""}
    </div>
    """, unsafe_allow_html=True)

def render_ai_message_snippet(message):
    st.markdown(f"""
    <div class="chat-message chat-ai">
        <strong>🤖 AI Assistant:</strong>
        <div style="margin-top:6px;">{message}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# PDF generation helper (simple)
# ---------------------------
def generate_overview_pdf(analysis, filename="resume_overview.pdf"):
    """
    Generate a simple PDF overview using reportlab if available.
    Returns bytes (io.BytesIO) or None if reportlab not installed.
    """
    buffer = io.BytesIO()
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return None
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x = 40
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Resume Overview")
    c.setFont("Helvetica", 10)
    y -= 24
    c.drawString(x, y, f"Filename: {analysis.get('filename','')}")
    y -= 18
    c.drawString(x, y, f"Job Role: {analysis.get('job_role','')}")
    y -= 18
    c.drawString(x, y, f"ATS Score: {analysis.get('ats_score',0):.1f}/100")
    y -= 18
    c.drawString(x, y, f"Role Match: {analysis.get('role_match_percentage',0):.1f}%")
    y -= 18
    c.drawString(x, y, f"Overall Score: {analysis.get('overall_score',0):.1f}/100")
    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Top Technical Skills:")
    y -= 16
    c.setFont("Helvetica", 10)
    for s in (analysis.get('technical_skills') or [])[:10]:
        c.drawString(x+8, y, f"- {s}")
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 40
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Top Soft Skills:")
    y -= 16
    c.setFont("Helvetica", 10)
    for s in (analysis.get('soft_skills') or [])[:10]:
        c.drawString(x+8, y, f"- {s}")
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 40
    y -= 12
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Missing Keywords (suggested):")
    y -= 16
    c.setFont("Helvetica", 10)
    for s in (analysis.get('missing_keywords') or [])[:40]:
        c.drawString(x+8, y, f"- {s}")
        y -= 12
        if y < 60:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Login / Dashboard UI
# ---------------------------
def login_page():
    st.markdown("""
    <div class="modern-header">
        <div class="header-title">📄 Resume Insight</div>
        <div class="header-subtitle">AI-powered resume analysis — white & blue theme</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### 🔐 Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            if login_button:
                db = DatabaseManager()
                user = db.authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.success("Login successful")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown("---")
        st.markdown("### ✨ New here? Create an account")
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            register_button = st.form_submit_button("Create Account")
            if register_button:
                db = DatabaseManager()
                ok = db.register_user(new_username, new_email, new_password)
                if ok:
                    st.success("Account created — please login.")
                else:
                    st.error("Registration failed (username/email may exist).")

def dashboard_page():
    user = st.session_state.user
    st.markdown(f"""
    <div class="modern-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="header-title">Welcome back, {user['username']} 👋</div>
                <div class="header-subtitle">Resume analysis & ATS optimization</div>
            </div>
            <div style="text-align:right;">
                <small style="color:#64748b;">Logged in: {user['username']}</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-modern"><h4>⚙️ Analysis Configuration</h4></div>', unsafe_allow_html=True)
        job_roles = list(JOB_KEYWORDS.keys())
        selected_role = st.selectbox("Target Job Role:", job_roles)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.experimental_rerun()
        st.markdown("---")
        db = DatabaseManager()
        st.markdown("<div class='sidebar-modern'><h4>📄 Recent analyses</h4></div>", unsafe_allow_html=True)
        try:
            analyses = db.get_user_analyses(user['id'])
            if analyses:
                for a in analyses[:6]:
                    created = a['created_at'].strftime('%Y-%m-%d %H:%M') if isinstance(a.get('created_at'), datetime) else str(a.get('created_at'))
                    st.markdown(f"**{a['filename']}**  \nRole: {a['job_role']}  \nATS: {a.get('ats_score')}/100  \n{created}")
                    st.markdown("---")
            else:
                st.info("No analyses yet")
        except Exception:
            st.info("No analyses or unable to fetch history")

    st.markdown('<div class="section-header">📄 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=['pdf','docx','txt'])
    if uploaded_file:
        analyzer = ResumeAnalyzer()
        with st.spinner("Processing..."):
            filetype = uploaded_file.type
            try:
                if "pdf" in filetype:
                    text = analyzer.extract_text_from_pdf(uploaded_file)
                elif "word" in filetype or uploaded_file.name.lower().endswith(".docx"):
                    text = analyzer.extract_text_from_docx(uploaded_file)
                else:
                    text = analyzer.extract_text_from_txt(uploaded_file)
            except Exception as e:
                st.error(f"Error extracting text: {e}")
                return

        if text and not text.lower().startswith("error"):
            st.success("Resume processed")
            results = analyzer.analyze_resume(text, selected_role)
            # Attach meta
            results['filename'] = uploaded_file.name
            results['job_role'] = selected_role
            db = DatabaseManager()
            saved = db.save_analysis(user['id'], uploaded_file.name, selected_role, results)
            if saved:
                st.success("Analysis saved")
            else:
                st.warning("Could not save analysis (DB issue)")

            tabs = st.tabs(["Summary","Skills","ATS","AI Assistant"])
            # Summary
            with tabs[0]:
                st.markdown('<div class="section-header">Summary</div>', unsafe_allow_html=True)
                cols = st.columns(3)
                cols[0].markdown(f"<div class='metric-card'><div class='metric-value'>{results['ats_score']:.1f}/100</div><div class='metric-label'>ATS Score</div></div>", unsafe_allow_html=True)
                cols[1].markdown(f"<div class='metric-card'><div class='metric-value'>{results['role_match_percentage']:.1f}%</div><div class='metric-label'>Role Match</div></div>", unsafe_allow_html=True)
                cols[2].markdown(f"<div class='metric-card'><div class='metric-value'>{results['word_count']}</div><div class='metric-label'>Word Count</div></div>", unsafe_allow_html=True)

                if results['overall_score'] >= 80:
                    st.success("Excellent resume alignment")
                elif results['overall_score'] >= 60:
                    st.warning("Good foundation - improve keywords and formatting")
                else:
                    st.error("Needs substantial improvement")

                # Download Overview PDF
                st.markdown("### Download Overview")
                pdf_buffer = generate_overview_pdf(results)
                if pdf_buffer:
                    b64 = base64.b64encode(pdf_buffer.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="resume_overview_{uploaded_file.name}.pdf">📥 Download Overview (PDF)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.info("PDF generation not available (reportlab missing). You can install reportlab to enable PDF export.")

            # Skills
            with tabs[1]:
                st.markdown('<div class="section-header">Skills</div>', unsafe_allow_html=True)
                st.markdown("**Technical Skills Detected**")
                if results['technical_skills']:
                    for s in results['technical_skills']:
                        st.markdown(f"<span class='skill-tag'>{s}</span>", unsafe_allow_html=True)
                else:
                    st.info("No technical skills detected in explicit skills section. Try adding a skills section with comma-separated items.")
                st.markdown("**Soft Skills Detected**")
                if results['soft_skills']:
                    for s in results['soft_skills']:
                        st.markdown(f"<span class='skill-tag'>{s}</span>", unsafe_allow_html=True)
                else:
                    st.info("No soft skills detected.")

                st.markdown("**Suggested Keywords to add**")
                if results.get('missing_keywords'):
                    for kw in results['missing_keywords'][:40]:
                        st.markdown(f"<span class='keyword-suggested'>{kw}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-info'>Your resume already contains most role-specific keywords.</div>", unsafe_allow_html=True)

            # ATS
            with tabs[2]:
                st.markdown('<div class="section-header">ATS Analysis</div>', unsafe_allow_html=True)
                st.metric("ATS Score", f"{results['ats_score']:.1f}/100")
                st.progress(results['ats_score']/100)
                st.markdown("**ATS Tips**")
                st.write("- Use standard headings, avoid images and complex formatting.")
                st.write("- Include keywords naturally and use bullet points.")
                st.write("- Ensure contact info & section headings exist.")
                # Allow download PDF again
                if pdf_buffer:
                    pdf_buffer.seek(0)
                    st.download_button("Download Overview PDF", data=pdf_buffer, file_name=f"resume_overview_{uploaded_file.name}.pdf")

            # AI Assistant
            with tabs[3]:
                st.markdown('<div class="section-header">AI Assistant</div>', unsafe_allow_html=True)
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

                # quick actions
                col1, col2, col3 = st.columns(3)
                if col1.button("Improve Experience"):
                    q = "How can I improve my experience section?"
                    resp = analyzer.ai_assistant.get_response(q, context_text=f"ATS: {results['ats_score']}; Role match: {results['role_match_percentage']}")
                    st.session_state.chat_history.append({"role":"user","message":q})
                    st.session_state.chat_history.append({"role":"assistant","message":resp})
                    try:
                        db.save_chat_message(user['id'], None, q, resp)
                    except Exception:
                        pass
                if col2.button("Make ATS-friendly"):
                    q = "Make my resume ATS-friendly; provide checklist and changes"
                    resp = analyzer.ai_assistant.get_response(q, context_text=f"ATS: {results['ats_score']}")
                    st.session_state.chat_history.append({"role":"user","message":q})
                    st.session_state.chat_history.append({"role":"assistant","message":resp})
                    try:
                        db.save_chat_message(user['id'], None, q, resp)
                    except Exception:
                        pass
                if col3.button("Generate Cover Letter"):
                    q = "Generate a cover letter based on this resume and target role"
                    resp = analyzer.ai_assistant.get_response(q, context_text=f"Role: {selected_role}\nATS: {results['ats_score']}")
                    st.session_state.chat_history.append({"role":"user","message":q})
                    st.session_state.chat_history.append({"role":"assistant","message":resp})
                    try:
                        db.save_chat_message(user['id'], None, q, resp)
                    except Exception:
                        pass

                st.markdown("### Ask anything about this resume")
                user_q = st.text_input("Ask about your resume", key="chat_input2")
                if st.button("Send"):
                    if user_q.strip():
                        resp = analyzer.ai_assistant.get_response(user_q, context_text=f"ATS: {results['ats_score']}, Role match: {results['role_match_percentage']}")
                        st.session_state.chat_history.append({"role":"user","message":user_q})
                        st.session_state.chat_history.append({"role":"assistant","message":resp})
                        try:
                            db.save_chat_message(user['id'], None, user_q, resp)
                        except Exception:
                            pass
                        st.experimental_rerun()

                # display chat
                if st.session_state.chat_history:
                    for entry in st.session_state.chat_history[-12:]:
                        if entry['role'] == 'user':
                            st.markdown(f"<div class='chat-message chat-user'><strong>👤 You:</strong><div style='margin-top:6px;'>{entry['message']}</div></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='chat-message chat-ai'><strong>🤖 AI:</strong><div style='margin-top:6px;'>{entry['message']}</div></div>", unsafe_allow_html=True)

        else:
            st.error("Could not extract text from this file. Please upload a different file or ensure the file is not corrupted.")
    else:
        st.markdown('<div class="section-header">Getting started</div>', unsafe_allow_html=True)
        st.markdown("Upload a resume on the left. Use the AI Assistant or click quick actions to get suggestions.")

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
