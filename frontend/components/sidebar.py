"""
Sidebar component for Resume Insight application.
"""

import streamlit as st
import sys
import os

# Add the parent directory to Python path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from app.config import JOB_KEYWORDS
except ImportError:
    # Fallback job keywords if app.config is not available
    JOB_KEYWORDS = {
        "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy"],
        "Software Engineer": ["programming", "java", "python", "javascript", "react"],
        "Product Manager": ["product", "strategy", "roadmap", "stakeholder", "analytics"],
        "Marketing Manager": ["marketing", "digital marketing", "seo", "social media"],
        "Data Analyst": ["sql", "excel", "python", "tableau", "power bi"],
        "DevOps Engineer": ["docker", "kubernetes", "aws", "azure", "jenkins"],
        "UI/UX Designer": ["figma", "sketch", "adobe xd", "prototyping", "wireframes"],
        "Cybersecurity Analyst": ["security", "penetration testing", "vulnerability assessment"],
        "Business Analyst": ["requirements gathering", "process improvement", "stakeholder management"],
        "Full Stack Developer": ["html", "css", "javascript", "react", "angular"],
        "Machine Learning Engineer": ["python", "tensorflow", "pytorch", "scikit-learn"],
        "AI Engineer": ["artificial intelligence", "machine learning", "deep learning"]
    }


def create_sidebar():
    """Create the sidebar with configuration options."""
    st.header("Analysis Configuration")
    
    # Job role selection
    job_roles = list(JOB_KEYWORDS.keys())
    selected_role = st.selectbox(
        "Target Job Role:",
        job_roles,
        index=job_roles.index(st.session_state.get('selected_job_role', 'Software Engineer')),
        help="Select the job role you're targeting to get relevant keyword analysis"
    )
    
    # Update session state
    st.session_state.selected_job_role = selected_role
    
    st.divider()
    
    # User email input
    user_email = st.text_input(
        "Your Email:",
        value=st.session_state.get('user_email', ''),
        help="Used to save your analysis results"
    )
    st.session_state.user_email = user_email
    
    st.divider()
    
    # Analysis features
    st.header("Analysis Features")
    st.markdown("""
    **Comprehensive Analysis:**
    - ATS Compatibility Score
    - Skills Detection & Matching
    - Section Structure Analysis
    - Grammar & Language Check
    - Keyword Optimization
    - PDF Report Generation
    
    **AI Assistant:**
    - Personalized Resume Advice
    - Quick Expert Recommendations
    - Interactive Q&A
    """)
    
    st.divider()
    
    # Quick actions
    st.header("Quick Actions")
    
    if st.button("🔄 Re-analyze Resume", use_container_width=True):
        if st.session_state.resume_id:
            st.session_state.analysis_results = None
            st.session_state.ai_recommendations = None
            st.rerun()
        else:
            st.warning("Please upload a resume first")
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.resume_id = None
        st.session_state.analysis_results = None
        st.session_state.ai_recommendations = None
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    # System status
    st.header("System Status")
    
    # Check API connectivity
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Error")
    except:
        st.error("❌ Backend Offline")
    
    # LLM Status
    try:
        response = requests.get("http://localhost:8000/api/llm/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["configuration"]["initialized"]:
                st.success("✅ AI Assistant Ready")
            else:
                st.warning("⚠️ AI Assistant Initializing")
        else:
            st.warning("⚠️ AI Assistant Unavailable")
    except:
        st.warning("⚠️ AI Assistant Offline")
    
    # RAG Status
    try:
        response = requests.get("http://localhost:8000/api/rag/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["configuration"]["initialized"]:
                st.success("✅ RAG System Ready")
            else:
                st.warning("⚠️ RAG System Initializing")
        else:
            st.warning("⚠️ RAG System Unavailable")
    except:
        st.warning("⚠️ RAG System Offline")
