"""
Beautiful Modern Resume Insight Dashboard with Professional Design
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

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Insight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Modern CSS with Professional Colors
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #cbd5e1 100%);
    min-height: 100vh;
}

.main {
    padding: 0;
}

/* Beautiful Streamlit Overrides */
.stApp > div {
    background: transparent;
}

.stApp > div > div {
    background: transparent;
}

/* Beautiful Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid #e2e8f0;
}

/* Beautiful Main Content */
.css-1v0mbdj {
    background: transparent;
}

/* Beautiful Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white;
    border-radius: 16px;
    padding: 0.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

/* Beautiful File Uploader */
.stFileUploader > div > div {
    background: white;
    border: 2px dashed #3b82f6;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

.stFileUploader > div > div:hover {
    border-color: #1e40af;
    background: #f8fafc;
    transform: scale(1.01);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}

/* Beautiful Success Messages */
.stSuccess {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #22c55e;
    border-radius: 12px;
    color: #166534;
}

/* Beautiful Warning Messages */
.stWarning {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    color: #92400e;
}

/* Beautiful Error Messages */
.stError {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    border: 1px solid #ef4444;
    border-radius: 12px;
    color: #991b1b;
}

/* Beautiful Info Messages */
.stInfo {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #3b82f6;
    border-radius: 12px;
    color: #1e40af;
}

/* Beautiful Modern Header */
.modern-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(30, 64, 175, 0.15);
    position: relative;
    overflow: hidden;
}

.modern-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23dots)"/></svg>');
    pointer-events: none;
}

.header-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.5rem;
    text-shadow: 0 4px 8px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}

.header-subtitle {
    font-size: 1.3rem;
    color: rgba(255,255,255,0.95);
    font-weight: 500;
    position: relative;
    z-index: 1;
}

/* Beautiful Modern Cards */
.modern-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.modern-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #06b6d4);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.modern-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}

/* Beautiful Modern Metrics */
.metric-card {
    background: white;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    border: 3px solid #f1f5f9;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #06b6d4, #10b981);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
    border-color: #3b82f6;
}

.metric-value {
    font-size: 3.5rem;
    font-weight: 900;
    color: #1e40af;
    margin-bottom: 0.8rem;
    text-shadow: 0 2px 4px rgba(30, 64, 175, 0.2);
    position: relative;
    z-index: 1;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.02em;
}

.metric-label {
    font-size: 1.1rem;
    color: #64748b;
    font-weight: 700;
    position: relative;
    z-index: 1;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Beautiful Modern Skill Tags */
.skill-tag {
    display: inline-block;
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    font-size: 0.875rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.2);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.skill-tag::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.skill-tag:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 8px 25px rgba(30, 64, 175, 0.3);
}

.skill-tag:hover::before {
    left: 100%;
}

/* Beautiful Modern Alerts */
.alert-modern {
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin: 1rem 0;
    border-left: 4px solid;
    position: relative;
    overflow: hidden;
}

.alert-success {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left-color: #22c55e;
    color: #166534;
}

.alert-warning {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left-color: #f59e0b;
    color: #92400e;
}

.alert-error {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
    border-left-color: #ef4444;
    color: #991b1b;
}

.alert-info {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left-color: #3b82f6;
    color: #1e40af;
}

/* Beautiful Modern Sidebar */
.sidebar-modern {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

/* Beautiful Modern Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 6px 20px rgba(30, 64, 175, 0.2);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(30, 64, 175, 0.3);
}

.stButton > button:hover::before {
    left: 100%;
}

/* Beautiful Modern File Uploader */
.stFileUploader > div {
    background: white;
    border: 2px dashed #3b82f6;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

.stFileUploader > div:hover {
    border-color: #1e40af;
    background: #f8fafc;
    transform: scale(1.01);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}

/* Beautiful Modern Tabs */
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 12px 12px 0 0;
    padding: 1rem 2rem;
    margin: 0 0.25rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(30, 64, 175, 0.2);
    border-color: #1e40af;
}

/* Beautiful Modern Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(30, 64, 175, 0.2);
}

/* Beautiful Modern Chat Interface */
.chat-container {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    border: 2px solid #f1f5f9;
}

.chat-message {
    padding: 2rem;
    border-radius: 20px;
    margin: 1rem 0;
    max-width: 85%;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    position: relative;
    overflow: hidden;
    font-family: 'Inter', sans-serif;
}

.chat-message::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #06b6d4);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

.chat-user {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    margin-left: auto;
    border: 2px solid #1e40af;
}

.chat-ai {
    background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
    color: white;
    margin-right: auto;
    border: 2px solid #06b6d4;
    font-size: 1rem;
    line-height: 1.6;
}

.chat-ai strong {
    font-weight: 800;
    font-size: 1.1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.chat-ai ol {
    margin: 1rem 0;
    padding-left: 1.5rem;
}

.chat-ai li {
    margin: 0.5rem 0;
    font-size: 1rem;
    line-height: 1.5;
}

.chat-user strong {
    font-weight: 800;
    font-size: 1.1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Beautiful Modern Login Form */
.login-container {
    background: white;
    border-radius: 20px;
    padding: 3rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    max-width: 450px;
    margin: 2rem auto;
    border: 1px solid rgba(226, 232, 240, 0.8);
    position: relative;
    overflow: hidden;
}

.login-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1e40af, #3b82f6, #06b6d4);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

/* Beautiful Modern Input Fields */
.stSelectbox > div > div {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    transition: all 0.3s ease;
}

.stSelectbox > div > div:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

.stTextInput > div > div > input {
    background: white;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
}

/* Beautiful Modern Section Headers */
.section-header-modern {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 10px 25px rgba(30, 64, 175, 0.2);
    position: relative;
    overflow: hidden;
}

.section-header-modern::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="dots" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="10" cy="10" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23dots)"/></svg>');
    pointer-events: none;
}

/* Beautiful Modern Keywords Section */
.keywords-section {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

.keyword-suggested {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    margin: 0.3rem;
    font-size: 0.875rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    transition: all 0.3s ease;
    display: inline-block;
}

.keyword-suggested:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
}

/* Beautiful Modern AI Assistant */
.ai-assistant-modern {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    border: 1px solid rgba(226, 232, 240, 0.8);
    position: relative;
    overflow: hidden;
}

.ai-assistant-modern::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #06b6d4, #0891b2, #1e40af, #3b82f6);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}

/* Beautiful Modern Quick Actions */
.quick-action {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem;
    font-weight: 600;
    box-shadow: 0 6px 20px rgba(30, 64, 175, 0.2);
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.quick-action:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(30, 64, 175, 0.3);
}

.quick-action::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.quick-action:hover::before {
    left: 100%;
}

/* Beautiful Modern History Cards */
.history-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    border: 1px solid rgba(226, 232, 240, 0.8);
    transition: all 0.3s ease;
}

.history-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

/* Beautiful Modern Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}

/* Beautiful Modern Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(226, 232, 240, 0.5);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
}

/* Beautiful Modern Tables */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

/* Beautiful Modern Charts */
.plotly-chart {
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}

/* Beautiful Modern Responsive */
@media (max-width: 768px) {
    .header-title {
        font-size: 2.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
    }
    
    .modern-card {
        padding: 1.5rem;
    }
}

/* Beautiful Modern Glass Effect */
.glass-effect {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 16px;
}

/* Beautiful Modern Success States */
.success-gradient {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
}

.warning-gradient {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

.error-gradient {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.info-gradient {
    background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
}
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'resume_insight'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'port': os.getenv('DB_PORT', '5432')
}

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


class DatabaseManager:
    """Beautiful modern database manager for PostgreSQL operations."""
    
    def __init__(self):
        self.config = DB_CONFIG
    
    def get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(**self.config)
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    def create_tables(self):
        """Create necessary tables."""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                # Users table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Resume analyses table
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
                
                # Chat history table
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
        """Authenticate user."""
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
        """Register new user."""
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
        """Save resume analysis to database."""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                # Convert analysis_data to JSON string
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
        """Get user's analysis history."""
        conn = self.get_connection()
        if not conn:
            return []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM resume_analyses 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC
                """, (user_id,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Error fetching analyses: {e}")
            return []
        finally:
            conn.close()
    
    def save_chat_message(self, user_id, session_id, message, response):
        """Save chat message to database."""
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_history (user_id, session_id, message, response)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, session_id, message, response))
                conn.commit()
                return True
        except Exception as e:
            st.error(f"Error saving chat message: {e}")
            return False
        finally:
            conn.close()


class ResumeAnalyzer:
    """Beautiful modern resume analyzer."""
    
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.ai_assistant = AIAssistant()  # Initialize AI assistant
    
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


class AIAssistant:
    """Beautiful modern AI Assistant for resume guidance."""
    
    def __init__(self):
        self.responses = {
            "experience": [
                "**🚀 Experience Section Optimization:**\n\n• Use bullet points with action verbs (Led, Managed, Developed)\n• Quantify achievements with numbers (Increased sales by 25%)\n• Focus on results, not just duties\n• Tailor content to match job requirements\n• Use reverse chronological order\n\n**💡 Pro Tip:** Start each bullet with a strong action verb to make your experience stand out!",
                "**⭐ Make Your Experience Stand Out:**\n\n• Start each bullet with a strong action verb\n• Include metrics and measurable results\n• Show progression and growth in your roles\n• Highlight relevant projects and achievements\n• Keep descriptions concise but impactful\n\n**🎯 Focus:** Show how you added value, not just what you did!"
            ],
            "skills": [
                "**🎯 Skills Section Optimization:**\n\n1. Organize technical and soft skills separately\n2. Match skills to job description requirements\n3. Include both hard and soft skills\n4. Provide proficiency levels when appropriate\n5. Update skills regularly to stay current\n\n**💪 Power Skills:** Technical skills + Soft skills = Winning combination!",
                "**⚡ Skills That Get You Hired:**\n\n1. Technical skills relevant to the role\n2. Soft skills like leadership and communication\n3. Industry-specific tools and technologies\n4. Certifications and training\n5. Languages (if relevant to the position)\n\n**🔥 Hot Tip:** Include both technical and soft skills for maximum impact!"
            ],
            "ats": [
                "**🤖 ATS-Friendly Resume Tips:**\n\n• Use standard section headings (Experience, Education, Skills)\n• Include relevant keywords naturally\n• Avoid images, graphics, and complex formatting\n• Use common fonts like Arial or Calibri\n• Save as PDF to preserve formatting\n\n**✅ ATS Checklist:** Standard headings + Keywords + Simple format = ATS success!",
                "**📋 ATS Optimization Checklist:**\n\n• Standard section headings\n• Contact information at the top\n• Bullet points for easy scanning\n• Relevant keywords throughout\n• Simple, clean formatting\n• PDF format preferred\n\n**🎯 Goal:** Make it easy for ATS systems to read and parse your resume!"
            ],
            "keywords": [
                "**🔍 Keyword Strategy:**\n\n• Study job descriptions in your field\n• Use industry-specific terminology\n• Include both acronyms and full terms\n• Incorporate keywords naturally\n• Don't overstuff - keep it readable\n\n**💡 Pro Strategy:** Keywords should flow naturally in your content!",
                "**🎯 Finding the Right Keywords:**\n\n• Look at job postings for your target role\n• Use industry forums and websites\n• Check company websites for terminology\n• Include technical skills and tools\n• Add soft skills relevant to the role\n\n**🚀 Power Move:** Use keywords from the actual job description you're applying for!"
            ],
            "format": [
                "**📄 Resume Formatting Best Practices:**\n\n• Use clear, professional headings\n• Consistent bullet points and spacing\n• Readable fonts (10-12pt)\n• Appropriate white space\n• Clean, professional layout\n\n**✨ Visual Appeal:** Clean formatting = Professional impression!",
                "**🎨 Visual Appeal Tips:**\n\n• Use consistent formatting throughout\n• Choose professional colors (black text on white)\n• Include adequate white space\n• Use bullet points for easy scanning\n• Keep it to 1-2 pages maximum\n\n**💎 Remember:** Your resume is your first impression - make it count!"
            ],
            "general": [
                "**🌟 General Resume Tips:**\n\n• Customize for each job application\n• Proofread carefully for errors\n• Use a professional email address\n• Include a compelling summary\n• Show quantifiable achievements\n\n**🎯 Success Formula:** Customization + Quality + Relevance = Job offers!",
                "**💼 Career Success Tips:**\n\n• Research the company and role\n• Match your experience to job requirements\n• Highlight transferable skills\n• Show career progression\n• Include relevant certifications\n\n**🚀 Pro Tip:** Every resume should tell a story of growth and achievement!"
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
        response = random.choice(self.responses[category])
        
        # Add context if available
        if context:
            response += f"\n\n**📊 Based on your analysis:**\n{context}"
        
        return response


def display_metric_card(title, value, description=""):
    """Display a metric in a beautiful modern card format."""
    st.markdown(f"""
    <div class="metric-card fade-in-up">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f"<small>{description}</small>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def display_alert_box(message, alert_type="info"):
    """Display beautiful modern alert box."""
    st.markdown(f"""
    <div class="alert-modern alert-{alert_type} fade-in-up">
        {message}
    </div>
    """, unsafe_allow_html=True)


def login_page():
    """Display beautiful modern login page."""
    st.markdown("""
    <div class="login-container fade-in-up">
        <h2 style="text-align: center; color: #1e40af; margin-bottom: 2rem; font-weight: 700;">
            📊 Resume Insight Pro
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
    """Display beautiful modern dashboard."""
    user = st.session_state.user
    
    # Beautiful Modern Header
    st.markdown(f"""
    <div class="modern-header fade-in-up">
        <div class="header-title">Welcome back, {user['username']}! 👋</div>
        <div class="header-subtitle">📊 Beautiful Modern Resume Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Beautiful Modern Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-modern fade-in-up">
            <h3 style="color: #1e40af; margin-bottom: 1rem;">⚙️ Analysis Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        job_roles = list(JOB_KEYWORDS.keys())
        selected_role = st.selectbox(
            "🎯 Target Job Role:",
            job_roles,
            help="Select the job role you're targeting"
        )
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()
        
        # Analysis History
        st.markdown("""
        <div class="sidebar-modern fade-in-up">
            <h3 style="color: #1e40af; margin-bottom: 1rem;">📊 Analysis History</h3>
        </div>
        """, unsafe_allow_html=True)
        
        db = DatabaseManager()
        analyses = db.get_user_analyses(user['id'])
        
        if analyses:
            for analysis in analyses[:5]:  # Show last 5
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
    
    # Main content
    st.markdown('<div class="section-header-modern fade-in-up">📄 Upload Your Resume</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📁 Select your resume file",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if uploaded_file is not None:
        # Show file information
        file_details = {
            "📄 Filename": uploaded_file.name,
            "📊 File size": f"{uploaded_file.size / 1024:.1f} KB",
            "🔧 File type": uploaded_file.type
        }
        
        with st.expander("📋 File Information", expanded=False):
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
        
        # Extract text based on file type
        analyzer = ResumeAnalyzer()
        
        with st.spinner("🔄 Processing your resume..."):
            try:
                if uploaded_file.type == "application/pdf":
                    text = analyzer.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = analyzer.extract_text_from_docx(uploaded_file)
                else:  # txt
                    text = analyzer.extract_text_from_txt(uploaded_file)
            except Exception as e:
                st.error(f"❌ Error extracting text: {str(e)}")
                return
        
        if "Error" not in text and text.strip():
            st.success("✅ Resume processed successfully!")
            
            try:
                # Analyze the resume
                results = analyzer.analyze_resume(text, selected_role)
                
                # Save analysis to database
                db = DatabaseManager()
                if db.save_analysis(user['id'], uploaded_file.name, selected_role, results):
                    st.success("💾 Analysis saved to history!")
                else:
                    st.warning("⚠️ Could not save analysis to history")
                
                # Create tabs for different analysis views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Summary", "🔍 Skills Analysis", 
                    "🎯 ATS Analysis", "🤖 AI Assistant"
                ])
                
                with tab1:
                    st.markdown('<div class="section-header-modern fade-in-up">📊 Resume Summary</div>', unsafe_allow_html=True)
                    
                    # Key metrics
                    metric_cols = st.columns(3)
                    
                    with metric_cols[0]:
                        display_metric_card("ATS Score", f"{results['ats_score']:.1f}/100")
                    
                    with metric_cols[1]:
                        display_metric_card("Role Match", f"{results['role_match_percentage']:.1f}%")
                    
                    with metric_cols[2]:
                        display_metric_card("Word Count", f"{results['word_count']}")
                    
                    
                    # Overall assessment
                    overall_score = results['overall_score']
                    
                    if overall_score >= 80:
                        display_alert_box("🎉 Excellent resume! Your resume shows strong alignment with the target role and good ATS compatibility.", "success")
                    elif overall_score >= 60:
                        display_alert_box("⚠️ Good foundation with room for improvement. Focus on adding more role-specific keywords and optimizing for ATS.", "warning")
                    else:
                        display_alert_box("🚨 Significant improvements needed. Consider restructuring sections, adding relevant keywords, and improving ATS compatibility.", "error")
                
                with tab2:
                    st.markdown('<div class="section-header-modern fade-in-up">🔍 Skills Analysis</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 💻 Technical Skills Detected")
                        tech_skills = results['technical_skills']
                        
                        if tech_skills:
                            tech_html = ""
                            for skill in tech_skills:
                                tech_html += f'<span class="skill-tag">{skill}</span>'
                            st.markdown(tech_html, unsafe_allow_html=True)
                            st.metric("Technical Skills Count", len(tech_skills))
                        else:
                            display_alert_box("⚠️ No technical skills detected. Consider adding a dedicated skills section.", "warning")
                    
                    with col2:
                        st.markdown("### 🤝 Soft Skills Detected")
                        soft_skills = results['soft_skills']
                        
                        if soft_skills:
                            soft_html = ""
                            for skill in soft_skills:
                                soft_html += f'<span class="skill-tag">{skill}</span>'
                            st.markdown(soft_html, unsafe_allow_html=True)
                            st.metric("Soft Skills Count", len(soft_skills))
                        else:
                            display_alert_box("ℹ️ Limited soft skills detected. Consider highlighting leadership, communication, and teamwork skills.", "info")
                    
                    st.divider()
                    
                    # Role-specific analysis
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
                    
                    # Missing keywords - Suggested Keywords Section
                    missing_keywords = results['missing_keywords']
                    
                    if missing_keywords:
                        st.markdown("### 🔑 Suggested Keywords to Add")
                        st.markdown("""
                        <div class="keywords-section fade-in-up">
                            <p style="color: #64748b; margin-bottom: 1rem;">
                                💡 These keywords are commonly found in {selected_role} job descriptions:
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        missing_html = ""
                        for keyword in missing_keywords[:15]:  # Show top 15
                            missing_html += f'<span class="keyword-suggested">{keyword}</span>'
                        st.markdown(missing_html, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="alert-modern alert-info fade-in-up">
                            <strong>💡 Pro Tip:</strong> Try to naturally incorporate these keywords into your resume sections. 
                            Don't just add them randomly - weave them into your experience descriptions and skills sections.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI Assistant in Skills Analysis
                    st.divider()
                    st.markdown("### 🤖 AI Assistant for Skills")
                    
                    st.markdown("""
                    <div class="ai-assistant-modern fade-in-up">
                        <h4 style="color: #1e40af; margin-bottom: 1rem;">💬 Quick AI Questions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick AI questions for skills
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💡 How to improve my skills section?", use_container_width=True):
                            response = analyzer.ai_assistant.get_response("How to improve my skills section?")
                            st.markdown(f"""
                            <div class="chat-message chat-ai fade-in-up">
                                <strong>🤖 AI Assistant:</strong><br>
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("🔍 What keywords should I add?", use_container_width=True):
                            response = analyzer.ai_assistant.get_response("What keywords should I add?")
                            st.markdown(f"""
                            <div class="chat-message chat-ai fade-in-up">
                                <strong>🤖 AI Assistant:</strong><br>
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                    
                
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
                
                with tab4:
                    st.markdown('<div class="section-header-modern fade-in-up">🤖 AI Resume Assistant</div>', unsafe_allow_html=True)
                    
                    # Initialize chat history in session state
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Quick questions
                    st.markdown("### 🚀 Quick Questions")
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
                    st.markdown("### 💬 Ask Me Anything")
                    
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
                        st.markdown("### 📱 Conversation History")
                        
                        for i, chat in enumerate(reversed(st.session_state.chat_history[-6:])):  # Show last 6 messages
                            if chat["role"] == "user":
                                st.markdown(f"""
                                <div class="chat-message chat-user fade-in-up">
                                    <strong>👤 You:</strong> {chat["message"]}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="chat-message chat-ai fade-in-up">
                                    <strong>🤖 AI Assistant:</strong><br>
                                    {chat["message"]}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # AI Assistant info
                    st.divider()
                    st.markdown("""
                    <div class="ai-assistant-modern fade-in-up">
                        <h4 style="color: #1e40af;">🎯 AI Assistant Features:</h4>
                        <ul style="color: #64748b;">
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
                st.error(f"❌ Error during analysis: {str(e)}")
                st.error("Please check your resume format and try again.")
    
    else:
        # Instructions when no file is uploaded
        st.markdown('<div class="section-header-modern fade-in-up">🚀 Getting Started</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="modern-card fade-in-up">
                <h3 style="color: #1e40af; margin-bottom: 1rem;">✨ Analysis Features</h3>
                <ul style="color: #64748b; line-height: 1.8;">
                    <li>🎯 ATS Compatibility Scoring</li>
                    <li>🔍 Skills Detection & Matching</li>
                    <li>🔑 Keyword Optimization</li>
                    <li>📋 Section Structure Analysis</li>
                    <li>🤖 AI-Powered Recommendations</li>
                    <li>📊 Professional Visualizations</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card fade-in-up">
                <h3 style="color: #1e40af; margin-bottom: 1rem;">📄 Supported Formats</h3>
                <ul style="color: #64748b; line-height: 1.8;">
                    <li>📄 PDF Documents</li>
                    <li>📄 DOCX Files</li>
                    <li>📄 TXT Files</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application function."""
    # Initialize database
    db = DatabaseManager()
    db.create_tables()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Check login status
    if not st.session_state.logged_in:
        login_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()
