"""
Application constants and configuration values.
"""

# Application metadata
APP_NAME = "Resume Insight"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-Powered Resume Analysis & Optimization System"

# File processing constants
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg']

# Analysis scoring weights
ATS_WEIGHT = 0.4
ROLE_MATCH_WEIGHT = 0.4
SKILLS_WEIGHT = 0.2

# Score thresholds
EXCELLENT_THRESHOLD = 80
GOOD_THRESHOLD = 60
POOR_THRESHOLD = 40

# API endpoints
API_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
RESUME_ENDPOINT = f"{API_BASE_URL}/api/resume"
LLM_ENDPOINT = f"{API_BASE_URL}/api/llm"
RAG_ENDPOINT = f"{API_BASE_URL}/api/rag"

# UI constants
MAX_SKILLS_DISPLAY = 20
MAX_KEYWORDS_DISPLAY = 15
MAX_CHAT_HISTORY = 10
MAX_RECOMMENDATIONS_DISPLAY = 10

# Chart colors
COLORS = {
    'primary': '#007bff',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Chart color palettes
COLOR_PALETTES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'pastel': ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'],
    'bright': ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
}

# Error messages
ERROR_MESSAGES = {
    'file_too_large': f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB",
    'unsupported_format': f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
    'upload_failed': "Failed to upload file. Please try again.",
    'analysis_failed': "Analysis failed. Please check your file and try again.",
    'api_unavailable': "Backend service is currently unavailable.",
    'invalid_email': "Please enter a valid email address.",
    'no_resume': "Please upload a resume first.",
    'no_analysis': "No analysis results available. Please analyze your resume first."
}

# Success messages
SUCCESS_MESSAGES = {
    'file_uploaded': "Resume uploaded successfully!",
    'analysis_complete': "Analysis completed successfully!",
    'recommendations_generated': "AI recommendations generated!",
    'chat_sent': "Message sent successfully!",
    'data_cleared': "All data cleared successfully!"
}

# Info messages
INFO_MESSAGES = {
    'processing': "Processing your resume...",
    'analyzing': "Analyzing your resume...",
    'generating_recommendations': "Generating AI recommendations...",
    'sending_message': "Sending message to AI assistant...",
    'loading': "Loading...",
    'initializing': "Initializing services..."
}

# Tooltips
TOOLTIPS = {
    'job_role': "Select the job role you're targeting to get relevant keyword analysis",
    'user_email': "Used to save your analysis results and enable personalized features",
    'file_upload': f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)} (Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB)",
    'ats_score': "Applicant Tracking System compatibility score (0-100)",
    'role_match': "Percentage match with target job role requirements",
    'overall_score': "Combined score based on ATS compatibility and role matching",
    'skills_count': "Number of technical and soft skills detected",
    'sections_count': "Number of resume sections found and analyzed"
}

# Default values
DEFAULTS = {
    'job_role': 'Software Engineer',
    'user_email': '',
    'max_results': 5,
    'similarity_threshold': 0.7,
    'temperature': 0.7,
    'max_tokens': 1000
}

# Feature flags
FEATURES = {
    'ai_chat': True,
    'rag_system': True,
    'pdf_generation': True,
    'bulk_analysis': True,
    'comparison_mode': True,
    'export_options': True,
    'dark_mode': False,
    'advanced_analytics': True
}

# Database table names
DB_TABLES = {
    'users': 'users',
    'resumes': 'resumes',
    'analyses': 'analyses',
    'job_descriptions': 'job_descriptions',
    'chat_sessions': 'chat_sessions',
    'chat_messages': 'chat_messages'
}

# Cache keys
CACHE_KEYS = {
    'analysis_results': 'analysis_results',
    'ai_recommendations': 'ai_recommendations',
    'similar_resumes': 'similar_resumes',
    'user_sessions': 'user_sessions',
    'system_status': 'system_status'
}

# Rate limiting
RATE_LIMITS = {
    'api_calls_per_minute': 60,
    'file_uploads_per_hour': 10,
    'analysis_requests_per_hour': 20,
    'chat_messages_per_minute': 10
}

# Security settings
SECURITY = {
    'max_session_duration': 3600,  # 1 hour
    'max_file_age': 86400,  # 24 hours
    'allowed_origins': ['http://localhost:8501', 'http://127.0.0.1:8501'],
    'csrf_protection': True,
    'input_validation': True
}
