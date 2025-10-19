"""
Styling utilities for Resume Insight frontend.
"""

def load_custom_css():
    """Load custom CSS for better UI styling."""
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #28a745;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .skill-tag {
        display: inline-block;
        background-color: #e9ecef;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        font-size: 0.875rem;
        border: 1px solid #dee2e6;
    }
    
    .section-header {
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        color: #495057;
    }
    
    .nav-pills {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Custom button styles */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed #dee2e6;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    .stFileUploader > div:hover {
        border-color: #007bff;
        background-color: #e7f3ff;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #007bff;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 6px;
        border-left: 4px solid;
    }
    
    /* Chat interface styling */
    .chat-message {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
    
    .chat-user {
        background-color: #e7f3ff;
        border-left-color: #007bff;
    }
    
    .chat-ai {
        background-color: #f0f8f0;
        border-left-color: #28a745;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .metric-card {
            margin: 0.25rem 0;
            padding: 0.75rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .skill-tag {
            font-size: 0.75rem;
            padding: 0.2rem 0.4rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #2d3748;
            border-color: #4a5568;
            color: #e2e8f0;
        }
        
        .metric-value {
            color: #68d391;
        }
        
        .metric-label {
            color: #a0aec0;
        }
        
        .skill-tag {
            background-color: #4a5568;
            border-color: #718096;
            color: #e2e8f0;
        }
        
        .section-header {
            border-color: #4a5568;
            color: #e2e8f0;
        }
    }
    </style>
    """, unsafe_allow_html=True)
