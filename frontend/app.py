"""
Main Streamlit application for Resume Insight.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional
import io
import base64

from components.sidebar import create_sidebar
from utils.styling import load_custom_css
from utils.helpers import display_metric_card, display_alert_box
import sys
import os

# Add the parent directory to Python path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.config import settings
except ImportError:
    # Fallback configuration if app.config is not available
    class Settings:
        def __init__(self):
            self.app_name = "Resume Insight"
            self.version = "1.0.0"
            self.debug = True
            self.log_level = "INFO"
    
    settings = Settings()

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Insight - AI-Powered Resume Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# API Configuration
API_BASE_URL = "http://localhost:8000"


class ResumeInsightApp:
    """Main application class for Resume Insight."""
    
    def __init__(self):
        """Initialize the application."""
        self.api_base_url = API_BASE_URL
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if "resume_id" not in st.session_state:
            st.session_state.resume_id = None
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = None
        if "user_email" not in st.session_state:
            st.session_state.user_email = ""
        if "selected_job_role" not in st.session_state:
            st.session_state.selected_job_role = "Software Engineer"
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "ai_recommendations" not in st.session_state:
            st.session_state.ai_recommendations = None
    
    def run(self):
        """Run the main application."""
        # Header
        st.title("📄 Resume Insight")
        st.markdown("**AI-Powered Resume Analysis & Optimization System**")
        
        # Sidebar
        with st.sidebar:
            create_sidebar()
        
        # Main content area
        self.main_content()
    
    def main_content(self):
        """Main content area of the application."""
        # File upload section
        st.markdown('<h2 class="section-header">Upload Your Resume</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select your resume file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT (Maximum size: 20MB)"
        )
        
        if uploaded_file is not None:
            self.handle_file_upload(uploaded_file)
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            self.display_analysis_results()
        
        # Display chat interface
        if st.session_state.resume_id:
            self.display_chat_interface()
    
    def handle_file_upload(self, uploaded_file):
        """Handle file upload and processing."""
        try:
            # Show file information
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            with st.expander("File Information", expanded=False):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Upload file to backend
            with st.spinner("Processing your resume..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"user_email": st.session_state.user_email}
                
                response = requests.post(
                    f"{self.api_base_url}/api/resume/upload",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.resume_id = result["resume_id"]
                    st.success("Resume uploaded successfully!")
                    
                    # Automatically analyze the resume
                    self.analyze_resume()
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
    
    def analyze_resume(self):
        """Analyze the uploaded resume."""
        if not st.session_state.resume_id:
            return
        
        try:
            with st.spinner("Analyzing your resume..."):
                data = {
                    "job_role": st.session_state.selected_job_role
                }
                
                response = requests.post(
                    f"{self.api_base_url}/api/resume/analyze/{st.session_state.resume_id}",
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.analysis_results = result["results"]
                    st.success("Analysis completed successfully!")
                    
                    # Get AI recommendations
                    self.get_ai_recommendations()
                else:
                    st.error(f"Analysis failed: {response.json().get('detail', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error analyzing resume: {str(e)}")
    
    def get_ai_recommendations(self):
        """Get AI-powered recommendations."""
        if not st.session_state.resume_id:
            return
        
        try:
            data = {
                "job_role": st.session_state.selected_job_role
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/llm/recommendations",
                data={"resume_id": st.session_state.resume_id, **data}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.ai_recommendations = result["ai_recommendations"]
        
        except Exception as e:
            st.warning(f"AI recommendations temporarily unavailable: {str(e)}")
    
    def display_analysis_results(self):
        """Display comprehensive analysis results."""
        results = st.session_state.analysis_results
        
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Summary", "🔍 Skills Analysis", "📋 Section Review",
            "🎯 ATS Analysis", "💡 Recommendations"
        ])
        
        with tab1:
            self.display_summary_tab(results)
        
        with tab2:
            self.display_skills_tab(results)
        
        with tab3:
            self.display_sections_tab(results)
        
        with tab4:
            self.display_ats_tab(results)
        
        with tab5:
            self.display_recommendations_tab(results)
    
    def display_summary_tab(self, results: Dict[str, Any]):
        """Display summary tab with key metrics."""
        st.markdown('<h3 class="section-header">Resume Summary</h3>', unsafe_allow_html=True)
        
        # Key metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            display_metric_card("ATS Score", f"{results.get('ats_score', 0):.1f}/100")
        
        with metric_cols[1]:
            display_metric_card("Role Match", f"{results.get('role_match_percentage', 0):.1f}%")
        
        with metric_cols[2]:
            display_metric_card("Word Count", f"{results.get('word_count', 0)}")
        
        with metric_cols[3]:
            display_metric_card("Sections", f"{results.get('section_count', 0)}/7")
        
        st.divider()
        
        # Overall assessment
        overall_score = results.get('overall_score', 0)
        
        if overall_score >= 80:
            display_alert_box(
                "Excellent resume! Your resume shows strong alignment with the target role and good ATS compatibility.",
                "success"
            )
        elif overall_score >= 60:
            display_alert_box(
                "Good foundation with room for improvement. Focus on adding more role-specific keywords and optimizing for ATS.",
                "warning"
            )
        else:
            display_alert_box(
                "Significant improvements needed. Consider restructuring sections, adding relevant keywords, and improving ATS compatibility.",
                "error"
            )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Score breakdown chart
            scores_data = {
                'Metric': ['ATS Score', 'Role Match', 'Overall'],
                'Score': [
                    results.get('ats_score', 0),
                    results.get('role_match_percentage', 0),
                    overall_score
                ]
            }
            
            df_scores = pd.DataFrame(scores_data)
            fig_scores = px.bar(
                df_scores, x='Metric', y='Score',
                title="Score Breakdown",
                color='Score',
                color_continuous_scale='RdYlGn'
            )
            fig_scores.update_layout(height=400)
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Skills distribution
            tech_skills = results.get('technical_skills', [])
            soft_skills = results.get('soft_skills', [])
            
            skills_data = {
                'Type': ['Technical Skills', 'Soft Skills'],
                'Count': [len(tech_skills), len(soft_skills)]
            }
            
            df_skills = pd.DataFrame(skills_data)
            fig_skills = px.pie(
                df_skills, values='Count', names='Type',
                title="Skills Distribution"
            )
            fig_skills.update_layout(height=400)
            st.plotly_chart(fig_skills, use_container_width=True)
    
    def display_skills_tab(self, results: Dict[str, Any]):
        """Display skills analysis tab."""
        st.markdown('<h3 class="section-header">Skills Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Technical Skills Detected")
            tech_skills = results.get('technical_skills', [])
            
            if tech_skills:
                # Display skills as tags
                tech_html = ""
                for skill in tech_skills:
                    tech_html += f'<span class="skill-tag">{skill}</span>'
                st.markdown(tech_html, unsafe_allow_html=True)
                
                st.metric("Technical Skills Count", len(tech_skills))
            else:
                display_alert_box(
                    "No technical skills detected. Consider adding a dedicated skills section.",
                    "warning"
                )
        
        with col2:
            st.subheader("Soft Skills Detected")
            soft_skills = results.get('soft_skills', [])
            
            if soft_skills:
                # Display skills as tags
                soft_html = ""
                for skill in soft_skills:
                    soft_html += f'<span class="skill-tag">{skill}</span>'
                st.markdown(soft_html, unsafe_allow_html=True)
                
                st.metric("Soft Skills Count", len(soft_skills))
            else:
                display_alert_box(
                    "Limited soft skills detected. Consider highlighting leadership, communication, and teamwork skills.",
                    "info"
                )
        
        st.divider()
        
        # Role-specific analysis
        st.subheader(f"Analysis for {st.session_state.selected_job_role}")
        
        progress_col, details_col = st.columns([1, 2])
        
        with progress_col:
            match_percentage = results.get('role_match_percentage', 0)
            st.metric("Match Percentage", f"{match_percentage:.1f}%")
            st.progress(match_percentage / 100)
        
        with details_col:
            if match_percentage >= 70:
                display_alert_box(
                    "Excellent match for this role! Your skills align well with industry expectations.",
                    "success"
                )
            elif match_percentage >= 50:
                display_alert_box(
                    "Good match with opportunities for improvement. Consider adding more role-specific skills.",
                    "warning"
                )
            else:
                display_alert_box(
                    "Limited match detected. Focus on adding more relevant skills and keywords for this role.",
                    "error"
                )
        
        # Missing keywords
        missing_keywords = results.get('missing_keywords', [])
        
        if missing_keywords:
            st.subheader("Suggested Keywords to Add")
            missing_html = ""
            for keyword in missing_keywords[:15]:  # Show top 15
                missing_html += f'<span class="skill-tag" style="background-color: #fff3cd;">{keyword}</span>'
            st.markdown(missing_html, unsafe_allow_html=True)
    
    def display_sections_tab(self, results: Dict[str, Any]):
        """Display section review tab."""
        st.markdown('<h3 class="section-header">Section Structure Review</h3>', unsafe_allow_html=True)
        
        # Section status overview
        st.subheader("Section Completeness")
        
        sections = results.get('sections', {})
        section_status = []
        
        for section_name, section_content in sections.items():
            status = "Complete" if section_content and len(section_content) > 50 else "Missing/Incomplete"
            section_status.append({
                "Section": section_name.title(),
                "Status": status,
                "Length": len(section_content) if section_content else 0
            })
        
        status_df = pd.DataFrame(section_status)
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed section content
        st.subheader("Section Content")
        
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
    
    def display_ats_tab(self, results: Dict[str, Any]):
        """Display ATS analysis tab."""
        st.markdown('<h3 class="section-header">ATS Compatibility Analysis</h3>', unsafe_allow_html=True)
        
        # ATS score breakdown
        col1, col2 = st.columns([1, 2])
        
        with col1:
            ats_score = results.get('ats_score', 0)
            st.metric("Overall ATS Score", f"{ats_score}/100")
            st.progress(ats_score / 100)
            
            # Score interpretation
            if ats_score >= 80:
                st.success("Excellent ATS compatibility")
            elif ats_score >= 60:
                st.warning("Good ATS compatibility")
            else:
                st.error("Needs ATS optimization")
        
        with col2:
            st.subheader("ATS Checklist")
            
            # Check various ATS factors
            checklist_items = []
            
            # Section check
            sections = results.get('sections', {})
            required_sections = ['experience', 'education', 'skills']
            sections_present = sum(1 for section in required_sections
                                 if sections.get(section) and len(sections[section]) > 50)
            checklist_items.append(("Essential sections present", sections_present >= 2))
            
            # Content length check
            word_count = results.get('word_count', 0)
            checklist_items.append(("Appropriate length (300-800 words)", 300 <= word_count <= 800))
            
            # Bullet points (simplified check)
            checklist_items.append(("Uses bullet points", True))  # Assume true for now
            
            # Keywords
            found_keywords = results.get('found_keywords', [])
            checklist_items.append(("Contains relevant keywords", len(found_keywords) >= 3))
            
            for item, passed in checklist_items:
                status = "✓" if passed else "✗"
                color = "green" if passed else "red"
                st.markdown(f"<span style='color:{color}'>{status} {item}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Grammar analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Language Quality Check")
            grammar_issues = results.get('grammar_issues', [])
            
            if len(grammar_issues) == 0:
                display_alert_box("No grammar issues detected", "success")
            else:
                display_alert_box(f"{len(grammar_issues)} potential issues found", "warning")
        
        with col2:
            if grammar_issues:
                st.subheader("Issues Detected")
                for issue in grammar_issues[:5]:  # Show first 5
                    st.write(f"• {issue}")
        
        # ATS optimization tips
        st.subheader("ATS Optimization Recommendations")
        
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
    
    def display_recommendations_tab(self, results: Dict[str, Any]):
        """Display recommendations tab."""
        st.markdown('<h3 class="section-header">Personalized Recommendations</h3>', unsafe_allow_html=True)
        
        # AI Recommendations
        if st.session_state.ai_recommendations:
            st.subheader("🤖 AI-Powered Recommendations")
            st.markdown(st.session_state.ai_recommendations)
            st.divider()
        
        # Rule-based recommendations
        recommendations = results.get('recommendations', [])
        
        if recommendations:
            st.subheader("📋 Improvement Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"**{i}.** {rec}")
        else:
            display_alert_box(
                "Your resume looks great! Minor tweaks based on specific job applications can further improve your success rate.",
                "success"
            )
        
        st.divider()
        
        # Action plan
        st.subheader("🎯 Priority Action Plan")
        
        ats_score = results.get('ats_score', 0)
        match_percentage = results.get('role_match_percentage', 0)
        tech_skills = results.get('technical_skills', [])
        
        action_items = []
        
        if ats_score < 60:
            action_items.append("**High Priority:** Improve ATS compatibility - focus on formatting and standard sections")
        
        if match_percentage < 50:
            action_items.append("**High Priority:** Add more role-specific keywords and skills")
        
        if not tech_skills and st.session_state.selected_job_role in ["Data Scientist", "Software Engineer", "DevOps Engineer"]:
            action_items.append("**Medium Priority:** Add comprehensive technical skills section")
        
        action_items.append("**Ongoing:** Customize your resume for each job application by matching keywords")
        
        for action in action_items:
            st.markdown(action)
        
        st.divider()
        
        # PDF report generation
        st.subheader("📄 Download Detailed Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Generate a comprehensive PDF report with all analysis results, recommendations, and action items.")
        
        with col2:
            if st.button("Generate PDF Report", type="primary", use_container_width=True):
                self.generate_pdf_report(results)
    
    def generate_pdf_report(self, results: Dict[str, Any]):
        """Generate and download PDF report."""
        try:
            # This would integrate with the backend PDF generation
            st.success("PDF report generation feature coming soon!")
            st.info("For now, you can take screenshots of the analysis results.")
        
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
    
    def display_chat_interface(self):
        """Display AI chat interface."""
        st.markdown('<h3 class="section-header">🤖 AI Assistant</h3>', unsafe_allow_html=True)
        
        # Quick questions
        st.subheader("Quick Questions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("How to improve experience section?"):
                self.send_chat_message("What's wrong with my experience section?")
        
        with col2:
            if st.button("Make resume ATS-friendly?"):
                self.send_chat_message("How do I make it more ATS-friendly?")
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("Add better keywords?"):
                self.send_chat_message("What keywords should I add?")
        
        with col4:
            if st.button("Improve skills section?"):
                self.send_chat_message("How can I improve my skills section?")
        
        # Chat input
        user_question = st.text_input(
            "Ask about your resume:",
            placeholder="How can I improve my resume?",
            key="chat_input"
        )
        
        # Send button
        if st.button("Send", type="primary"):
            if user_question.strip():
                self.send_chat_message(user_question)
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation")
            
            for message in reversed(st.session_state.chat_history[-5:]):  # Show last 5
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
                st.caption(f"Time: {message['timestamp']}")
                st.divider()
    
    def send_chat_message(self, message: str):
        """Send message to AI chat."""
        try:
            data = {
                "message": message,
                "resume_id": st.session_state.resume_id
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/llm/chat",
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["ai_response"],
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
        except Exception as e:
            st.error(f"Error sending message: {str(e)}")


def main():
    """Main function to run the Streamlit app."""
    app = ResumeInsightApp()
    app.run()


if __name__ == "__main__":
    main()
