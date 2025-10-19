"""
Helper utilities for Resume Insight frontend.
"""

import streamlit as st
from typing import Any, Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def display_metric_card(title: str, value: str, description: str = ""):
    """Display a metric in a card format."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f"<small>{description}</small>" if description else ""}
    </div>
    """, unsafe_allow_html=True)


def display_alert_box(message: str, alert_type: str = "info"):
    """Display alert box with different types."""
    box_class = f"{alert_type}-box"
    st.markdown(f"""
    <div class="{box_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)


def create_score_gauge(score: float, title: str, color: str = "green"):
    """Create a gauge chart for scores."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_skills_chart(technical_skills: List[str], soft_skills: List[str]):
    """Create a skills distribution chart."""
    data = {
        'Type': ['Technical Skills', 'Soft Skills'],
        'Count': [len(technical_skills), len(soft_skills)]
    }
    
    df = pd.DataFrame(data)
    fig = px.pie(
        df, 
        values='Count', 
        names='Type',
        title="Skills Distribution",
        color_discrete_sequence=['#1f77b4', '#ff7f0e']
    )
    
    fig.update_layout(height=400)
    return fig


def create_keywords_chart(found_keywords: List[str], missing_keywords: List[str]):
    """Create a keywords analysis chart."""
    data = {
        'Status': ['Found Keywords', 'Missing Keywords'],
        'Count': [len(found_keywords), len(missing_keywords)]
    }
    
    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x='Status',
        y='Count',
        title="Keywords Analysis",
        color='Status',
        color_discrete_map={
            'Found Keywords': '#28a745',
            'Missing Keywords': '#dc3545'
        }
    )
    
    fig.update_layout(height=400)
    return fig


def create_section_completeness_chart(sections: Dict[str, str]):
    """Create a section completeness chart."""
    section_data = []
    
    for section_name, content in sections.items():
        status = "Complete" if content and len(content) > 50 else "Incomplete"
        section_data.append({
            'Section': section_name.title(),
            'Status': status,
            'Length': len(content) if content else 0
        })
    
    df = pd.DataFrame(section_data)
    
    # Create stacked bar chart
    fig = px.bar(
        df,
        x='Section',
        y='Length',
        color='Status',
        title="Section Completeness",
        color_discrete_map={
            'Complete': '#28a745',
            'Incomplete': '#dc3545'
        }
    )
    
    fig.update_layout(height=400)
    return fig


def create_ats_breakdown_chart(ats_score: float, word_count: int, sections_count: int):
    """Create ATS score breakdown chart."""
    # Calculate individual components (simplified)
    sections_score = min(sections_count * 15, 40)  # Max 40 points
    length_score = 20 if 300 <= word_count <= 800 else 10
    formatting_score = min(ats_score - sections_score - length_score, 40)
    
    data = {
        'Component': ['Sections', 'Length', 'Formatting'],
        'Score': [sections_score, length_score, formatting_score],
        'Max': [40, 20, 40]
    }
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current Score',
        x=df['Component'],
        y=df['Score'],
        marker_color='#007bff'
    ))
    
    fig.add_trace(go.Bar(
        name='Maximum Possible',
        x=df['Component'],
        y=df['Max'],
        marker_color='#e9ecef',
        opacity=0.3
    ))
    
    fig.update_layout(
        title="ATS Score Breakdown",
        height=400,
        barmode='overlay'
    )
    
    return fig


def format_skills_as_tags(skills: List[str], max_display: int = 20) -> str:
    """Format skills as HTML tags."""
    if not skills:
        return ""
    
    # Limit number of displayed skills
    display_skills = skills[:max_display]
    
    tags_html = ""
    for skill in display_skills:
        tags_html += f'<span class="skill-tag">{skill}</span>'
    
    if len(skills) > max_display:
        tags_html += f'<span class="skill-tag">+{len(skills) - max_display} more</span>'
    
    return tags_html


def create_recommendations_list(recommendations: List[str]) -> str:
    """Format recommendations as HTML list."""
    if not recommendations:
        return ""
    
    html = "<ul>"
    for rec in recommendations:
        html += f"<li>{rec}</li>"
    html += "</ul>"
    
    return html


def get_score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 80:
        return "#28a745"  # Green
    elif score >= 60:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def get_score_emoji(score: float) -> str:
    """Get emoji based on score value."""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    else:
        return "🔴"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def create_progress_bar(value: float, max_value: float = 100) -> str:
    """Create HTML progress bar."""
    percentage = (value / max_value) * 100
    color = get_score_color(value)
    
    return f"""
    <div style="width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {color}; height: 20px; transition: width 0.3s ease;"></div>
    </div>
    <small style="color: #6c757d;">{value:.1f}/{max_value}</small>
    """


def create_comparison_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Create comparison table for analysis results."""
    comparison_data = []
    
    for key, value in data.items():
        if isinstance(value, (int, float)):
            comparison_data.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': value,
                'Status': 'Good' if value >= 70 else 'Needs Improvement'
            })
    
    return pd.DataFrame(comparison_data)


def create_timeline_chart(analysis_history: List[Dict[str, Any]]) -> go.Figure:
    """Create timeline chart for analysis history."""
    if not analysis_history:
        return go.Figure()
    
    dates = [item['date'] for item in analysis_history]
    scores = [item['score'] for item in analysis_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='#007bff', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Score Improvement Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        height=400
    )
    
    return fig


def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe display."""
    import re
    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 50:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:45] + ('.' + ext if ext else '')
    
    return sanitized


def create_download_button(data: bytes, filename: str, mime_type: str = "application/pdf"):
    """Create download button for files."""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="btn btn-primary">Download {filename}</a>'
    return href
