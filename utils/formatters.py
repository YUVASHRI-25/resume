"""
Data formatting utilities for Resume Insight application.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd


def format_score(score: Union[int, float], decimals: int = 1) -> str:
    """
    Format score with appropriate decimal places.
    
    Args:
        score: Score value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted score string
    """
    if score is None:
        return "N/A"
    
    return f"{float(score):.{decimals}f}"


def format_percentage(value: Union[int, float], decimals: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format as percentage
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return "N/A"
    
    return f"{float(value):.{decimals}f}%"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return "N/A"
    
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: int) -> str:
    """
    Format duration in human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def format_list_as_string(items: List[str], max_items: int = 10, separator: str = ", ") -> str:
    """
    Format list as string with truncation.
    
    Args:
        items: List of items to format
        max_items: Maximum number of items to show
        separator: Separator between items
        
    Returns:
        Formatted string
    """
    if not items:
        return "None"
    
    if len(items) <= max_items:
        return separator.join(items)
    else:
        visible_items = items[:max_items]
        remaining_count = len(items) - max_items
        return separator.join(visible_items) + f" (+{remaining_count} more)"


def format_skills_for_display(skills: List[str], max_skills: int = 20) -> Dict[str, Any]:
    """
    Format skills for display with categorization.
    
    Args:
        skills: List of skills
        max_skills: Maximum number of skills to display
        
    Returns:
        Formatted skills data
    """
    if not skills:
        return {
            "display_skills": [],
            "total_count": 0,
            "truncated": False
        }
    
    display_skills = skills[:max_skills]
    truncated = len(skills) > max_skills
    
    return {
        "display_skills": display_skills,
        "total_count": len(skills),
        "truncated": truncated,
        "remaining_count": len(skills) - max_skills if truncated else 0
    }


def format_analysis_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format analysis results for display.
    
    Args:
        results: Raw analysis results
        
    Returns:
        Formatted results
    """
    formatted = {}
    
    # Format scores
    score_fields = ['ats_score', 'role_match_percentage', 'overall_score']
    for field in score_fields:
        if field in results:
            formatted[field] = format_score(results[field])
    
    # Format counts
    count_fields = ['word_count', 'section_count']
    for field in count_fields:
        if field in results:
            formatted[field] = f"{results[field]:,}"
    
    # Format skills
    if 'technical_skills' in results:
        formatted['technical_skills'] = format_skills_for_display(results['technical_skills'])
    
    if 'soft_skills' in results:
        formatted['soft_skills'] = format_skills_for_display(results['soft_skills'])
    
    # Format keywords
    if 'found_keywords' in results:
        formatted['found_keywords'] = format_skills_for_display(results['found_keywords'])
    
    if 'missing_keywords' in results:
        formatted['missing_keywords'] = format_skills_for_display(results['missing_keywords'])
    
    # Format sections
    if 'sections' in results:
        formatted['sections'] = {}
        for section_name, content in results['sections'].items():
            formatted['sections'][section_name] = {
                'content': content,
                'length': len(content) if content else 0,
                'status': 'Complete' if content and len(content) > 50 else 'Incomplete'
            }
    
    # Format timestamps
    if 'analysis_timestamp' in results:
        formatted['analysis_timestamp'] = format_timestamp(results['analysis_timestamp'])
    
    return formatted


def format_recommendations(recommendations: List[str]) -> Dict[str, Any]:
    """
    Format recommendations for display.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        Formatted recommendations
    """
    if not recommendations:
        return {
            "recommendations": [],
            "count": 0,
            "formatted_text": "No specific recommendations available."
        }
    
    # Categorize recommendations
    priority_keywords = ['high priority', 'urgent', 'critical', 'important']
    medium_keywords = ['medium priority', 'consider', 'suggest']
    
    high_priority = []
    medium_priority = []
    other = []
    
    for rec in recommendations:
        rec_lower = rec.lower()
        if any(keyword in rec_lower for keyword in priority_keywords):
            high_priority.append(rec)
        elif any(keyword in rec_lower for keyword in medium_keywords):
            medium_priority.append(rec)
        else:
            other.append(rec)
    
    return {
        "recommendations": recommendations,
        "count": len(recommendations),
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "other": other,
        "formatted_text": "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])
    }


def format_grammar_issues(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format grammar issues for display.
    
    Args:
        issues: List of grammar issues
        
    Returns:
        Formatted grammar issues
    """
    if not issues:
        return {
            "issues": [],
            "count": 0,
            "severity": "None",
            "formatted_text": "No grammar issues detected."
        }
    
    # Categorize by severity
    critical_keywords = ['error', 'incorrect', 'wrong']
    warning_keywords = ['suggestion', 'consider', 'might']
    
    critical = []
    warnings = []
    other = []
    
    for issue in issues:
        message = issue.get('message', '').lower()
        if any(keyword in message for keyword in critical_keywords):
            critical.append(issue)
        elif any(keyword in message for keyword in warning_keywords):
            warnings.append(issue)
        else:
            other.append(issue)
    
    # Determine overall severity
    if critical:
        severity = "Critical"
    elif warnings:
        severity = "Warning"
    else:
        severity = "Info"
    
    return {
        "issues": issues,
        "count": len(issues),
        "critical": critical,
        "warnings": warnings,
        "other": other,
        "severity": severity,
        "formatted_text": "\n".join([f"• {issue.get('message', 'Unknown issue')}" for issue in issues])
    }


def format_comparison_results(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format job comparison results for display.
    
    Args:
        comparison: Comparison results
        
    Returns:
        Formatted comparison results
    """
    formatted = {}
    
    # Format similarity score
    if 'similarity_score' in comparison:
        formatted['similarity_score'] = format_percentage(comparison['similarity_score'])
    
    # Format skills
    if 'matched_skills' in comparison:
        formatted['matched_skills'] = format_skills_for_display(comparison['matched_skills'])
    
    if 'missing_skills' in comparison:
        formatted['missing_skills'] = format_skills_for_display(comparison['missing_skills'])
    
    # Format keywords
    if 'job_keywords' in comparison:
        formatted['job_keywords'] = format_skills_for_display(comparison['job_keywords'])
    
    # Format recommendations
    if 'recommendations' in comparison:
        formatted['recommendations'] = format_recommendations(comparison['recommendations'])
    
    # Format timestamp
    if 'comparison_timestamp' in comparison:
        formatted['comparison_timestamp'] = format_timestamp(comparison['comparison_timestamp'])
    
    return formatted


def format_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format chat message for display.
    
    Args:
        message: Chat message data
        
    Returns:
        Formatted message
    """
    formatted = {
        "role": message.get("role", "unknown"),
        "content": message.get("content", ""),
        "timestamp": format_timestamp(message.get("timestamp", datetime.utcnow())),
        "formatted_time": format_timestamp(message.get("timestamp", datetime.utcnow())).split(" ")[1]  # Time only
    }
    
    return formatted


def format_api_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format API response for consistent display.
    
    Args:
        response_data: Raw API response
        
    Returns:
        Formatted response
    """
    formatted = {}
    
    # Format common fields
    if 'status' in response_data:
        formatted['status'] = response_data['status']
    
    if 'message' in response_data:
        formatted['message'] = response_data['message']
    
    if 'timestamp' in response_data:
        formatted['timestamp'] = format_timestamp(response_data['timestamp'])
    
    # Format data fields
    if 'data' in response_data:
        formatted['data'] = response_data['data']
    
    # Format error fields
    if 'error' in response_data:
        formatted['error'] = response_data['error']
    
    return formatted


def create_summary_text(results: Dict[str, Any]) -> str:
    """
    Create a summary text from analysis results.
    
    Args:
        results: Analysis results
        
    Returns:
        Summary text
    """
    ats_score = results.get('ats_score', 0)
    role_match = results.get('role_match_percentage', 0)
    overall_score = results.get('overall_score', 0)
    
    summary_parts = []
    
    # Overall assessment
    if overall_score >= 80:
        summary_parts.append("Your resume shows excellent quality with strong ATS compatibility and role alignment.")
    elif overall_score >= 60:
        summary_parts.append("Your resume has a good foundation with room for improvement.")
    else:
        summary_parts.append("Your resume needs significant improvements to be competitive.")
    
    # ATS assessment
    if ats_score >= 80:
        summary_parts.append("ATS compatibility is excellent.")
    elif ats_score >= 60:
        summary_parts.append("ATS compatibility is good but can be improved.")
    else:
        summary_parts.append("ATS compatibility needs attention.")
    
    # Role match assessment
    if role_match >= 70:
        summary_parts.append("Strong alignment with target role requirements.")
    elif role_match >= 50:
        summary_parts.append("Moderate alignment with target role requirements.")
    else:
        summary_parts.append("Limited alignment with target role requirements.")
    
    return " ".join(summary_parts)


def sanitize_for_json(data: Any) -> Any:
    """
    Sanitize data for JSON serialization.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data
    """
    if isinstance(data, datetime):
        return data.isoformat()
    elif isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)
