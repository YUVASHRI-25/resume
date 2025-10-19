"""
Input validation utilities for Resume Insight application.
"""

import re
from typing import Any, Dict, List, Optional, Union
from utils.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, ERROR_MESSAGES


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_filename(filename: str) -> bool:
    """
    Validate filename format and extension.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not filename or not isinstance(filename, str):
        return False
    
    # Check for dangerous characters
    dangerous_chars = r'[<>:"/\\|?*]'
    if re.search(dangerous_chars, filename):
        return False
    
    # Check file extension
    if '.' not in filename:
        return False
    
    extension = filename.split('.')[-1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """
    Validate file size.
    
    Args:
        file_size: File size in bytes
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(file_size, int) and 0 < file_size <= MAX_FILE_SIZE


def validate_job_role(job_role: str) -> bool:
    """
    Validate job role selection.
    
    Args:
        job_role: Job role to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not job_role or not isinstance(job_role, str):
        return False
    
    # Basic validation - should not contain special characters
    if re.search(r'[<>:"/\\|?*]', job_role):
        return False
    
    return len(job_role.strip()) > 0


def validate_text_content(text: str, min_length: int = 10, max_length: int = 50000) -> bool:
    """
    Validate text content.
    
    Args:
        text: Text content to validate
        min_length: Minimum required length
        max_length: Maximum allowed length
        
    Returns:
        True if valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    text_length = len(text.strip())
    return min_length <= text_length <= max_length


def validate_score(score: Union[int, float], min_value: float = 0, max_value: float = 100) -> bool:
    """
    Validate score values.
    
    Args:
        score: Score to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(score, (int, float)):
        return False
    
    return min_value <= score <= max_value


def validate_resume_id(resume_id: str) -> bool:
    """
    Validate resume ID format.
    
    Args:
        resume_id: Resume ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not resume_id or not isinstance(resume_id, str):
        return False
    
    # UUID format validation
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return re.match(uuid_pattern, resume_id, re.IGNORECASE) is not None


def validate_analysis_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate analysis request data.
    
    Args:
        data: Request data to validate
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Validate resume_id
    if 'resume_id' not in data:
        errors.append("resume_id is required")
    elif not validate_resume_id(data['resume_id']):
        errors.append("Invalid resume_id format")
    
    # Validate job_role
    if 'job_role' not in data:
        errors.append("job_role is required")
    elif not validate_job_role(data['job_role']):
        errors.append("Invalid job_role")
    
    if errors:
        raise ValidationError(f"Validation failed: {'; '.join(errors)}")
    
    return data


def validate_upload_request(data: Dict[str, Any], file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate file upload request.
    
    Args:
        data: Request data to validate
        file_info: File information to validate
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Validate user email
    if 'user_email' not in data:
        errors.append("user_email is required")
    elif not validate_email(data['user_email']):
        errors.append("Invalid email format")
    
    # Validate file info
    if 'filename' not in file_info:
        errors.append("filename is required")
    elif not validate_filename(file_info['filename']):
        errors.append(f"Invalid filename or unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    if 'file_size' not in file_info:
        errors.append("file_size is required")
    elif not validate_file_size(file_info['file_size']):
        errors.append(f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    
    if errors:
        raise ValidationError(f"Validation failed: {'; '.join(errors)}")
    
    return data


def validate_chat_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate chat request data.
    
    Args:
        data: Request data to validate
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Validate message
    if 'message' not in data:
        errors.append("message is required")
    elif not validate_text_content(data['message'], min_length=1, max_length=1000):
        errors.append("Invalid message content")
    
    # Validate resume_id if provided
    if 'resume_id' in data and data['resume_id']:
        if not validate_resume_id(data['resume_id']):
            errors.append("Invalid resume_id format")
    
    if errors:
        raise ValidationError(f"Validation failed: {'; '.join(errors)}")
    
    return data


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent XSS and other attacks.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove script tags and content
    text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    text = text[:10000]
    
    return text.strip()


def validate_api_response(response_data: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate API response data.
    
    Args:
        response_data: Response data to validate
        required_fields: List of required field names
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response_data, dict):
        return False
    
    for field in required_fields:
        if field not in response_data:
            return False
    
    return True


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate application configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Validate required configuration keys
    required_keys = ['database_url', 'secret_key']
    for key in required_keys:
        if key not in config:
            errors.append(f"{key} is required")
    
    # Validate database URL format
    if 'database_url' in config:
        db_url = config['database_url']
        if not isinstance(db_url, str) or not db_url.startswith(('postgresql://', 'sqlite:///')):
            errors.append("Invalid database_url format")
    
    # Validate secret key
    if 'secret_key' in config:
        secret_key = config['secret_key']
        if not isinstance(secret_key, str) or len(secret_key) < 32:
            errors.append("secret_key must be at least 32 characters long")
    
    if errors:
        raise ValidationError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return config


def get_validation_error_message(error_type: str) -> str:
    """
    Get user-friendly error message for validation errors.
    
    Args:
        error_type: Type of validation error
        
    Returns:
        Error message
    """
    return ERROR_MESSAGES.get(error_type, "Validation error occurred")


def validate_batch_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate batch processing request.
    
    Args:
        data: Request data to validate
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # Validate resume_ids
    if 'resume_ids' not in data:
        errors.append("resume_ids is required")
    else:
        resume_ids = data['resume_ids']
        if not isinstance(resume_ids, list):
            errors.append("resume_ids must be a list")
        elif len(resume_ids) == 0:
            errors.append("resume_ids cannot be empty")
        elif len(resume_ids) > 50:  # Limit batch size
            errors.append("Too many resume_ids. Maximum 50 allowed")
        else:
            for resume_id in resume_ids:
                if not validate_resume_id(resume_id):
                    errors.append(f"Invalid resume_id: {resume_id}")
                    break
    
    # Validate job_role if provided
    if 'job_role' in data and data['job_role']:
        if not validate_job_role(data['job_role']):
            errors.append("Invalid job_role")
    
    if errors:
        raise ValidationError(f"Validation failed: {'; '.join(errors)}")
    
    return data
