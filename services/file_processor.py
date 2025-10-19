"""
File processing utilities for handling various document formats.
"""

import os
import logging
import aiofiles
from typing import Optional, Dict, Any
from pathlib import Path
import uuid
from datetime import datetime

# File processing imports
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Handles file upload, processing, and text extraction from various formats.
    Supports PDF, DOCX, TXT, and image files with OCR capabilities.
    """
    
    def __init__(self):
        """Initialize the file processor."""
        self.upload_dir = Path(settings.storage.upload_dir)
        self.max_file_size = settings.storage.max_file_size
        self.allowed_extensions = settings.storage.allowed_extensions
        
        # Initialize OCR if available
        self.paddle_ocr = None
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
        
        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_uploaded_file(self, file, filename: Optional[str] = None) -> str:
        """
        Save uploaded file to storage.
        
        Args:
            file: Uploaded file object
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            # Generate unique filename if not provided
            if not filename:
                file_extension = file.filename.split('.')[-1] if file.filename else 'txt'
                filename = f"{uuid.uuid4()}.{file_extension}"
            
            # Ensure filename is safe
            filename = self._sanitize_filename(filename)
            
            # Create file path
            file_path = self.upload_dir / filename
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    async def extract_text(self, file_path: str, file_extension: str) -> str:
        """
        Extract text from file based on its extension.
        
        Args:
            file_path: Path to the file
            file_extension: File extension (pdf, docx, txt, etc.)
            
        Returns:
            Extracted text content
        """
        try:
            file_extension = file_extension.lower()
            
            if file_extension == 'pdf':
                return await self._extract_from_pdf(file_path)
            elif file_extension == 'docx':
                return await self._extract_from_docx(file_path)
            elif file_extension == 'txt':
                return await self._extract_from_txt(file_path)
            elif file_extension in ['png', 'jpg', 'jpeg']:
                return await self._extract_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    async def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            # Try pdfplumber first (better for complex layouts)
            if PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    if text.strip():
                        return text.strip()
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed: {e}")
            
            # Fallback to PyPDF2
            if PYPDF2_AVAILABLE:
                try:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                    return text.strip()
                except Exception as e:
                    logger.warning(f"PyPDF2 extraction failed: {e}")
            
            raise Exception("No PDF extraction method available")
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
    
    async def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            if not DOCX_AVAILABLE:
                raise Exception("python-docx not available")
            
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    async def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            return content.strip()
            
        except UnicodeDecodeError:
            # Try with different encodings
            encodings = ['latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                        content = await f.read()
                    return content.strip()
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Could not decode text file with any supported encoding")
            
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise
    
    async def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            # Try PaddleOCR first (better accuracy)
            if self.paddle_ocr:
                try:
                    result = self.paddle_ocr.ocr(file_path)
                    text = ""
                    
                    if result and result[0]:
                        for line in result[0]:
                            if line and len(line) >= 2:
                                text += line[1][0] + "\n"
                    
                    if text.strip():
                        return text.strip()
                except Exception as e:
                    logger.warning(f"PaddleOCR extraction failed: {e}")
            
            # Fallback to Tesseract
            if OCR_AVAILABLE:
                try:
                    image = Image.open(file_path)
                    text = pytesseract.image_to_string(image)
                    return text.strip()
                except Exception as e:
                    logger.warning(f"Tesseract extraction failed: {e}")
            
            raise Exception("No OCR method available")
            
        except Exception as e:
            logger.error(f"Image OCR extraction failed: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            Success status
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File information dictionary
        """
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found"}
            
            stat = os.stat(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            return {
                "filename": Path(file_path).name,
                "file_path": file_path,
                "file_size": stat.st_size,
                "file_extension": file_extension,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"error": str(e)}
    
    def validate_file(self, file, filename: str) -> Dict[str, Any]:
        """
        Validate uploaded file.
        
        Args:
            file: File object
            filename: Original filename
            
        Returns:
            Validation results
        """
        try:
            # Check file size
            if hasattr(file, 'size') and file.size > self.max_file_size:
                return {
                    "valid": False,
                    "error": f"File too large. Maximum size: {self.max_file_size} bytes"
                }
            
            # Check file extension
            if filename:
                file_extension = filename.split('.')[-1].lower()
                if file_extension not in self.allowed_extensions:
                    return {
                        "valid": False,
                        "error": f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}"
                    }
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return {"valid": False, "error": str(e)}
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported file formats."""
        return {
            "supported_extensions": self.allowed_extensions,
            "max_file_size": self.max_file_size,
            "pdf_extraction": PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE,
            "docx_extraction": DOCX_AVAILABLE,
            "ocr_available": OCR_AVAILABLE or PADDLEOCR_AVAILABLE,
            "paddleocr_available": PADDLEOCR_AVAILABLE,
            "tesseract_available": OCR_AVAILABLE
        }
