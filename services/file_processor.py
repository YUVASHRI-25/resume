
import os
import re
import logging
import aiofiles
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Optional imports for different file formats
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Handles file upload, processing, and text extraction from PDFs, DOCX, TXT, and image files.
    Includes OCR and multi-layer fallback mechanisms.
    """

    def __init__(self):
        """Initialize the file processor."""
        self.upload_dir = Path(settings.storage.upload_dir)
        self.max_file_size = settings.storage.max_file_size
        self.allowed_extensions = settings.storage.allowed_extensions

        # Initialize OCR (PaddleOCR preferred)
        self.paddle_ocr = None
        if PADDLEOCR_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                logger.info("✅ PaddleOCR initialized successfully.")
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize PaddleOCR: {e}")

        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # FILE SAVE
    # -------------------------------------------------------------------------
    async def save_uploaded_file(self, file, filename: Optional[str] = None) -> str:
        """Save uploaded file to storage."""
        try:
            if not filename:
                ext = file.filename.split('.')[-1] if file.filename else 'txt'
                filename = f"{uuid.uuid4()}.{ext}"

            filename = self._sanitize_filename(filename)
            file_path = self.upload_dir / filename

            async with aiofiles.open(file_path, 'wb') as f:
                chunk_size = 1024 * 1024
                while chunk := await file.read(chunk_size):
                    await f.write(chunk)

            logger.info(f"📁 File saved: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"❌ Error saving file: {e}")
            raise

    # -------------------------------------------------------------------------
    # TEXT EXTRACTION
    # -------------------------------------------------------------------------
    async def extract_text(self, file_path: str, file_extension: str) -> str:
        """Extract text from various file types."""
        file_extension = file_extension.lower().lstrip('.')

        try:
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

    # -------------------------------------------------------------------------
    # PDF EXTRACTION (Main: PyMuPDF)
    # -------------------------------------------------------------------------
    async def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF (fitz) with fallback methods."""

        def clean_text(text: str) -> str:
            """Remove nulls, extra whitespace, and normalize."""
            if not text:
                return ""
            text = text.replace('\x00', ' ')
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{2,}', '\n\n', text)
            return text.strip()

        text = ""

        # 1️⃣ Try PyMuPDF
        if PYMUPDF_AVAILABLE:
            try:
                def extract_with_fitz(path):
                    doc = fitz.open(path)
                    parts = []
                    for page in doc:
                        txt = page.get_text("text")  # accurate mode
                        if txt:
                            parts.append(txt)
                    doc.close()
                    return clean_text("\n".join(parts))

                text = await asyncio.to_thread(extract_with_fitz, file_path)
                if len(text) > 500:
                    return text
            except Exception as e:
                logger.warning(f"⚠️ PyMuPDF extraction failed: {e}")

        # 2️⃣ Try pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                def extract_with_pdfplumber(path):
                    parts = []
                    with pdfplumber.open(path) as pdf:
                        for page in pdf.pages:
                            txt = page.extract_text()
                            if txt:
                                parts.append(txt)
                    return clean_text("\n".join(parts))

                text = await asyncio.to_thread(extract_with_pdfplumber, file_path)
                if len(text) > 500:
                    return text
            except Exception as e:
                logger.warning(f"⚠️ pdfplumber extraction failed: {e}")

        # 3️⃣ Try PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                def extract_with_pypdf2(path):
                    parts = []
                    with open(path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            parts.append(page.extract_text() or "")
                    return clean_text("\n".join(parts))

                text = await asyncio.to_thread(extract_with_pypdf2, file_path)
                if len(text) > 500:
                    return text
            except Exception as e:
                logger.warning(f"⚠️ PyPDF2 extraction failed: {e}")

        # 4️⃣ OCR fallback for scanned PDFs
        if PDF2IMAGE_AVAILABLE and (OCR_AVAILABLE or self.paddle_ocr):
            try:
                def ocr_pdf(path):
                    images = convert_from_path(path, dpi=300)
                    ocr_text = []

                    for img in images:
                        if self.paddle_ocr:
                            try:
                                result = self.paddle_ocr.ocr(img)
                                if result and result[0]:
                                    for line in result[0]:
                                        if line and len(line) >= 2:
                                            ocr_text.append(line[1][0])
                                continue
                            except Exception:
                                pass

                        if OCR_AVAILABLE:
                            ocr_text.append(pytesseract.image_to_string(img))

                    return clean_text("\n".join(ocr_text))

                text = await asyncio.to_thread(ocr_pdf, file_path)
                if text:
                    return text
            except Exception as e:
                logger.warning(f"⚠️ OCR fallback failed: {e}")

        if not text:
            raise Exception("❌ No PDF extraction method returned usable text.")

        return text

    # -------------------------------------------------------------------------
    # DOCX / TXT / IMAGE EXTRACTION
    # -------------------------------------------------------------------------
    async def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        if not DOCX_AVAILABLE:
            raise Exception("python-docx not available")
        try:
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise

    async def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                async with aiofiles.open(file_path, 'r', encoding=enc) as f:
                    return (await f.read()).strip()
            except UnicodeDecodeError:
                continue
        raise Exception("Could not decode TXT file with supported encodings")

    async def _extract_from_image(self, file_path: str) -> str:
        """Extract text from image (PaddleOCR → Tesseract fallback)."""
        try:
            if self.paddle_ocr:
                try:
                    result = self.paddle_ocr.ocr(file_path)
                    lines = [line[1][0] for line in result[0] if len(line) >= 2]
                    if lines:
                        return "\n".join(lines)
                except Exception as e:
                    logger.warning(f"PaddleOCR image extraction failed: {e}")

            if OCR_AVAILABLE:
                img = Image.open(file_path)
                return pytesseract.image_to_string(img).strip()

            raise Exception("No OCR backend available.")
        except Exception as e:
            logger.error(f"Image OCR extraction failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # FILE MANAGEMENT HELPERS
    # -------------------------------------------------------------------------
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ File deleted: {file_path}")
                return True
            logger.warning(f"File not found for deletion: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Return file metadata."""
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        stat = os.stat(file_path)
        return {
            "filename": Path(file_path).name,
            "file_path": file_path,
            "file_size": stat.st_size,
            "file_extension": Path(file_path).suffix.lower().lstrip('.'),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def validate_file(self, file, filename: str) -> Dict[str, Any]:
        """Validate uploaded file size and type."""
        try:
            if hasattr(file, 'size') and file.size > self.max_file_size:
                return {"valid": False, "error": "File too large."}

            ext = filename.split('.')[-1].lower()
            if ext not in self.allowed_extensions:
                return {"valid": False, "error": f"Unsupported file type: {ext}"}

            return {"valid": True}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters and limit filename length."""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = f"{name[:250]}{ext}"
        return filename

    def get_supported_formats(self) -> Dict[str, Any]:
        """Return info on supported formats."""
        return {
            "pdf_extraction": PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE,
            "docx_extraction": DOCX_AVAILABLE,
            "ocr_available": OCR_AVAILABLE or PADDLEOCR_AVAILABLE,
            "paddleocr": PADDLEOCR_AVAILABLE,
            "tesseract": OCR_AVAILABLE,
            "supported_extensions": self.allowed_extensions,
            "max_file_size": self.max_file_size,
        }