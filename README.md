# 📄 Resume Insight - AI-Powered Resume Analysis System

**Resume Insight** is a comprehensive AI-powered system that analyzes resumes and job descriptions using advanced NLP and LLMs. It extracts skills, evaluates ATS compatibility, generates personalized improvement suggestions, and compares resumes with job descriptions to find skill gaps and compute matching scores.

## 🌟 Key Features

### 🤖 AI-Powered Analysis
- **Mistral 7B Integration**: Advanced LLM for personalized recommendations
- **RAG System**: Retrieval-Augmented Generation with Chroma DB
- **Semantic Analysis**: Sentence transformers for content understanding
- **Smart Recommendations**: Context-aware improvement suggestions

### 📊 Comprehensive Analysis
- **ATS Compatibility**: Automated scoring and optimization
- **Skills Matching**: Technical and soft skills detection
- **Keyword Optimization**: Role-specific keyword analysis
- **Section Analysis**: Complete resume structure evaluation
- **Grammar Check**: Language quality assessment

### 🎯 Multi-Format Support
- **PDF Processing**: Advanced PDF text extraction with pdfplumber
- **DOCX Support**: Microsoft Word document parsing
- **OCR Capability**: Image-based resume processing with PaddleOCR
- **Text Processing**: Plain text file support

### 🚀 Modern Architecture
- **FastAPI Backend**: High-performance API server
- **Streamlit Frontend**: Interactive web interface
- **PostgreSQL Database**: Robust data persistence
- **Redis Caching**: Fast session management
- **Chroma DB**: Vector database for RAG system

## 📁 Project Structure

```
resume_insight/
├── app/                          # Main application directory
│   ├── __init__.py
│   ├── main.py                   # FastAPI main application
│   ├── config.py                 # Configuration management
│   ├── database.py               # Database connection and models
│   └── api/                      # API endpoints
│       ├── __init__.py
│       ├── resume.py             # Resume analysis endpoints
│       ├── llm.py                # LLM integration endpoints
│       └── rag.py                # RAG system endpoints
├── core/                         # Core business logic
│   ├── __init__.py
│   ├── analyzer.py               # Main ResumeAnalyzer class
│   ├── llm_service.py            # LLM service integration
│   └── rag_service.py            # RAG system implementation
├── services/                     # External services
│   ├── __init__.py
│   ├── file_processor.py         # File processing utilities
│   └── nlp_processor.py          # NLP processing utilities
├── frontend/                     # Streamlit frontend
│   ├── __init__.py
│   ├── app.py                    # Main Streamlit application
│   ├── components/               # UI components
│   │   ├── __init__.py
│   │   └── sidebar.py            # Sidebar component
│   └── utils/                    # Frontend utilities
│       ├── __init__.py
│       ├── styling.py            # CSS and styling
│       └── helpers.py            # Helper functions
├── utils/                        # Utility functions
│   ├── __init__.py
│   ├── constants.py              # Application constants
│   ├── validators.py             # Input validation
│   └── formatters.py             # Data formatting utilities
├── data/                         # Data files
│   ├── uploads/                  # Uploaded files
│   ├── chroma_db/                # Chroma DB storage
│   └── embeddings/               # Pre-computed embeddings
├── tests/                        # Test files
├── static/                       # Static files
├── env.example                   # Environment variables template
├── .gitignore
├── requirements.txt
├── setup.py                      # Setup script
├── run.py                        # Application entry point
└── README.md
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd resume_insight
```

2. **Run the setup script**
```bash
python setup.py
```

3. **Start the application**
```bash
# Windows
start.bat

# Unix/Mac
./start.sh

# Manual
python run.py dev
```

### Option 2: Manual Setup

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

4. **Set up environment variables**
```bash
cp env.example .env
# Edit .env with your configuration
```

5. **Start the application**
```bash
python run.py dev
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/resume_insight
REDIS_URL=redis://localhost:6379

# LLM API Keys (At least one required)
MISTRAL_API_KEY=your_mistral_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# AWS Configuration (Optional - for S3 storage)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=resume-insight-storage

# Application Configuration
SECRET_KEY=your_secret_key_here_change_this_in_production
DEBUG=True
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:8501", "http://127.0.0.1:8501"]

# File Upload Configuration
MAX_FILE_SIZE=20971520  # 20MB in bytes
ALLOWED_EXTENSIONS=pdf,docx,txt,png,jpg,jpeg

# Chroma DB Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Required API Keys

You need at least one LLM API key:

- **Mistral API**: Get your API key from [Mistral AI](https://console.mistral.ai/)
- **OpenAI API**: Get your API key from [OpenAI](https://platform.openai.com/)

## 🎯 Usage

### Access Points

After starting the application:

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc

### Basic Workflow

1. **Upload Resume**: Upload your resume in PDF, DOCX, or TXT format
2. **Select Job Role**: Choose your target job role from the dropdown
3. **Get Analysis**: View comprehensive analysis results across multiple tabs
4. **AI Recommendations**: Get personalized suggestions from the AI assistant
5. **Download Report**: Generate and download a detailed PDF report

### API Endpoints

#### Resume Analysis
- `POST /api/resume/upload` - Upload resume file
- `POST /api/resume/analyze/{resume_id}` - Analyze uploaded resume
- `GET /api/resume/{resume_id}` - Get analysis results
- `POST /api/resume/compare` - Compare resume with job description

#### LLM Integration
- `POST /api/llm/recommendations` - Get AI recommendations
- `POST /api/llm/chat` - Interactive chat with AI assistant
- `POST /api/llm/skill-suggestions` - Get skill suggestions
- `POST /api/llm/ats-optimization` - Get ATS optimization tips

#### RAG System
- `POST /api/rag/similar-resumes` - Find similar resumes
- `POST /api/rag/context` - Get contextual recommendations
- `POST /api/rag/index-resume` - Add resume to RAG index

## 🛠️ Development

### Running Individual Services

```bash
# Backend only
python run.py backend

# Frontend only
python run.py frontend

# Both services (development)
python run.py dev
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/
```

### Database Management

```bash
# Initialize database (if using PostgreSQL)
alembic upgrade head

# Create migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## 📊 Analysis Features

### ATS Compatibility Scoring
- **Section Completeness**: Checks for essential sections
- **Formatting**: Evaluates bullet points, headers, and structure
- **Content Length**: Ensures appropriate word count
- **Contact Information**: Validates email and phone presence

### Skills Analysis
- **Technical Skills**: Detects programming languages, tools, frameworks
- **Soft Skills**: Identifies leadership, communication, teamwork skills
- **Role Matching**: Compares skills with job requirements
- **Missing Skills**: Suggests additional skills to add

### Keyword Optimization
- **Job-Specific Keywords**: Matches resume content with role requirements
- **Industry Terms**: Identifies relevant industry terminology
- **Skill Keywords**: Finds technical and soft skill mentions
- **Density Analysis**: Evaluates keyword distribution

### AI-Powered Recommendations
- **Personalized Suggestions**: Tailored advice based on analysis
- **Improvement Areas**: Identifies specific weaknesses
- **Action Items**: Provides prioritized improvement tasks
- **Best Practices**: Shares industry-standard recommendations

## 🔧 Troubleshooting

### Common Issues

1. **Backend not starting**
   - Check if port 8000 is available
   - Verify database connection in `.env`
   - Ensure all dependencies are installed

2. **Frontend not loading**
   - Check if port 8501 is available
   - Verify backend is running
   - Check browser console for errors

3. **File upload fails**
   - Check file size (max 20MB)
   - Verify file format (PDF, DOCX, TXT)
   - Ensure upload directory exists

4. **AI features not working**
   - Verify API keys in `.env`
   - Check internet connection
   - Ensure API quotas are not exceeded

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 5GB free space minimum
- **OS**: Windows 10+, macOS 10.14+, or Linux

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for providing the Mistral 7B model
- **OpenAI** for GPT model integration
- **Chroma DB** for vector database capabilities
- **Streamlit** for the frontend framework
- **FastAPI** for the backend framework

## 📞 Support

For support, please:
1. Check the [troubleshooting section](#-troubleshooting)
2. Search existing [issues](https://github.com/your-repo/issues)
3. Create a new issue with detailed information

---

**Made with ❤️ for better resumes and career success!**
