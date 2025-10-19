"""
Setup script for Resume Insight application.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "data/uploads",
        "data/chroma_db",
        "data/embeddings",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True


def download_spacy_model():
    """Download spaCy English model."""
    print("🧠 Downloading spaCy model...")
    return run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model")


def setup_environment():
    """Setup environment variables."""
    print("⚙️ Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as src:
            content = src.read()
        
        with open(env_file, 'w') as dst:
            dst.write(content)
        
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your actual configuration values")
    else:
        print("ℹ️  .env file already exists or template not found")


def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"💾 Available memory: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM available. Performance may be affected.")
    except ImportError:
        print("ℹ️  psutil not available, skipping memory check")
    
    # Check disk space
    try:
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"💽 Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("⚠️  Warning: Less than 5GB disk space available.")
    except:
        print("ℹ️  Could not check disk space")


def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    # Test imports
    test_imports = [
        "streamlit",
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "plotly",
        "requests",
        "sqlalchemy",
        "chromadb",
        "sentence_transformers"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All core dependencies imported successfully")
    return True


def create_startup_scripts():
    """Create startup scripts for different platforms."""
    print("📝 Creating startup scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting Resume Insight...
echo.
echo Backend will be available at: http://localhost:8000
echo Frontend will be available at: http://localhost:8501
echo API Documentation: http://localhost:8000/docs
echo.
python run.py dev
pause
"""
    
    with open("start.bat", "w") as f:
        f.write(windows_script)
    print("✅ Created start.bat for Windows")
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting Resume Insight..."
echo ""
echo "Backend will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:8501"
echo "API Documentation: http://localhost:8000/docs"
echo ""
python run.py dev
"""
    
    with open("start.sh", "w") as f:
        f.write(unix_script)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("start.sh", 0o755)
    
    print("✅ Created start.sh for Unix systems")


def main():
    """Main setup function."""
    print("🚀 Resume Insight Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check system requirements
    check_system_requirements()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("⚠️  Warning: Failed to download spaCy model. Some features may not work.")
    
    # Setup environment
    setup_environment()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    # Create startup scripts
    create_startup_scripts()
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 50)
    print("Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Start the application:")
    print("   - Windows: Double-click start.bat")
    print("   - Unix/Mac: ./start.sh")
    print("   - Manual: python run.py dev")
    print("")
    print("Access points:")
    print("  - Frontend: http://localhost:8501")
    print("  - Backend API: http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("")
    print("For help, check the README.md file")


if __name__ == "__main__":
    main()
