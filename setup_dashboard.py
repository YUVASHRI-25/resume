"""
Setup script for Resume Insight Dashboard with PostgreSQL.
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "dashboard_requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def setup_database():
    """Setup PostgreSQL database."""
    print("\n🗄️ Setting up PostgreSQL database...")
    print("Please ensure PostgreSQL is installed and running.")
    print("You can install PostgreSQL from: https://www.postgresql.org/download/")
    
    # Get database credentials
    print("\n📝 Database Configuration:")
    db_host = input("Database Host (default: localhost): ").strip() or "localhost"
    db_name = input("Database Name (default: resume_insight): ").strip() or "resume_insight"
    db_user = input("Database User (default: postgres): ").strip() or "postgres"
    db_password = input("Database Password: ").strip()
    db_port = input("Database Port (default: 5432): ").strip() or "5432"
    
    # Create .env file
    env_content = f"""# Database Configuration
DB_HOST={db_host}
DB_NAME={db_name}
DB_USER={db_user}
DB_PASSWORD={db_password}
DB_PORT={db_port}

# Application Configuration
APP_NAME=Resume Insight Dashboard
APP_VERSION=1.0.0
DEBUG=True
LOG_LEVEL=INFO

# Security
SECRET_KEY=your_secret_key_here

# File Upload Configuration
MAX_FILE_SIZE=20971520
ALLOWED_EXTENSIONS=pdf,docx,txt

# Session Configuration
SESSION_TIMEOUT=3600
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Environment file created!")
    
    # Test database connection
    print("\n🔍 Testing database connection...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        conn.close()
        print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Please check your database credentials and ensure PostgreSQL is running.")
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    directories = [
        "data",
        "data/uploads",
        "data/analyses",
        "logs",
        "static",
        "static/css",
        "static/js",
        "static/images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 Resume Insight Dashboard Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at package installation.")
        return
    
    # Create directories
    create_directories()
    
    # Setup database
    if not setup_database():
        print("❌ Setup failed at database configuration.")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Ensure PostgreSQL is running")
    print("2. Run the dashboard: streamlit run dashboard_app.py")
    print("3. Open your browser to http://localhost:8501")
    print("4. Create an account and start analyzing resumes!")
    
    print("\n🔧 Configuration Files Created:")
    print("- .env (database configuration)")
    print("- dashboard_requirements.txt (Python packages)")
    print("- dashboard_app.py (main application)")

if __name__ == "__main__":
    main()
