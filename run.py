"""
Application entry point for Resume Insight.
"""

import uvicorn
import streamlit as st
from app.config import settings
from app.main import app as fastapi_app


def run_backend():
    """Run the FastAPI backend server."""
    print("Starting Resume Insight Backend Server...")
    print(f"API Documentation: http://{settings.api.host}:{settings.api.port}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


def run_frontend():
    """Run the Streamlit frontend."""
    print("Starting Resume Insight Frontend...")
    print("Frontend will be available at: http://localhost:8501")
    
    import subprocess
    import sys
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])


def main():
    """Main entry point."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "backend":
            run_backend()
        elif command == "frontend":
            run_frontend()
        elif command == "dev":
            print("Starting both backend and frontend in development mode...")
            print("Backend: http://localhost:8000")
            print("Frontend: http://localhost:8501")
            print("API Docs: http://localhost:8000/docs")
            
            import threading
            import time
            
            # Start backend in a separate thread
            backend_thread = threading.Thread(target=run_backend)
            backend_thread.daemon = True
            backend_thread.start()
            
            # Wait a moment for backend to start
            time.sleep(3)
            
            # Start frontend
            run_frontend()
        else:
            print("Usage: python run.py [backend|frontend|dev]")
            print("  backend  - Run only the FastAPI backend")
            print("  frontend - Run only the Streamlit frontend")
            print("  dev      - Run both backend and frontend")
    else:
        print("Resume Insight - AI-Powered Resume Analysis System")
        print("=" * 50)
        print("Usage: python run.py [backend|frontend|dev]")
        print()
        print("Commands:")
        print("  backend  - Run only the FastAPI backend server")
        print("  frontend - Run only the Streamlit frontend")
        print("  dev      - Run both backend and frontend (development)")
        print()
        print("Examples:")
        print("  python run.py dev      # Start both services")
        print("  python run.py backend  # Start only backend")
        print("  python run.py frontend # Start only frontend")
        print()
        print("After starting:")
        print("  Backend API: http://localhost:8000")
        print("  Frontend UI: http://localhost:8501")
        print("  API Docs: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
