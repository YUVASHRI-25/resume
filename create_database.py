"""
Create PostgreSQL database for Resume Insight Dashboard.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database():
    """Create the resume_insight database."""
    try:
        # Connect to PostgreSQL server (not to a specific database)
        conn = psycopg2.connect(
            host="localhost",
            database="postgres",  # Connect to default postgres database
            user="postgres",
            password=input("Enter your PostgreSQL password: "),
            port="5432"
        )
        
        # Set isolation level to autocommit
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        # Create cursor
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'resume_insight'")
        exists = cursor.fetchone()
        
        if exists:
            print("✅ Database 'resume_insight' already exists!")
        else:
            # Create database
            cursor.execute("CREATE DATABASE resume_insight")
            print("✅ Database 'resume_insight' created successfully!")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ Creating PostgreSQL database...")
    if create_database():
        print("\n🎉 Database setup complete!")
        print("Now you can run the dashboard setup again:")
        print("python setup_dashboard.py")
    else:
        print("\n❌ Database creation failed.")
        print("Please check your PostgreSQL installation and credentials.")
