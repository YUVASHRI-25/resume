#!/usr/bin/env python3
"""
Quick setup script to configure your OpenAI API key.
Run this script to set up your API key properly.
"""

import os
from dotenv import load_dotenv

def setup_api_key():
    """Setup OpenAI API key interactively."""
    
    print("🔑 OpenAI API Key Setup")
    print("=" * 40)
    
    # Check if API key already exists
    load_dotenv('.env', override=True)
    existing_key = os.getenv('OPENAI_API_KEY')
    
    if existing_key and existing_key != 'sk-your-actual-api-key-here':
        print(f"✅ API key already set: {existing_key[:8]}...")
        return True
    
    print("Please enter your OpenAI API key:")
    print("(Get it from: https://platform.openai.com/api-keys)")
    print("(The key should start with 'sk-' and be about 50+ characters long)")
    
    api_key = input("\n🔑 Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ Invalid API key format! Should start with 'sk-'")
        return False
    
    if len(api_key) < 20:
        print("❌ API key seems too short! Should be 50+ characters")
        return False
    
    # Set environment variable for current session
    os.environ['OPENAI_API_KEY'] = api_key
    print(f"✅ API key set for current session: {api_key[:8]}...")
    
    # Try to create .env file
    try:
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print("✅ API key saved to .env file")
    except Exception as e:
        print(f"⚠️ Could not save to .env file: {e}")
        print("You'll need to set the environment variable each time you run the app")
    
    return True

def test_api_key():
    """Test the API key by making a simple call."""
    try:
        from openai import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No API key found!")
            return False
        
        print("\n🧪 Testing API key...")
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello! Just testing the API key."}
            ],
            max_tokens=10
        )
        
        if response and response.choices:
            print("✅ API key works perfectly!")
            return True
        else:
            print("❌ API key test failed!")
            return False
            
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Resume AI Dashboard - API Key Setup")
    print("=" * 50)
    
    # Setup API key
    if setup_api_key():
        # Test the API key
        if test_api_key():
            print("\n🎉 Setup complete! Your Resume AI dashboard should now work.")
            print("Run: streamlit run beautiful_dashboard_ai.py")
        else:
            print("\n❌ Setup failed. Please check your API key.")
    else:
        print("\n❌ Setup cancelled.")

if __name__ == "__main__":
    main()


