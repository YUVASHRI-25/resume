#!/usr/bin/env python3
"""
Test script to verify OpenAI integration in the Resume AI dashboard.
Run this script to test if your OpenAI API key is working correctly.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def test_openai_connection():
    """Test OpenAI API connection and response generation."""
    
    # Load environment variables
    load_dotenv('.env', override=True)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("Please set OPENAI_API_KEY in your .env file or environment variables.")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        return False
    
    print(f"✅ Found OpenAI API key: {api_key[:8]}...")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        
        # Test API call
        print("🔄 Testing API call...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond briefly and professionally."
                },
                {
                    "role": "user",
                    "content": "Hello! Can you help me improve my resume?"
                }
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        if response and response.choices:
            print("✅ API call successful!")
            print(f"🤖 AI Response: {response.choices[0].message.content.strip()}")
            return True
        else:
            print("❌ API call failed - no response received")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing OpenAI Integration for Resume AI Dashboard")
    print("=" * 50)
    
    success = test_openai_connection()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 OpenAI integration test PASSED!")
        print("Your Resume AI dashboard should now work with AI features enabled.")
    else:
        print("❌ OpenAI integration test FAILED!")
        print("Please check your API key and try again.")
    
    return success

if __name__ == "__main__":
    main()


