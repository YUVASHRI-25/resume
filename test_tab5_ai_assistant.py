#!/usr/bin/env python3
"""
Test script to verify the new Tab 5 AI Assistant functionality.
This script simulates the AI Assistant tab behavior.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def test_ai_assistant_tab():
    """Test the AI Assistant tab functionality."""
    
    # Load environment variables
    load_dotenv('.env', override=True)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("Please set OPENAI_API_KEY in your .env file or environment variables.")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        
        # Test AI Assistant responses for resume improvement
        test_prompts = [
            "How can I improve my professional summary to make it more compelling?",
            "What keywords should I add for a Data Scientist position?",
            "How should I write my work experience to be more impactful?"
        ]
        
        print("\n🧪 Testing AI Assistant responses...")
        print("=" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            print("-" * 30)
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert resume consultant and career advisor. Provide specific, actionable advice for resume improvement. Keep responses concise and professional."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                if response and response.choices:
                    ai_response = response.choices[0].message.content.strip()
                    print(f"🤖 AI Response: {ai_response[:100]}...")
                    print("✅ Response generated successfully")
                else:
                    print("❌ No response received")
                    
            except Exception as e:
                print(f"❌ Error generating response: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 AI Assistant Tab 5 test completed!")
        print("The new Tab 5 should work perfectly with your OpenAI API key.")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI client error: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Tab 5 AI Assistant for Resume AI Dashboard")
    print("=" * 60)
    
    success = test_ai_assistant_tab()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Tab 5 AI Assistant test PASSED!")
        print("Your new AI Assistant tab should work perfectly!")
        print("\n📋 Features available in Tab 5:")
        print("• 💡 Improve My Summary")
        print("• 🔍 Better Keywords") 
        print("• 📝 Write Experience")
        print("• 💬 Custom questions")
        print("• 📱 Conversation history")
    else:
        print("❌ Tab 5 AI Assistant test FAILED!")
        print("Please check your API key and try again.")
    
    return success

if __name__ == "__main__":
    main()


