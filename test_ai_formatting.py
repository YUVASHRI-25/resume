#!/usr/bin/env python3
"""
Test script to verify the new AI Assistant formatting works correctly.
This script tests the clean formatting without markdown.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def test_ai_formatting():
    """Test AI Assistant formatting with clean output."""
    
    # Load environment variables
    load_dotenv('.env', override=True)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key or api_key == 'sk-your-actual-api-key-here':
        print("❌ Please set your real OpenAI API key!")
        print("Run: python setup_api_key.py")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized successfully")
        
        # Test prompt for clean formatting
        test_prompt = "How can I improve my professional summary to make it more compelling?"
        
        print(f"\n🧪 Testing AI formatting with prompt: '{test_prompt}'")
        print("=" * 60)
        
        # Test the AI response with clean formatting instructions
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert resume consultant and career advisor. Provide specific, actionable advice for resume improvement. Format your response as clean, numbered points without any markdown formatting (no **bold**, no # headers, no bullet points). Use simple numbered lists with clear, professional language. Keep responses concise and professional."
                },
                {
                    "role": "user",
                    "content": test_prompt
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        if response and response.choices:
            raw_response = response.choices[0].message.content.strip()
            print("📝 Raw AI Response:")
            print("-" * 30)
            print(raw_response)
            
            # Test the cleaning function
            cleaned_response = clean_ai_response(raw_response)
            print("\n✨ Cleaned Response:")
            print("-" * 30)
            print(cleaned_response)
            
            # Check if formatting is clean
            has_markdown = any(char in cleaned_response for char in ['*', '#', '`'])
            if not has_markdown:
                print("\n✅ Formatting is clean - no markdown detected!")
                return True
            else:
                print("\n❌ Still contains markdown formatting")
                return False
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def clean_ai_response(response: str) -> str:
    """Clean AI response to remove markdown formatting and ensure proper numbering."""
    import re
    
    # Remove markdown bold formatting
    response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
    
    # Remove markdown italic formatting
    response = re.sub(r'\*(.*?)\*', r'\1', response)
    
    # Remove markdown headers
    response = re.sub(r'^#+\s*', '', response, flags=re.MULTILINE)
    
    # Remove bullet points and replace with proper numbering
    response = re.sub(r'^[\s]*[-•]\s*', '', response, flags=re.MULTILINE)
    
    # Clean up any remaining markdown
    response = re.sub(r'`(.*?)`', r'\1', response)
    
    # Ensure proper line breaks
    response = response.replace('\n\n', '\n')
    
    # Clean up extra spaces
    response = re.sub(r'\n\s+', '\n', response)
    
    return response.strip()

def main():
    """Main test function."""
    print("🧪 Testing AI Assistant Clean Formatting")
    print("=" * 50)
    
    success = test_ai_formatting()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 AI formatting test PASSED!")
        print("Your AI Assistant will now display clean, properly formatted responses!")
        print("\n📋 Expected formatting:")
        print("1. Clean numbered points")
        print("2. No markdown formatting")
        print("3. Proper alignment")
        print("4. Professional appearance")
    else:
        print("❌ AI formatting test FAILED!")
        print("Please check your API key and try again.")
    
    return success

if __name__ == "__main__":
    main()


