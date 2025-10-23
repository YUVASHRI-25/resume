# OpenAI Integration Setup Guide

## 🚀 Quick Setup

### 1. Get Your OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Click "Create new secret key"
4. Copy the API key (it starts with `sk-`)

### 2. Configure Your API Key

#### Option A: Using .env file (Recommended)
1. Create a `.env` file in your project root
2. Add your API key:
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

#### Option B: Using Environment Variables
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-actual-api-key-here

# Linux/Mac
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

#### Option C: Using the Dashboard Interface
1. Run your Streamlit dashboard
2. In the sidebar, enter your API key in the "OpenAI API Key" field
3. The key will be saved for the current session

### 3. Test the Integration
Run the test script to verify everything works:
```bash
python test_openai_integration.py
```

### 4. Run Your Dashboard
```bash
streamlit run beautiful_dashboard_ai.py
```

## 🔧 Troubleshooting

### Common Issues:

1. **"AI unavailable" message**
   - Check if your API key is correctly set
   - Verify the API key is valid and has credits

2. **"Error calling LLM" message**
   - Check your internet connection
   - Verify your OpenAI account has sufficient credits
   - Try a different model (gpt-3.5-turbo instead of gpt-4o-mini)

3. **Import errors**
   - Make sure you have installed the required packages:
   ```bash
   pip install openai python-dotenv streamlit
   ```

### Supported Models:
- `gpt-4o-mini` (recommended, cost-effective)
- `gpt-3.5-turbo` (cheaper alternative)
- `gpt-4` (most capable, more expensive)

## 💡 Features Enabled with OpenAI:

- **Smart Resume Analysis**: AI-powered insights and recommendations
- **Interactive Chat**: Ask questions about your resume
- **Personalized Suggestions**: Tailored advice based on your specific resume
- **Keyword Optimization**: AI-suggested keywords for your target role
- **ATS Optimization**: AI-powered ATS compatibility tips

## 🔒 Security Notes:

- Never commit your API key to version control
- Use environment variables or .env files
- The .env file is already in .gitignore
- API keys are only stored in session state (temporary)

## 📊 Cost Estimation:

- **gpt-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **gpt-3.5-turbo**: ~$0.50 per 1M input tokens, ~$1.50 per 1M output tokens
- Typical resume analysis: ~$0.001-0.01 per analysis

## 🆘 Need Help?

If you're still having issues:
1. Check the test script output
2. Verify your OpenAI account status
3. Try the dashboard's built-in API key input
4. Check the console for error messages


