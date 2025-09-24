#!/usr/bin/env python3
"""
Setup script for improved LLM models.
Run this to install dependencies and test model loading.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded."""
    print("\nTesting model loading...")
    try:
        from app.qa_improved import build_plain_llm_improved
        print("✅ Improved QA module imported successfully!")
        
        # Test loading a model (this might take a while on first run)
        print("Loading model (this may take a few minutes on first run)...")
        llm = build_plain_llm_improved()
        print("✅ Model loaded successfully!")
        
        # Test a simple query
        print("Testing with a simple query...")
        response = llm.invoke("What is artificial intelligence?")
        print(f"✅ Test response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def setup_openai():
    """Setup OpenAI API key."""
    print("\nOpenAI Setup:")
    print("To use OpenAI models (best quality), you need an API key.")
    print("1. Get an API key from: https://platform.openai.com/api-keys")
    print("2. Create a .env file in your project root")
    print("3. Add: OPENAI_API_KEY=your_api_key_here")
    print("4. The system will automatically use OpenAI if available")

def main():
    """Main setup function."""
    print("🚀 Setting up improved LLM for RAG Doc Assistant")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test model loading
    if not test_model_loading():
        print("\n⚠️  Model loading failed. You can still use the basic models.")
    
    # Setup instructions
    setup_openai()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Run your Flask app: python -m app.ui_server")
    print("2. Try both RAG and LLM modes in the web interface")
    print("3. For best results, consider setting up OpenAI API key")

if __name__ == "__main__":
    main()
