#!/usr/bin/env python3
"""
Job Hunter - Resume Tailor Agent
Main entry point for the application

Now using:
- Latest LangChain (0.3.x)
- Direct OpenAI embeddings (no vector DB)
- Simple in-memory vector store
- LangChain's OpenAI integration
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Import our modules
from agents.resume_agent import ResumeTailorAgent
from rag.vector_store import JobVectorStore

# Load environment variables
load_dotenv()

def check_environment():
    """Verify that all required environment variables are set"""
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables in your .env file")
        return False
    
    print("✅ Environment variables loaded successfully")
    return True

def main():
    """Main application entry point"""
    print("=" * 60)
    print("🎯 JOB HUNTER - Resume Tailor Agent")
    print("   (Lightweight OpenAI Setup)")
    print("=" * 60)
    print()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print("🚀 Starting Job Hunter Agent...")
    print()
    
    # Initialize components
    print("📝 Initializing Resume Tailor Agent (using gpt-4o-mini)...")
    agent = ResumeTailorAgent(model="gpt-4o-mini")
    
    print("💾 Setting up Vector Store (Direct OpenAI embeddings)...")
    vector_store = JobVectorStore()
    
    # Try to load existing vector store
    if vector_store.load():
        print("   ✓ Loaded existing vector store")
    else:
        print("   ⓘ No existing vector store found (will create on first use)")
    
    print()
    print("✅ Setup complete!")
    print()
    print("💡 What's different:")
    print("   ✓ No torch/transformers (saved ~400MB+)")
    print("   ✓ No FAISS/ChromaDB (saved ~100MB)")
    print("   ✓ Direct OpenAI API usage")
    print("   ✓ Latest LangChain architecture (0.3.x)")
    print("   ✓ Total install: ~20MB only!")
    print()
    print("📚 Try the demo: python examples/demo_lightweight.py")
    print()

if __name__ == "__main__":
    main()
