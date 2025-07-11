"""Basic test for aipython functionality."""

import os
import sys
sys.path.insert(0, '.')

from aipython.ai_client import AIClient


def test_ai_client():
    """Test basic AI client functionality."""
    # Skip if no API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  Skipping test - no ANTHROPIC_API_KEY found")
        return
    
    try:
        client = AIClient()
        
        # Test code generation
        code = client.generate_code("create a list of numbers from 1 to 5")
        print("✅ Code generation test passed")
        print(f"Generated: {code}")
        
        # Test question answering
        answer = client.ask_question("What is Python?")
        print("✅ Question answering test passed")
        print(f"Answer: {answer[:100]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == '__main__':
    test_ai_client()
