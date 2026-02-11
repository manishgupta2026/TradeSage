
import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env explicitly to simulate local/dev environment
load_dotenv(override=True)

try:
    from src.llm.engine import LLMEngine
    
    print("🧪 Testing LLM Configuration...")
    print(f"   ENV LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"   ENV GROQ_API_KEY: {'*' * 4 if os.getenv('GROQ_API_KEY') else 'Not Set'}")
    
    engine = LLMEngine()
    
    print(f"   Resolved Provider: {engine.provider}")
    
    if engine.provider == "groq":
        print("✅ SUCCESS: Provider correctly set to 'groq'")
        sys.exit(0)
    else:
        print(f"❌ FAILURE: Expected 'groq', got '{engine.provider}'")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Runtime Error: {e}")
    sys.exit(1)
