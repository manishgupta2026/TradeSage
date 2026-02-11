import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class LLMEngine:
    def __init__(self, model_path: str = None, n_gpu_layers: int = 20, n_ctx: int = 2048):
        # Auto-detect provider if not explicitly set
        self.provider = os.getenv("LLM_PROVIDER", "").lower()
        
        if not self.provider:
            if os.getenv("GROQ_API_KEY"):
                self.provider = "groq"
            elif os.getenv("GEMINI_API_KEY"):
                self.provider = "gemini"
            elif os.getenv("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            else:
                self.provider = "local"
                
        self.max_tokens = 4096 # API supports higher limits
        
        print(f"Initializing LLM Engine with provider: {self.provider.upper()}")

        if self.provider == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in .env")
                self.client = Groq(api_key=api_key)
                self.model = "llama-3.3-70b-versatile" 
            except ImportError:
                print("Error: 'groq' library not installed. Run `pip install groq`.")
                raise

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in .env")
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
            except ImportError:
                print("Error: 'google-generativeai' library not installed.")
                raise

        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in .env")
                self.client = Anthropic(api_key=api_key)
                self.model = "claude-3-5-sonnet-20240620"
            except ImportError:
                print("Error: 'anthropic' library not installed. Run `pip install anthropic`.")
                raise

        else: # Default to Local Llama
            try:
                from llama_cpp import Llama
                if not model_path or not os.path.exists(model_path):
                    raise FileNotFoundError(f"Local model not found at {model_path}")
                
                print(f"Loading local model from {model_path}...")
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    verbose=True
                )
            except ImportError:
                print("WARNING: llama_cpp not installed.")
                raise

    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.1) -> str:
        import time
        retries = 3
        backoff = 2

        if self.provider == "groq":
            for attempt in range(retries):
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful trading assistant. Extract strategies in JSON format."},
                            {"role": "user", "content": prompt}
                        ],
                        model=self.model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return chat_completion.choices[0].message.content
                except Exception as e:
                    if "429" in str(e):
                        print(f"Groq Rate Limit (429). Retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        print(f"Groq API Error: {e}")
                        return ""
            return ""

        elif self.provider == "gemini":
            for attempt in range(retries):
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=dict(
                            candidate_count=1,
                            max_output_tokens=max_tokens,
                            temperature=temperature
                        )
                    )
                    return response.text
                except Exception as e:
                    if "429" in str(e) or "resource_exhausted" in str(e).lower():
                        print(f"Gemini Rate Limit. Retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        print(f"Gemini API Error: {e}")
                        return ""
            return ""

        elif self.provider == "anthropic":
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system="You are a helpful trading assistant. Extract strategies in JSON format.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
            except Exception as e:
                print(f"Anthropic API Error: {e}")
                return ""

        else: # Local
            try:
                output = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    stop=["</s>", "[/INST]", "Q:"],
                    echo=False,
                    temperature=temperature
                )
                return output['choices'][0]['text']
            except Exception as e:
                print(f"Local LLM Error: {e}")
                return ""
