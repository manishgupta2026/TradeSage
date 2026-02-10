import os
import json
import glob
from typing import List, Dict

from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.chunking import StrategyChunker
from src.llm.engine import LLMEngine
from src.llm.prompts import format_strategy_prompt
from src.extraction.validator import validate_strategy_data

class StrategyExtractor:
    def __init__(self, data_dir: str, model_path: str):
        self.data_dir = data_dir
        self.raw_pdfs_dir = os.path.join(data_dir, "raw_pdfs")
        self.processed_text_dir = os.path.join(data_dir, "processed_text")
        self.output_dir = os.path.join(data_dir, "strategies")
        
        os.makedirs(self.output_dir, exist_ok=True)

        print("Initializing components...")
        self.pdf_processor = PDFProcessor(self.processed_text_dir)
        self.chunker = StrategyChunker()
        
        # Initialize LLM (Provider logic handled inside LLMEngine)
        try:
            self.llm = LLMEngine(model_path)
        except Exception as e:
            print(f"Failed to initialize LLM Engine: {e}")
            self.llm = None

    def process_all_pdfs(self):
        pdf_files = glob.glob(os.path.join(self.raw_pdfs_dir, "*.pdf"))
        print(f"Found {len(pdf_files)} PDFs to process.")

        for pdf_path in pdf_files:
            print(f"Processing {os.path.basename(pdf_path)}...")
            try:
                strategies = self.extract_from_pdf(pdf_path)
                if strategies:
                    self.save_strategies(strategies, os.path.basename(pdf_path))
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        # 1. Extract Text
        result = self.pdf_processor.extract_text(pdf_path)
        if not result.get("content"):
            return []

        # 2. Chunk Text
        chunks = self.chunker.chunk_text(result["content"])
        print(f" - Split into {len(chunks)} chunks.")

        strategies = []
        if not self.llm:
            print(" - LLM not initialized, skipping extraction.")
            return []

        # 3. Extract Strategies
        for i, chunk in enumerate(chunks):
            print(f" - Analyzing chunk {i+1}/{len(chunks)}...")
            prompt = format_strategy_prompt(chunk)
            response = self.llm.generate(prompt)
            
            extracted = self._parse_response(response)
            if extracted:
                # Add metadata
                for s in extracted:
                    s["source_file"] = os.path.basename(pdf_path)
                strategies.extend(extracted)

        return strategies

    def _parse_response(self, response: str) -> List[Dict]:
        try:
            # Simple cleanup to find JSON array
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                raw_data = json.loads(json_str)
                
                valid_strategies = []
                for item in raw_data:
                    # Validate against schema
                    strategy = validate_strategy_data(item)
                    if strategy:
                        valid_strategies.append(strategy.dict())
                
                return valid_strategies
            return []
        except json.JSONDecodeError:
            print(" - Failed to parse JSON from response.")
            return []

    def save_strategies(self, strategies: List[Dict], source_filename: str = "strategies"):
        base_name = os.path.splitext(source_filename)[0]
        output_path = os.path.join(self.output_dir, f"{base_name}_strategies.json")
        
        # If file exists, maybe we want to append or overwrite? Overwrite per PDF is fine.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(strategies, f, indent=2)
        print(f"Saved {len(strategies)} strategies to {output_path}")

if __name__ == "__main__":
    # Example usage
    # extractor = StrategyExtractor("z:/Trade AI/data", "z:/Trade AI/models/mistral-7b.gguf")
    # extractor.process_all_pdfs()
    pass
