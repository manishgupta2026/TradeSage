import pdfplumber
import os
from typing import List, Dict, Optional

class PDFProcessor:
    def __init__(self, processed_dir: str):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def extract_text(self, pdf_path: str) -> Dict[str, str]:
        """
        Extracts text from a single PDF file.
        Returns a dictionary with metadata and full text content.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        filename = os.path.basename(pdf_path)
        full_text = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # Add page marker for context
                        full_text.append(f"--- Page {i+1} ---\n{text}")
            
            content = "\n\n".join(full_text)
            return {
                "source": filename,
                "content": content,
                "page_count": len(full_text)
            }
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return {"source": filename, "content": "", "error": str(e)}

    def save_text(self, filename: str, content: str):
        """Saves extracted text to the processed directory."""
        output_path = os.path.join(self.processed_dir, f"{filename}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved extracted text to {output_path}")

if __name__ == "__main__":
    # Test execution
    processor = PDFProcessor("z:/Trade AI/data/processed_text")
    # processor.extract_text(...) 
