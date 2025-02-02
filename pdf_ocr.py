#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path

class PDFExtractor:
    # Define class-level attributes for available methods
    TEXT_METHODS = [
        'pymupdf', 'pdfplumber', 'pypdf', 'pdfminer',
        'tesseract', 'kraken', 'easyocr', 'paddleocr',
        'doctr', 'ocrmypdf'
    ]
    TABLE_METHODS = ['camelot']

    def __init__(self, pdf_path: str, languages: List[str] = ['eng'], dpi: int = 300):
        self.pdf_path = pdf_path
        self.languages = languages
        self.dpi = dpi
        self.text = ""
        self.available_methods = self._check_available_methods()

    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which extraction methods are available."""
        methods = {
            'pymupdf': False,
            'pdfplumber': False,
            'pypdf': False,
            'pdfminer': False,
            'tesseract': False,
            'kraken': False,
            'easyocr': False,
            'paddleocr': False,
            'doctr': False,
            'ocrmypdf': False,
            'camelot': False
        }

        # Check each library independently
        try:
            import fitz
            methods['pymupdf'] = True
        except ImportError:
            logging.debug("PyMuPDF not available")

        try:
            import pdfplumber
            methods['pdfplumber'] = True
        except ImportError:
            logging.debug("pdfplumber not available")

        try:
            import pypdf
            methods['pypdf'] = True
        except ImportError:
            logging.debug("pypdf not available")

        try:
            import pdfminer.high_level
            methods['pdfminer'] = True
        except ImportError:
            logging.debug("pdfminer not available")

        # Check for Tesseract
        if shutil.which('tesseract'):
            try:
                import pytesseract
                from PIL import Image
                from pdf2image import convert_from_path
                methods['tesseract'] = True
            except ImportError:
                logging.debug("pytesseract dependencies not available")

        # Check for Kraken
        if shutil.which('kraken'):
            try:
                import kraken
                methods['kraken'] = True
            except ImportError:
                logging.debug("kraken not available")

        # Check for EasyOCR
        try:
            import easyocr
            methods['easyocr'] = True
        except ImportError:
            logging.debug("easyocr not available")
        except Exception as e:
            logging.debug(f"easyocr import error: {e}")

        # Check for PaddleOCR
        try:
            from paddleocr import PaddleOCR
            methods['paddleocr'] = True
        except ImportError:
            logging.debug("paddleocr not available")
        except Exception as e:
            logging.debug(f"paddleocr import error: {e}")

        # Check for DocTR
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            methods['doctr'] = True
        except ImportError:
            logging.debug("doctr not available")
        except Exception as e:
            logging.debug(f"doctr import error: {e}")

        # Check for OCRmyPDF
        try:
            import ocrmypdf
            methods['ocrmypdf'] = True
        except ImportError:
            logging.debug("ocrmypdf not available")

        # Check for Camelot
        try:
            import camelot
            methods['camelot'] = True
        except ImportError:
            logging.debug("camelot not available")

        return methods

    def extract_with_pymupdf(self) -> str:
        """Extract text using PyMuPDF (fitz)."""
        if not self.available_methods['pymupdf']:
            return ""
        
        try:
            import fitz
            text = ""
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            logging.error(f"PyMuPDF extraction failed: {e}")
            return ""

    def extract_with_pdfplumber(self) -> str:
        """Extract text using pdfplumber."""
        if not self.available_methods['pdfplumber']:
            return ""
        
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logging.error(f"PDFPlumber extraction failed: {e}")
            return ""

    def extract_with_pypdf(self) -> str:
        """Extract text using pypdf."""
        if not self.available_methods['pypdf']:
            return ""
        
        try:
            import pypdf
            text = ""
            with open(self.pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        except Exception as e:
            logging.error(f"PyPDF extraction failed: {e}")
            return ""

    def extract_with_pdfminer(self) -> str:
        """Extract text using pdfminer."""
        if not self.available_methods['pdfminer']:
            return ""
        
        try:
            from pdfminer.high_level import extract_text
            return extract_text(self.pdf_path)
        except Exception as e:
            logging.error(f"PDFMiner extraction failed: {e}")
            return ""

    def perform_ocr(self) -> str:
        """Perform OCR using Tesseract."""
        if not self.available_methods['tesseract']:
            return ""
        
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from PIL import Image

            text = ""
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images, 1):
                logging.info(f"Processing page {i}/{len(images)}")
                try:
                    page_text = pytesseract.image_to_string(
                        image, 
                        lang='+'.join(self.languages)
                    )
                    text += page_text + "\n\n"
                except Exception as e:
                    logging.error(f"OCR failed on page {i}: {e}")
                    continue
                
            return text
        except Exception as e:
            logging.error(f"OCR process failed: {e}")
            return ""

    def extract_with_kraken(self) -> str:
        """Perform OCR using Kraken."""
        if not self.available_methods['kraken']:
            return ""
        
        try:
            from kraken import pageseg
            from kraken.lib import models
            from kraken import binarization
            from pdf2image import convert_from_path
            
            text = ""
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images, 1):
                logging.info(f"Processing page {i}/{len(images)} with Kraken")
                try:
                    # Binarize the image
                    binary = binarization.nlbin(image)
                    # Segment the image
                    segments = pageseg.segment(binary)
                    # Load model based on language
                    model = models.load_any('en-default.mlmodel')  # You can add language-specific models
                    # Perform OCR
                    page_text = model.predict(segments)
                    text += page_text + "\n\n"
                except Exception as e:
                    logging.error(f"Kraken OCR failed on page {i}: {e}")
                    continue
            
            return text
        except Exception as e:
            logging.error(f"Kraken OCR process failed: {e}")
            return ""

    def extract_with_easyocr(self) -> str:
        """Extract text using EasyOCR."""
        if not self.available_methods['easyocr']:
            return ""
        
        try:
            import easyocr
            from pdf2image import convert_from_path
            
            text = ""
            reader = easyocr.Reader(self.languages)
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images, 1):
                logging.info(f"Processing page {i}/{len(images)} with EasyOCR")
                try:
                    results = reader.readtext(image)
                    page_text = "\n".join([text for _, text, _ in results])
                    text += page_text + "\n\n"
                except Exception as e:
                    logging.error(f"EasyOCR failed on page {i}: {e}")
                    continue
            
            return text
        except Exception as e:
            logging.error(f"EasyOCR process failed: {e}")
            return ""

    def extract_with_paddleocr(self) -> str:
        """Extract text using PaddleOCR."""
        if not self.available_methods['paddleocr']:
            return ""
        
        try:
            from paddleocr import PaddleOCR
            from pdf2image import convert_from_path
            
            text = ""
            ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can adjust language
            images = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images, 1):
                logging.info(f"Processing page {i}/{len(images)} with PaddleOCR")
                try:
                    results = ocr.ocr(image)
                    if results is not None:
                        page_text = "\n".join([line[1][0] for line in results])
                        text += page_text + "\n\n"
                except Exception as e:
                    logging.error(f"PaddleOCR failed on page {i}: {e}")
                    continue
            
            return text
        except Exception as e:
            logging.error(f"PaddleOCR process failed: {e}")
            return ""

    def extract_with_doctr(self) -> str:
        """Extract text using DocTR."""
        if not self.available_methods['doctr']:
            return ""
        
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            
            predictor = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_pdf(self.pdf_path)
            
            result = predictor(doc)
            text = result.render()
            
            return text
        except Exception as e:
            logging.error(f"DocTR extraction failed: {e}")
            return ""

    def extract_with_ocrmypdf(self) -> str:
        """Extract text using OCRmyPDF."""
        if not self.available_methods['ocrmypdf']:
            return ""
        
        try:
            import ocrmypdf
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                output_pdf = tmp_file.name
            
            try:
                ocrmypdf.ocr(
                    self.pdf_path,
                    output_pdf,
                    language=self.languages,
                    force_ocr=True,  # Only use force_ocr, not skip_text
                    optimize=0,      # Disable optimization for faster processing
                    progress_bar=False  # Disable progress bar as we have our own logging
                )
                
                # Extract text from the OCRed PDF using PyMuPDF
                import fitz
                text = ""
                with fitz.open(output_pdf) as doc:
                    for page in doc:
                        text += page.get_text()
                
                return text
            finally:
                if os.path.exists(output_pdf):
                    os.unlink(output_pdf)
                    
        except Exception as e:
            logging.error(f"OCRmyPDF process failed: {e}")
            return ""

    def extract_tables_with_camelot(self) -> List[Any]:
        """Extract tables using Camelot."""
        if not self.available_methods['camelot']:
            return []
        
        try:
            import camelot
            tables = camelot.read_pdf(self.pdf_path)
            return tables
        except Exception as e:
            logging.error(f"Camelot table extraction failed: {e}")
            return []

    def extract_text(self, preferred_method: Optional[str] = None) -> str:
        """Extract text using available methods with an optional preferred method."""
        # Validate method type: do not allow a table extraction method here
        if preferred_method in self.TABLE_METHODS:
            raise ValueError(f"'{preferred_method}' is a table extraction method, not a text extraction method. "
                             f"Available text extraction methods: {', '.join(self.TEXT_METHODS)}")
        
        # Define extraction methods mapping
        methods = {
            'pymupdf': self.extract_with_pymupdf,
            'pdfplumber': self.extract_with_pdfplumber,
            'pypdf': self.extract_with_pypdf,
            'pdfminer': self.extract_with_pdfminer,
            'tesseract': self.perform_ocr,
            'kraken': self.extract_with_kraken,
            'easyocr': self.extract_with_easyocr,
            'paddleocr': self.extract_with_paddleocr,
            'doctr': self.extract_with_doctr,
            'ocrmypdf': self.extract_with_ocrmypdf
        }
        
        # Reorder methods to try preferred method first (if provided)
        ordered_methods = []
        if preferred_method:
            if preferred_method in self.available_methods and self.available_methods[preferred_method]:
                if preferred_method in self.TEXT_METHODS:
                    ordered_methods.append(preferred_method)
                    logging.info(f"Will try preferred method '{preferred_method}' first")
                else:
                    logging.error(f"Method '{preferred_method}' is not a valid text extraction method")
                    return ""
            else:
                logging.warning(f"Preferred method '{preferred_method}' is not available")
        
        # Add remaining methods in default order
        ordered_methods.extend([m for m in self.TEXT_METHODS if m != preferred_method])
        
        # Try each method in order
        for method_name in ordered_methods:
            if not self.available_methods.get(method_name, False):
                continue
                
            logging.info(f"Attempting extraction with {method_name}...")
            try:
                text = methods[method_name]()
                if text and text.strip():
                    text_length = len(text)
                    word_count = len(text.split())
                    logging.info(f"Successfully extracted text using {method_name} "
                                 f"({text_length} characters, ~{word_count} words)")
                    return text
                else:
                    logging.info(f"Method {method_name} did not return any text, trying next method...")
            except Exception as e:
                logging.error(f"Method {method_name} failed: {format_error(e, True)}")
                logging.info("Trying next available method...")

        logging.error("All extraction methods failed")
        return ""

def setup_logging(verbosity: int = 0):
    """
    Set up logging configuration with multiple verbosity levels.
    Args:
        verbosity (int): 0 for normal, 1 for verbose (-v), 2 for very verbose (-vv), 3 for debug (-vvv)
    """
    if verbosity == 0:
        level = logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    elif verbosity == 1:
        level = logging.INFO
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
    elif verbosity == 2:
        level = logging.DEBUG
        format_str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    else:  # verbosity >= 3
        level = logging.DEBUG
        format_str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s - [%(filename)s:%(lineno)d]'

    logging.basicConfig(
        level=level,
        format=format_str
    )

    lib_level = logging.INFO if verbosity >= 2 else logging.WARNING
    debug_level = logging.DEBUG if verbosity >= 3 else lib_level
    
    loggers = ['PIL', 'pdfminer', 'pypdf', 'camelot', 'easyocr', 'paddleocr',
               'pdf2image', 'matplotlib', 'tensorflow', 'torch', 'numexpr']
    for logger_name in loggers:
        logging.getLogger(logger_name).setLevel(lib_level)

def format_error(e: Exception, include_traceback: bool = False) -> str:
    """Format error message with optional traceback."""
    import traceback
    error_type = type(e).__name__
    error_msg = str(e) or "No error message provided"
    
    if include_traceback:
        tb = traceback.format_exc()
        return f"{error_type}: {error_msg}\n\nTraceback:\n{tb}"
    return f"{error_type}: {error_msg}"

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF with multiple fallback methods")
    parser.add_argument("input_pdf", help="Path to input PDF file")
    parser.add_argument("output_file", help="Path to output text/table file")
    
    method_group = parser.add_mutually_exclusive_group()
    # Choices include both text and table extraction methods.
    method_group.add_argument("--method", 
                              choices=PDFExtractor.TEXT_METHODS + PDFExtractor.TABLE_METHODS,
                              help="Extraction method to use (text or table extraction)")
    
    parser.add_argument("--langs", default='eng',
                        help="OCR languages (comma-separated, e.g., 'eng,deu')")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for OCR (default: 300)")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Increase verbosity level (-v, -vv, -vvv)")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if not os.path.exists(args.input_pdf):
        logging.error(f"Input file not found: {args.input_pdf}")
        sys.exit(1)

    try:
        languages = [lang.strip() for lang in args.langs.split(',')]
        extractor = PDFExtractor(args.input_pdf, languages, args.dpi)
        
        logging.info("Available extraction methods:")
        logging.info("Text extraction methods:")
        for method in PDFExtractor.TEXT_METHODS:
            status = "Available" if extractor.available_methods.get(method, False) else "Not available"
            if args.verbose >= 1 or extractor.available_methods.get(method, False):
                logging.info(f"  {method}: {status}")
        
        logging.info("Table extraction methods:")
        for method in PDFExtractor.TABLE_METHODS:
            status = "Available" if extractor.available_methods.get(method, False) else "Not available"
            if args.verbose >= 1 or extractor.available_methods.get(method, False):
                logging.info(f"  {method}: {status}")
        
        if args.method in PDFExtractor.TABLE_METHODS:
            if not extractor.available_methods.get(args.method, False):
                logging.error(f"Table extraction method '{args.method}' is not available or not properly installed.")
                if args.verbose >= 2 and args.method == 'camelot':
                    logging.debug("Camelot requires additional dependencies:")
                    logging.debug("  pip install camelot-py opencv-python ghostscript")
                    logging.debug("  brew install ghostscript tcl-tk")
                sys.exit(1)
            
            tables = extractor.extract_tables_with_camelot()
            if tables:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Found {len(tables)} tables:\n\n")
                    for i, table in enumerate(tables, 1):
                        f.write(f"Table {i}:\n")
                        f.write(table.df.to_string())
                        f.write("\n\n")
                logging.info(f"Extracted {len(tables)} tables to {args.output_file}")
            else:
                logging.error("No tables found in the PDF")
                sys.exit(1)
        else:
            text = extractor.extract_text(args.method)
            if text:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"Text successfully saved to {args.output_file}")
                if args.verbose >= 1:
                    chars = len(text)
                    words = len(text.split())
                    logging.info(f"Extracted {chars} characters, approximately {words} words")
            else:
                logging.error("Failed to extract any text from the PDF")
                sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {format_error(e, include_traceback=args.verbose >= 2)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
