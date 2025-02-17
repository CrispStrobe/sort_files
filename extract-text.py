    #!/usr/bin/env python3
"""
Cross-platform PDF and EPUB text extraction tool

Installation:
1. Python packages (all platforms):
   pip install pymupdf pdfplumber pypdf pdfminer.six pytesseract pdf2image kraken easyocr paddleocr python-doctr ocrmypdf camelot-py opencv-python importlib-metadata tqdm

   Additional EPUB dependencies:
   pip install ebooklib beautifulsoup4 html2text mobi

2. System dependencies:
   Windows:
   - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
   - Ghostscript: https://ghostscript.com/releases/gsdnld.html
   - Poppler: https://github.com/oschwartz10612/poppler-windows/releases/

   macOS:
   brew install tesseract poppler ghostscript

   Linux:
   sudo apt-get install tesseract-ocr poppler-utils ghostscript
"""
import warnings
# Suppress common warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')

import os
import re
import sys
import logging
import argparse
from typing import Optional, List, Dict, Any,  Union, Tuple, Callable
import textwrap
import pkg_resources
import traceback
from contextlib import contextmanager
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


@contextmanager
def _progress_context(self, message: str):
    """Context manager for progress reporting"""
    progress = tqdm(
        desc=message,
        unit='pages',
        disable=False  # Changed from self._debug to False to always show progress
    )
    try:
        yield progress
    finally:
        progress.close()

class timeout:
    """Context manager for timeout"""
    def __init__(self, seconds):
        self.seconds = seconds
        self.timer = None

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        if self.seconds > 0:
            self.timer = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.timer)
            
class ProgressProxy:
    """Proxy for tqdm progress bar that can update description with engine info"""
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.current_engine = None
        
    def update(self, n: int = 1, engine: Optional[str] = None):
        """Update progress and optionally show current engine"""
        if engine and engine != self.current_engine:
            self.current_engine = engine
            self.progress_bar.set_description(f"Extracting text [{engine}]")
        self.progress_bar.update(n)
        
class ImportCache:
    """Global cache for imports and their availability"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules = {}
            cls._instance._available = {}
        return cls._instance
    
    def is_available(self, module_name: str, submodules: List[str] = None) -> bool:
        """Check if module and optional submodules can be imported"""
        cache_key = f"{module_name}:{','.join(submodules or [])}"
        if cache_key not in self._available:
            try:
                import importlib.util
                # Check main module
                if importlib.util.find_spec(module_name) is None:
                    self._available[cache_key] = False
                    return False
                # Check submodules if specified
                if submodules:
                    for submodule in submodules:
                        full_name = f"{module_name}.{submodule}"
                        if importlib.util.find_spec(full_name) is None:
                            self._available[cache_key] = False
                            return False
                self._available[cache_key] = True
            except Exception:
                self._available[cache_key] = False
        return self._available[cache_key]

    def import_module(self, module_name: str, submodule: str = None) -> Any:
        """Import and cache a module"""
        cache_key = f"{module_name}{f'.{submodule}' if submodule else ''}"
        if cache_key not in self._modules:
            try:
                if submodule:
                    main_module = __import__(module_name, fromlist=[submodule])
                    self._modules[cache_key] = getattr(main_module, submodule)
                else:
                    self._modules[cache_key] = __import__(module_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {cache_key}: {e}")
        return self._modules[cache_key]

class ExtractionManager:
    """Central manager for text extraction operations"""
    
    def __init__(self, debug: bool = False):
        self._debug = debug
        self._setup_logging(debug)
        self._binaries = self._check_binaries()
        self._versions = self._check_versions()
        self._extractors = {}
        
    def _setup_logging(self, debug: bool):
        """Configure logging with appropriate level and handlers"""
        level = logging.DEBUG if debug else logging.INFO
        format_str = '%(levelname)s: %(message)s' if debug else '%(message)s'
        
        # Clear any existing handlers
        logging.getLogger().handlers = []
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('extraction.log', encoding='utf-8')
            ]
        )
        
        # Suppress common warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    def _check_binaries(self) -> Dict[str, bool]:
        """Check availability of system binaries"""
        binaries = {
            'tesseract': False,
            'pdftoppm': False,
            'gs': False
        }
        
        for binary in binaries:
            try:
                result = subprocess.run(
                    [binary, '--version'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    encoding='utf-8'
                )
                binaries[binary] = True
                logging.debug(f"Found {binary}: {result.stdout.splitlines()[0]}")
            except Exception as e:
                logging.debug(f"Binary {binary} not found: {e}")
        
        return binaries

    def _check_versions(self) -> Dict[str, str]:
        """Get versions of installed Python packages"""
        versions = {}
        packages = [
            'pymupdf', 'pdfplumber', 'pypdf', 'pdfminer.six',
            'pytesseract', 'pdf2image', 'easyocr',
            'paddleocr', 'python-doctr', 'ocrmypdf', 'camelot-py',
            'ebooklib', 'beautifulsoup4', 'html2text', 'kraken'
        ]
        
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                versions[package] = version
                logging.debug(f"Found {package} version {version}")
            except Exception as e:
                logging.debug(f"Package {package} not found: {e}")
        
        return versions

    def _get_extractor(self, file_path: str) -> Union['PDFExtractor', 'EPUBExtractor']:
        """Get or create appropriate extractor for file type"""
        file_ext = os.path.splitext(file_path)[1].lower()
        cache_key = f"{file_ext}:{file_path}"
        
        if cache_key not in self._extractors:
            if file_ext == '.pdf':
                # Changed: Only pass debug parameter once
                self._extractors[cache_key] = PDFExtractor(debug=self._debug)
            elif file_ext == '.epub':
                self._extractors[cache_key] = EPUBExtractor(debug=self._debug)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        
        return self._extractors[cache_key]
    
    def extract(self, input_path: str, 
            output_path: Optional[str] = None,
            method: Optional[str] = None,
            password: Optional[str] = None,
            **kwargs) -> Union[str, bool]:
        """
        Extract text from a document with progress reporting and error handling
        
        Args:
            input_path: Path to input document
            output_path: Optional path for output text file
            method: Preferred extraction method
            password: Password for encrypted documents
            **kwargs: Additional extraction options
            
        Returns:
            Extracted text if output_path is None, else success boolean
        """
        try:
            # Get appropriate extractor
            extractor = self._get_extractor(input_path)
            
            # Configure extraction
            if password:
                extractor.set_password(password)
            
            # Extract text with progress reporting
            with tqdm(desc="Extracting text", disable=not self._debug, unit='pages') as pbar:
                def progress_callback(n: int, engine: Optional[str] = None):
                    if engine:
                        pbar.set_description(f"Extracting text [{engine}]")
                    pbar.update(n)
                    
                text = extractor.extract_text(
                    input_path,
                    preferred_method=method,
                    progress_callback=progress_callback,
                    **kwargs
                )
                
            # Validate and handle output
            if not self._validate_text(text):
                logging.warning("Extracted text may be of low quality")
                
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                return True
            return text
            
        except Exception as e:
            error_context = self._recover_from_error(e, "extraction")
            if error_context:
                logging.error(f"Extraction failed: {error_context}")
            else:
                logging.error(f"Extraction failed: {str(e)}")
                if self._debug:
                    traceback.print_exc()
            return False if output_path else ""

    @contextmanager
    def _progress_context(self, message: str):
        """Context manager for progress reporting"""
        progress = tqdm(
            desc=message,
            disable=not self._debug,
            unit='pages'
        )
        try:
            yield progress
        finally:
            progress.close()

    def _validate_text(self, text: str, min_length: int = 50) -> bool:
        """Validate extracted text quality"""
        if not text or len(text.strip()) < min_length:
            return False
            
        # Check for garbage content
        garbage_ratio = sum(1 for c in text if not c.isprintable()) / len(text)
        if garbage_ratio > 0.1:
            return False
            
        # Check line lengths
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False
            
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 20:
            return False
            
        return True

    def _recover_from_error(self, error: Exception, context: str = "") -> Optional[str]:
        """Try to recover from extraction errors"""
        error_str = str(error)
        
        if isinstance(error, MemoryError):
            self._clear_memory()
            return "Memory error occurred - cleared memory"
            
        if "PDF file is encrypted" in error_str:
            return "PDF is encrypted - try providing a password"
            
        if "PDF file is damaged" in error_str:
            return "PDF file appears to be damaged"
            
        if "no text extractable" in error_str.lower():
            return "No extractable text found - try OCR"
            
        if "not enough memory" in error_str.lower():
            self._clear_memory()
            return "Memory allocation failed - try reducing batch size"
            
        if "permission error" in error_str.lower():
            return "Permission denied - check file access"
            
        if "timeout" in error_str.lower():
            return "Operation timed out - try again"
            
        return None

    def _clear_memory(self):
        """Clear memory and cached data"""
        import gc
        gc.collect()
        
        # Clear GPU memory if available
        if self._check_gpu_available():
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except:
                    try:
                        import paddle
                        paddle.device.cuda.empty_cache()
                    except:
                        pass

    def _check_gpu_available(self) -> bool:
        """Enhanced GPU availability check"""
        try:
            import torch
            if torch.cuda.is_available():
                # Check if CUDA initialization works
                try:
                    torch.cuda.init()
                    device = torch.cuda.current_device()
                    capability = torch.cuda.get_device_capability(device)
                    logging.info(f"CUDA device available: {torch.cuda.get_device_name(device)} "
                            f"(Compute {capability[0]}.{capability[1]})")
                    return True
                except Exception as e:
                    logging.warning(f"CUDA available but initialization failed: {e}")
                    return False
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("MPS (Metal Performance Shaders) device available")
                return True
                
            logging.warning("No GPU detected - using CPU only")
            return False
        except ImportError:
            logging.warning("PyTorch not available - using CPU only")
            return False

class EPUBExtractor:
    """EPUB text extraction with multiple fallback methods"""
    
    def __init__(self, import_cache: ImportCache, debug: bool = False):
        self._import_cache = import_cache
        self._debug = debug
        self._checked_methods = {}
        self._available_methods = None

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                'ebooklib': self._import_cache.is_available('ebooklib'),
                'bs4': self._import_cache.is_available('bs4'),
                'html2text': self._import_cache.is_available('html2text'),
                'zipfile': True  # Basic fallback always available
            }
        return self._available_methods

    def extract_text(self, epub_path: str, preferred_method: Optional[str] = None,
                    progress_callback: Optional[Callable] = None) -> str:
        """
        Extract text with fallback methods and progress reporting
        
        Args:
            epub_path: Path to EPUB file
            preferred_method: Optional preferred extraction method
            progress_callback: Optional callback for progress updates
            
        Returns:
            Extracted text
        """
        methods = ['ebooklib', 'bs4', 'zipfile']
        if preferred_method:
            if preferred_method not in methods:
                raise ValueError(f"Invalid method: {preferred_method}")
            methods.insert(0, methods.pop(methods.index(preferred_method)))

        text = ""
        with tqdm(total=len(methods), desc="Trying extraction methods", unit="method") as method_pbar:
            for method in methods:
                if not self.available_methods.get(method, False):
                    method_pbar.update(1)
                    continue
                    
                try:
                    if progress_callback:
                        progress_callback(0, method)  # Signal start with method name
                    
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        epub_path,
                        lambda n: progress_callback(n, method) if progress_callback else None
                    )
                    
                    if text and text.strip():
                        method_pbar.update(1)
                        break
                        
                except Exception as e:
                    logging.debug(f"Error with {method}: {e}")
                    
                method_pbar.update(1)

        return text.strip()

    def extract_with_ebooklib(self, epub_path: str, progress_callback=None) -> str:
        """Extract using ebooklib with BeautifulSoup parsing and progress bars"""
        ebooklib = self._import_cache.import_module('ebooklib')
        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
        
        text_parts = []
        book = None
        
        try:
            with tqdm(desc="Loading EPUB", unit="file") as pbar:
                book = ebooklib.epub.read_epub(epub_path)
                pbar.update(1)
            
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            
            with tqdm(total=len(items), desc="Extracting content", unit="item") as pbar:
                for i, item in enumerate(items):
                    try:
                        content = item.get_content().decode('utf-8')
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove unwanted elements
                        for tag in soup(['script', 'style', 'nav']):
                            tag.decompose()
                        
                        # Extract text with layout preservation
                        text = self._process_html_content(soup)
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.debug(f"Item extraction failed: {e}")
                        continue
                    
        finally:
            book = None  # Release memory
            
        return '\n\n'.join(text_parts)

    def _process_html_content(self, soup) -> str:
        """Process HTML content with layout preservation"""
        text_parts = []
        
        # Process headings with progress
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        with tqdm(total=len(headings), desc="Processing headings", unit="heading", leave=False) as pbar:
            for tag in headings:
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(f"\n{text}\n")
                pbar.update(1)
        
        # Process paragraphs and other block elements with progress
        blocks = soup.find_all(['p', 'div', 'section'])
        with tqdm(total=len(blocks), desc="Processing blocks", unit="block", leave=False) as pbar:
            for tag in blocks:
                text = tag.get_text(strip=True)
                if text:
                    text_parts.append(text)
                pbar.update(1)
        
        return '\n\n'.join(text_parts)

    def extract_with_bs4(self, epub_path: str, progress_callback=None) -> str:
        """Extract using BeautifulSoup with zipfile and progress bars"""
        BeautifulSoup = self._import_cache.import_module('bs4').BeautifulSoup
        html2text = self._import_cache.import_module('html2text').HTML2Text()
        zipfile = self._import_cache.import_module('zipfile')
        
        text_parts = []
        
        try:
            with zipfile.ZipFile(epub_path) as zf:
                # Get HTML files
                html_files = [f for f in zf.namelist() 
                            if f.endswith(('.html', '.xhtml', '.htm'))]
                
                with tqdm(total=len(html_files), desc="Processing HTML files", unit="file") as pbar:
                    for i, html_file in enumerate(html_files):
                        try:
                            content = zf.read(html_file).decode('utf-8')
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Remove unwanted elements
                            for tag in soup(['script', 'style', 'nav']):
                                tag.decompose()
                            
                            # Convert to markdown-style text
                            html2text.ignore_links = True
                            html2text.ignore_images = True
                            text = html2text.handle(str(soup))
                            
                            if text.strip():
                                text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"File extraction failed: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"EPUB extraction failed: {e}")
            
        return '\n\n'.join(text_parts)

    def extract_with_zipfile(self, epub_path: str, progress_callback=None) -> str:
        """Basic fallback extraction using zipfile with progress bars"""
        zipfile = self._import_cache.import_module('zipfile')
        import re
        
        text_parts = []
        html_pattern = re.compile(r'<[^>]+>')
        
        try:
            with zipfile.ZipFile(epub_path) as zf:
                html_files = [f for f in zf.namelist() 
                            if f.endswith(('.html', '.xhtml', '.htm'))]
                
                with tqdm(total=len(html_files), desc="Extracting text", unit="file") as pbar:
                    for i, html_file in enumerate(html_files):
                        try:
                            content = zf.read(html_file).decode('utf-8')
                            
                            # Basic HTML cleaning
                            content = re.sub(r'<script.*?</script>', '', content, 
                                           flags=re.DOTALL)
                            content = re.sub(r'<style.*?</style>', '', content, 
                                           flags=re.DOTALL)
                            content = html_pattern.sub(' ', content)
                            
                            # Clean up whitespace
                            content = re.sub(r'\s+', ' ', content).strip()
                            
                            if content:
                                text_parts.append(content)
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"File extraction failed: {e}")
                            continue
                            
        except Exception as e:
            logging.error(f"EPUB extraction failed: {e}")
            
        return '\n\n'.join(text_parts)

class TableExtractor:
    """PDF table extraction using Camelot"""
    
    def __init__(self, import_cache: ImportCache):
        self._import_cache = import_cache
        self._camelot = None
        
    def extract_tables(self, pdf_path: str) -> List[Any]:
        """Extract tables using multiple methods"""
        if not self._init_camelot():
            return []
            
        tables = []
        methods = [
            ('lattice', {'line_scale': 40}),
            ('stream', {'edge_tol': 500})
        ]
        
        for method, params in methods:
            try:
                current_tables = self._camelot.read_pdf(
                    pdf_path,
                    flavor=method,
                    pages='all',
                    **params
                )
                
                if len(current_tables) > 0:
                    tables.extend(current_tables)
                    
            except Exception as e:
                logging.debug(f"Table extraction failed with {method}: {e}")
                continue
                
        return tables
        
    def _init_camelot(self) -> bool:
        """Initialize Camelot library"""
        if self._camelot is None:
            try:
                self._camelot = self._import_cache.import_module('camelot')
                return True
            except Exception as e:
                logging.error(f"Failed to initialize Camelot: {e}")
                return False
        return True

class ImportCache:
    """Global cache for imports and their availability"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modules = {}
            cls._instance._available = {}
        return cls._instance
    
    def is_available(self, module_name: str, submodules: List[str] = None) -> bool:
        """Check if module and optional submodules can be imported"""
        cache_key = f"{module_name}:{','.join(submodules or [])}"
        if cache_key not in self._available:
            try:
                import importlib.util
                # Check main module
                if importlib.util.find_spec(module_name) is None:
                    self._available[cache_key] = False
                    return False
                # Check submodules if specified
                if submodules:
                    for submodule in submodules:
                        full_name = f"{module_name}.{submodule}"
                        if importlib.util.find_spec(full_name) is None:
                            self._available[cache_key] = False
                            return False
                self._available[cache_key] = True
            except Exception:
                self._available[cache_key] = False
        return self._available[cache_key]

    def import_module(self, module_name: str, submodule: str = None) -> Any:
        """Import and cache a module"""
        cache_key = f"{module_name}{f'.{submodule}' if submodule else ''}"
        if cache_key not in self._modules:
            try:
                if submodule:
                    main_module = __import__(module_name, fromlist=[submodule])
                    self._modules[cache_key] = getattr(main_module, submodule)
                else:
                    self._modules[cache_key] = __import__(module_name)
            except ImportError as e:
                raise ImportError(f"Failed to import {cache_key}: {e}")
        return self._modules[cache_key]
    
class PDFExtractor:
    """Enhanced PDF text extraction with lazy loading and multiple fallback methods"""

    TEXT_METHODS = [
        'pymupdf',      # Fast native PDF parsing
        'pdfplumber',   # Good balance of speed and accuracy
        'pypdf',        # Simple but reliable
        'pdfminer',     # Good layout preservation
        'tesseract',    # OCR support
        'doctr',        # Deep learning OCR
        'easyocr',      # Alternative OCR
        'kraken'        # Advanced OCR
    ]
    TABLE_METHODS = ['camelot']
    
    def _setup_windows_paths(self):
        """Add binary paths to system PATH for Windows"""
        if platform.system() == 'Windows':
            paths_to_check = [
                r'C:\Program Files\Tesseract-OCR',
                r'C:\Program Files (x86)\Tesseract-OCR',
                r'C:\Program Files\gs\gs10.02.0\bin',  # Update version as needed
                r'C:\Program Files (x86)\gs\gs10.02.0\bin',
                r'C:\Program Files\poppler-24.02.0\Library\bin',  # Update version as needed
            ]
            
            for path in paths_to_check:
                if os.path.exists(path) and path not in os.environ['PATH']:
                    os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
    
    @property
    def languages(self):
        """Return list of OCR languages"""
        return ['eng']  # Add more languages as needed
    
    def __init__(self, debug=False):
        """Initialize PDF extractor"""
        self._debug = debug
        self._import_cache = ImportCache()
        self._initialized_methods = set()
        self._password = None
        self._current_doc = None
        self._ocr_initialized = {}
        
        # Setup Windows paths
        self._setup_windows_paths()
        
        # Check system dependencies
        self._binaries = self._check_system_dependencies()
        
        # Check all dependencies
        self._check_dependencies()
        
    def _check_system_dependencies(self) -> Dict[str, bool]:
        """Check and install system dependencies"""
        binaries = {
            'tesseract': False,
            'poppler': False,
            'ghostscript': False
        }
        
        system = platform.system()
        
        if system == 'Windows':
            # Check Tesseract
            tesseract_path = shutil.which('tesseract')
            if not tesseract_path:
                logging.info("Tesseract not found. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                
            # Check Poppler
            pdftoppm_path = shutil.which('pdftoppm')
            if not pdftoppm_path:
                logging.info("Poppler not found. Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
                
            # Check Ghostscript
            gs_path = shutil.which('gs')
            if not gs_path:
                logging.info("Ghostscript not found. Download from: https://ghostscript.com/releases/gsdnld.html")
                
            # Update binary status
            binaries['tesseract'] = bool(tesseract_path)
            binaries['poppler'] = bool(pdftoppm_path)
            binaries['ghostscript'] = bool(gs_path)
            
        return binaries
        
    def _check_dependencies(self):
        """Check all possible dependencies"""
        try:
            # Core text extraction modules
            import fitz  # pymupdf
            self._initialized_methods.add('pymupdf')
        except ImportError:
            pass

        try:
            import pdfplumber
            self._initialized_methods.add('pdfplumber')
        except ImportError:
            pass

        try:
            import pypdf
            self._initialized_methods.add('pypdf')
        except ImportError:
            pass

        try:
            import pdfminer
            self._initialized_methods.add('pdfminer')
        except ImportError:
            pass

        # OCR-related checks
        try:
            import pytesseract
            import pdf2image
            import shutil
            if shutil.which('tesseract'):
                self._initialized_methods.add('tesseract')
        except ImportError:
            pass

        try:
            import doctr
            self._initialized_methods.add('doctr')
        except ImportError:
            pass

        try:
            import easyocr
            self._initialized_methods.add('easyocr')
        except ImportError:
            pass

        try:
            import kraken
            self._initialized_methods.add('kraken')
        except ImportError:
            pass

        if self._debug:
            available = sorted(list(self._initialized_methods))
            logging.debug(f"Available extraction methods: {', '.join(available)}")
    
    def _is_method_available(self, method: str) -> bool:
        """Check if extraction method is available"""
        return method in self._initialized_methods
        
    def set_password(self, password: str):
        """Set password for encrypted PDFs"""
        self._password = password

    @property
    def available_methods(self) -> Dict[str, bool]:
        """Lazy load available methods"""
        if self._available_methods is None:
            self._available_methods = {
                method: self._is_method_available(method)
                for method in self.TEXT_METHODS + self.TABLE_METHODS
            }
        return self._available_methods

    def _is_method_available_old(self, method: str) -> bool:
        """Check method availability with dependencies"""
        if method not in self._checked_methods:
            try:
                if method == 'pymupdf':
                    self._checked_methods[method] = self._import_cache.is_available('fitz')
                elif method == 'pdfplumber':
                    self._checked_methods[method] = self._import_cache.is_available('pdfplumber')
                elif method == 'pypdf':
                    self._checked_methods[method] = self._import_cache.is_available('pypdf')
                elif method == 'pdfminer':
                    self._checked_methods[method] = self._import_cache.is_available('pdfminer.high_level')
                elif method == 'tesseract':
                    self._checked_methods[method] = (
                        self._import_cache.is_available('pytesseract') and
                        self._import_cache.is_available('pdf2image') and
                        self._binaries.get('tesseract', False)
                    )
                elif method == 'kraken':
                    self._checked_methods[method] = self._import_cache.is_available('kraken')
                elif method == 'easyocr':
                    self._checked_methods[method] = self._import_cache.is_available('easyocr')
                elif method == 'doctr':
                    self._checked_methods[method] = self._import_cache.is_available('doctr')
                elif method == 'camelot':
                    self._checked_methods[method] = (
                        self._import_cache.is_available('camelot') and
                        self._binaries.get('gs', False)
                    )
            except Exception as e:
                logging.debug(f"Error checking {method}: {e}")
                self._checked_methods[method] = False
                
        return self._checked_methods[method]

    def extract_text(self, pdf_path: str, preferred_method: Optional[str] = None,
                progress_callback: Optional[Callable] = None, **kwargs) -> str:
        """Extract text with optimized fallback methods"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
            
        # Start with fast methods, then fall back to slower ones if needed
        methods = ['pymupdf', 'pdfplumber', 'pypdf']  # Fast methods first
        
        # Only add OCR methods if text extraction fails
        ocr_methods = ['tesseract', 'doctr', 'easyocr', 'kraken']
        text_parts = []
        current_method = None

        # Handle preferred method
        if preferred_method:
            if preferred_method not in self.TEXT_METHODS:
                raise ValueError(f"Invalid method: {preferred_method}")
            if preferred_method in self._initialized_methods:
                methods.insert(0, preferred_method)

        try:
            # Try fast methods first
            for method in methods:
                if method not in self._initialized_methods:
                    continue
                    
                try:
                    current_method = method
                    if self._debug:
                        logging.info(f"Trying extraction with {method}...")
                    
                    if progress_callback:
                        progress_callback(0, method)
                        
                    extraction_func = getattr(self, f'extract_with_{method}')
                    text = extraction_func(
                        pdf_path,
                        lambda n: progress_callback(n, method) if progress_callback else None
                    )
                    
                    if text and text.strip():
                        text_parts.append(text.strip())
                        quality = self._assess_text_quality(text)
                        if self._debug:
                            logging.info(f"Text quality with {method}: {quality:.2f}")
                        if quality > 0.7:  # Good enough quality
                            if self._debug:
                                logging.info(f"Got good quality text from {method}, stopping here")
                            break
                    
                except Exception as e:
                    if self._debug:
                        logging.error(f"Error with {method}: {e}")
                    continue
                finally:
                    if progress_callback and current_method == method:
                        progress_callback(1, None)

            # Only try OCR if we didn't get good text and might need OCR
            if not text_parts or self._might_need_ocr(pdf_path):
                if self._debug:
                    logging.info("Initial extraction insufficient, trying OCR methods...")
                
                for method in ocr_methods:
                    if method not in self._initialized_methods:
                        continue
                        
                    try:
                        current_method = method
                        if self._debug:
                            logging.info(f"Trying OCR with {method}...")
                        
                        if progress_callback:
                            progress_callback(0, method)
                            
                        extraction_func = getattr(self, f'extract_with_{method}')
                        text = extraction_func(
                            pdf_path,
                            lambda n: progress_callback(n, method) if progress_callback else None
                        )
                        
                        if text and text.strip():
                            text_parts.append(text.strip())
                            if self._debug:
                                logging.info(f"Got text from {method}")
                            break  # One successful OCR method is enough
                            
                    except Exception as e:
                        if self._debug:
                            logging.error(f"Error with {method}: {e}")
                        continue
                    finally:
                        if progress_callback and current_method == method:
                            progress_callback(1, None)

        finally:
            self._cleanup()

        return "\n\n".join(text_parts).strip()

    def _might_need_ocr(self, pdf_path: str) -> bool:
        """Quick check if PDF might need OCR"""
        try:
            if 'pymupdf' in self._initialized_methods:
                import fitz
                doc = fitz.open(pdf_path)
                try:
                    # Check first 3 pages or all pages if less
                    pages_to_check = min(3, len(doc))
                    total_text = 0
                    
                    for i in range(pages_to_check):
                        text = doc[i].get_text()
                        total_text += len(text.strip())
                        
                    # If average text per page is very low, might need OCR
                    return (total_text / pages_to_check) < 100
                    
                finally:
                    doc.close()
        except Exception:
            pass
            
        return True  # Default to yes if we can't check

    def _assess_text_quality(self, text: str) -> float:
        """Assess extracted text quality"""
        if not text:
            return 0.0
            
        score = 0.0
        text = text.strip()
        
        # Basic text characteristics
        words = text.split()
        if not words:
            return 0.0
            
        # Check word lengths
        avg_word_len = sum(len(w) for w in words) / len(words)
        if 3 <= avg_word_len <= 10:
            score += 0.3
            
        # Check for reasonable text structure
        lines = text.split('\n')
        if lines:
            # Check line lengths
            avg_line_len = sum(len(l.strip()) for l in lines) / len(lines)
            if 30 <= avg_line_len <= 100:
                score += 0.3
                
        # Check for paragraph structure
        if '\n\n' in text:
            score += 0.2
            
        # Check character distribution
        alpha_count = sum(c.isalpha() for c in text)
        if len(text) > 0:
            alpha_ratio = alpha_count / len(text)
            if 0.6 <= alpha_ratio <= 0.9:
                score += 0.2
                
        return min(max(score, 0.0), 1.0)

    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """Quick check if PDF appears to be scanned"""
        try:
            # Try quick text extraction with pymupdf
            import fitz
            doc = fitz.open(pdf_path)
            first_page = doc[0]
            text = first_page.get_text()
            doc.close()
            
            # If first page has very little text, likely scanned
            return len(text.strip()) < 100
            
        except Exception:
            return False

    def _needs_further_processing(self, text: str) -> bool:
        """Check if text needs additional processing methods"""
        # Check text quality
        words = text.split()
        if len(words) < 100:  # Too short - try other methods
            return True
            
        # Check for common OCR/extraction artifacts
        artifacts = ['�', '□', '■', '○', '●', '¶']
        artifact_count = sum(text.count(a) for a in artifacts)
        if artifact_count > len(text) * 0.01:  # More than 1% artifacts
            return True
            
        # Check for layout issues
        lines = text.splitlines()
        if not lines:
            return True
            
        # Check for suspiciously short lines
        short_lines = sum(1 for line in lines if len(line.strip()) < 20)
        if short_lines > len(lines) * 0.5:  # More than 50% short lines
            return True
            
        # Check for reasonable paragraph structure
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:  # No clear paragraph breaks
            return True
            
        return False

    def _cleanup(self):
        """Clean up resources"""
        if self._current_doc:
            try:
                self._current_doc.close()
            except:
                pass
            self._current_doc = None
            
        import gc
        gc.collect()

    def _clear_memory(self):
        """Clear memory after processing"""
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
    def extract_with_pymupdf(self, pdf_path: str, progress_callback=None) -> str:
        try:
            fitz = self._import_cache.import_module('fitz')  # Use ImportCache
            text_parts = []
            
            doc = fitz.open(pdf_path)
            self._current_doc = doc
            
            if doc.needs_pass:
                if not self._password or not doc.authenticate(self._password):
                    raise ValueError("Invalid PDF password")
            
            total_pages = len(doc)
            with tqdm(total=total_pages, desc="PyMuPDF extraction", unit="pages") as pbar:
                for page_num in range(total_pages):
                    try:
                        page = doc[page_num]
                        # Try different extraction strategies
                        page_text = page.get_text("text", sort=True)
                        if not page_text.strip():
                            # Fallback to dict extraction for complex layouts
                            page_text = page.get_text("dict")
                            if isinstance(page_text, dict):
                                page_text = self._process_text_dict(page_text)
                        
                        if page_text.strip():
                            text_parts.append(page_text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.debug(f"Page {page_num + 1} extraction failed: {e}")
                        continue
                    finally:
                        page = None
            
            return "\n\n".join(text_parts)
            
        finally:
            if self._current_doc:
                try:
                    self._current_doc.close()
                except:
                    pass
                self._current_doc = None


    def _process_text_dict(self, text_dict: Dict) -> str:
        """Process PyMuPDF dict format text"""
        text_parts = []
        try:
            for block in text_dict.get('blocks', []):
                if 'lines' in block:
                    for line in block['lines']:
                        line_text = ' '.join(
                            span.get('text', '') 
                            for span in line.get('spans', [])
                        )
                        if line_text.strip():
                            text_parts.append(line_text)
        except Exception as e:
            logging.debug(f"Text dict processing failed: {e}")
        return '\n'.join(text_parts)

    def extract_with_pdfplumber(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pdfplumber with layout preservation and progress bar"""
        pdfplumber = self._import_cache.import_module('pdfplumber')
        text_parts = []
        
        try:
            with pdfplumber.open(pdf_path, password=self._password) as pdf:
                self._current_doc = pdf
                total_pages = len(pdf.pages)
                
                with tqdm(total=total_pages, desc="pdfplumber extraction", unit="pages") as pbar:
                    for page in pdf.pages:
                        try:
                            # Extract with layout settings
                            words = page.extract_words(
                                keep_blank_chars=True,
                                use_text_flow=True,
                                horizontal_ltr=True
                            )
                            
                            if words:
                                # Group words into lines
                                lines = self._group_words_into_lines(words)
                                text_parts.append('\n'.join(' '.join(line) for line in lines))
                            else:
                                # Fallback to basic extraction
                                text = page.extract_text()
                                if text.strip():
                                    text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"Page extraction failed: {e}")
                            continue
            
            return '\n\n'.join(text_parts)
            
        finally:
            self._current_doc = None
            
    def _group_words_into_lines(self, words: List[Dict]) -> List[List[str]]:
        """Group words into lines based on positions"""
        if not words:
            return []
        
        # Sort by top position and x position
        words.sort(key=lambda w: (round(w['top']), w['x0']))
        
        lines = []
        current_line = []
        last_top = round(words[0]['top'])
        
        for word in words:
            current_top = round(word['top'])
            if abs(current_top - last_top) > 3:  # Line break threshold
                if current_line:
                    lines.append(current_line)
                current_line = []
                last_top = current_top
            current_line.append(word['text'])
        
        if current_line:
            lines.append(current_line)
        
        return lines

    def extract_with_pypdf(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pypdf with encryption support and progress bar"""
        pypdf = self._import_cache.import_module('pypdf')
        text_parts = []
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                self._current_doc = reader
                
                if reader.is_encrypted:
                    if not reader.decrypt(self._password or ""):
                        raise ValueError("PDF is encrypted and requires a valid password")
                
                total_pages = len(reader.pages)
                with tqdm(total=total_pages, desc="pypdf extraction", unit="pages") as pbar:
                    for page in reader.pages:
                        try:
                            text = page.extract_text()
                            if text.strip():
                                text_parts.append(text.strip())
                            
                            pbar.update(1)
                            if progress_callback:
                                progress_callback(1)
                                
                        except Exception as e:
                            logging.debug(f"Page extraction failed: {e}")
                            continue
            
            return '\n\n'.join(text_parts)
            
        finally:
            self._current_doc = None

    def extract_with_pdfminer(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using pdfminer with layout analysis"""
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from io import StringIO
        
        try:
            output = StringIO()
            with open(pdf_path, 'rb') as file:
                # Configure layout parameters
                laparams = LAParams(
                    line_margin=0.5,
                    word_margin=0.1,
                    char_margin=2.0,
                    boxes_flow=0.5,
                    detect_vertical=True
                )
                
                extract_text_to_fp(
                    file, output,
                    laparams=laparams,
                    password=self._password,
                    codec='utf-8'
                )
                
                text = output.getvalue()
                
                # Post-process the extracted text
                if text.strip():
                    lines = text.splitlines()
                    processed_lines = []
                    current_para = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line and current_para:
                            processed_lines.append(' '.join(current_para))
                            current_para = []
                        elif line:
                            current_para.append(line)
                    
                    if current_para:
                        processed_lines.append(' '.join(current_para))
                    
                    return '\n\n'.join(processed_lines)
            
            return ""
            
        finally:
            if 'output' in locals():
                output.close()
    
    def _init_ocr(self, method: str) -> bool:
        """Initialize OCR engine with enhanced error handling"""
        if method not in self._ocr_initialized:
            try:
                if method == 'kraken':
                    try:
                        import kraken.lib
                        import kraken.binarization
                        import kraken.pageseg
                        import kraken.recognition
                        
                        # Try alternate model loading approaches
                        try:
                            from kraken.lib import models
                            self._model = models.load_any("en-default.mlmodel")
                        except (ImportError, AttributeError):
                            try:
                                from kraken.recognition import load_any
                                self._model = load_any("en-default.mlmodel")
                            except ImportError:
                                raise ImportError("Could not find Kraken model loading function")
                                
                        self._ocr_initialized[method] = True
                        
                    except Exception as e:
                        logging.error(f"Failed to initialize Kraken: {e}")
                        self._ocr_initialized[method] = False
                        
                elif method == 'doctr':
                    import doctr
                    import torch
                    
                    # Configure torch loading for security
                    torch.set_warn_always(False)  # Suppress repeated warnings
                    
                    # Initialize DocTR with explicit device placement
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self._predictor = doctr.models.ocr_predictor(pretrained=True).to(device)
                    
                    self._ocr_initialized[method] = True
                    
                elif method == 'easyocr':
                    import easyocr
                    import torch
                    
                    # Configure GPU settings
                    gpu = torch.cuda.is_available()
                    self._reader = easyocr.Reader(
                        ['en'],
                        gpu=gpu,
                        model_storage_directory='./models',
                        download_enabled=True
                    )
                    self._ocr_initialized[method] = True
                    
                elif method == 'tesseract':
                    import pytesseract
                    import pdf2image
                    from PIL import Image
                    
                    # Verify Tesseract installation
                    try:
                        pytesseract.get_tesseract_version()
                        self._ocr_initialized[method] = True
                    except pytesseract.TesseractNotFoundError:
                        logging.error("Tesseract not found. Please install Tesseract OCR.")
                        self._ocr_initialized[method] = False
                        
            except Exception as e:
                logging.error(f"Failed to initialize {method}: {e}")
                self._ocr_initialized[method] = False
                
        return self._ocr_initialized.get(method, False)

    def extract_with_tesseract(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using Tesseract OCR with progress bars"""
        if not self._init_ocr('tesseract'):
            return ""
            
        pytesseract = self._import_cache.import_module('pytesseract')
        pdf2image = self._import_cache.import_module('pdf2image')
        PIL = self._import_cache.import_module('PIL')
        
        text_parts = []
        images = None
        
        try:
            # Convert PDF to images with progress bar
            with tqdm(desc="Converting PDF to images", unit="page") as pbar:
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=300,
                    thread_count=os.cpu_count() or 1,
                    grayscale=True,
                    size=(None, 2000)  # Limit height for memory
                )
                pbar.update(len(images))
            
            # Process images with OCR
            with tqdm(total=len(images), desc="OCR Processing", unit="page") as pbar:
                for i, image in enumerate(images, 1):
                    try:
                        # Preprocess image
                        image = self._preprocess_image(image)
                        
                        # OCR with optimized settings
                        text = pytesseract.image_to_string(
                            image,
                            config='--oem 3 --psm 3',
                            lang='+'.join(self.languages)
                        )
                        
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.error(f"OCR failed on page {i}: {e}")
                        continue
                    finally:
                        image.close()
                        
        finally:
            if images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
        
        return '\n\n'.join(text_parts)


    def _preprocess_image(self, image) -> 'PIL.Image':
        """Optimize image for OCR"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = PIL.ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply adaptive thresholding
            threshold = self._adaptive_threshold(image)
            image = image.point(lambda x: 255 if x > threshold else 0)
            
            return image
        except Exception as e:
            logging.debug(f"Image preprocessing failed: {e}")
            return image

    def _adaptive_threshold(self, image, window_size=41, constant=2):
        """Calculate adaptive threshold for image"""
        try:
            import numpy as np
            img_array = np.array(image)
            mean = np.mean(img_array)
            std = np.std(img_array)
            return int(mean - constant * std)
        except:
            return 127

    def extract_with_easyocr(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using EasyOCR with GPU support"""
        if not self._init_ocr('easyocr'):
            return ""
            
        pdf2image = self._import_cache.import_module('pdf2image')
        numpy = self._import_cache.import_module('numpy')
        
        text_parts = []
        images = None
        
        try:
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,
                thread_count=os.cpu_count() or 1,
                grayscale=True
            )
            
            for i, image in enumerate(images, 1):
                try:
                    # Convert to numpy array
                    img_array = numpy.array(image)
                    
                    # Perform OCR
                    results = self._reader.readtext(
                        img_array,
                        detail=0,
                        paragraph=True,
                        batch_size=4
                    )
                    
                    if results:
                        text_parts.append('\n'.join(results))
                    
                    if progress_callback:
                        progress_callback(1)
                        
                except Exception as e:
                    logging.error(f"EasyOCR failed on page {i}: {e}")
                    continue
                finally:
                    image.close()
                    
        finally:
            if images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
            
            # Clear GPU memory
            self._clear_gpu_memory()
        
        return '\n\n'.join(text_parts)

    def extract_with_doctr(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using DocTR with progress bars"""
        if not self._init_ocr('doctr'):
            return ""
            
        doctr = self._import_cache.import_module('doctr')
        
        try:
            with tqdm(desc="Loading document", unit="file") as pbar:
                doc = doctr.io.DocumentFile.from_pdf(pdf_path)
                pbar.update(1)
            
            with tqdm(desc="DocTR processing", unit="page") as pbar:
                result = self._predictor(doc)
                pbar.update(len(doc))
            
            text = result.render()
            
            if progress_callback:
                progress_callback(len(doc))
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"DocTR extraction failed: {e}")
            return ""
        finally:
            self._clear_gpu_memory()
            
    def _configure_torch_security(self):
        """Configure PyTorch security settings"""
        try:
            import torch
            
            # Configure secure loading
            torch.backends.cudnn.benchmark = True  # Performance optimization
            torch.set_float32_matmul_precision('medium')  # Balance of speed and precision
            
            # Set security-related configurations
            torch.set_warn_always(False)
            
            # Use safer default tensor type - updated API calls
            torch.set_default_tensor_type(torch.FloatTensor)
            torch.set_default_dtype(torch.float32)
            
            # Configure device
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    device = torch.cuda.current_device()
                    logging.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
                except Exception as e:
                    logging.warning(f"CUDA initialization failed: {e}")
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logging.info("Using MPS (Metal Performance Shaders) device")
            else:
                logging.info("Using CPU device")
                
        except Exception as e:
            logging.debug(f"PyTorch security configuration failed: {e}")
            # Continue without PyTorch security settings
            pass

    def _clear_gpu_memory(self):
        """Clear GPU memory after processing"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except:
                pass

    def extract_with_kraken(self, pdf_path: str, progress_callback=None) -> str:
        """Extract text using Kraken with progress bars"""
        if not self._init_ocr('kraken'):
            return ""
            
        kraken = self._import_cache.import_module('kraken')
        pdf2image = self._import_cache.import_module('pdf2image')
        numpy = self._import_cache.import_module('numpy')
        
        text_parts = []
        images = None
        
        try:
            # Convert PDF to images with progress
            with tqdm(desc="Converting PDF to images", unit="page") as pbar:
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=300,
                    thread_count=os.cpu_count() or 1,
                    grayscale=True
                )
                pbar.update(len(images))
            
            # Process images with progress
            with tqdm(total=len(images), desc="Kraken processing", unit="page") as pbar:
                for i, image in enumerate(images, 1):
                    try:
                        img_array = numpy.array(image)
                        
                        # Binarization with progress update
                        binarized = kraken.binarization.nlbin(img_array)
                        
                        # Segment
                        segments = kraken.pageseg.segment(binarized)
                        
                        # Recognize
                        text = kraken.rpred.rpred(self._model, img_array, segments)
                        
                        if text.strip():
                            text_parts.append(text.strip())
                        
                        pbar.update(1)
                        if progress_callback:
                            progress_callback(1)
                            
                    except Exception as e:
                        logging.error(f"Kraken failed on page {i}: {e}")
                        continue
                    finally:
                        image.close()
                        
        finally:
            if images:
                for img in images:
                    try:
                        img.close()
                    except:
                        pass
        
        return '\n\n'.join(text_parts)
    
def setup_logging(verbosity: int = 0):
    """Set up logging configuration."""
    levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }
    level = levels.get(verbosity, logging.DEBUG)
    
    format_str = '%(message)s' if verbosity == 0 else '%(levelname)s: %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pdf_extraction.log')
        ]
    )

class DocumentProcessor:
    """Main document processing coordinator"""
    
    def __init__(self, debug: bool = False):
        self.manager = ExtractionManager(debug=debug)
        self._debug = debug
        
    def process_files(self, input_files: List[str], 
                 output_dir: Optional[str] = None,
                 method: Optional[str] = None,
                 password: Optional[str] = None,
                 extract_tables: bool = False,
                 max_workers: int = None,
                 **kwargs) -> Dict[str, Any]:
        """Process multiple files with interrupt handling"""
        results = {}
        failed = []
        
        # Ensure output directory exists
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine number of workers
        max_workers = max_workers or min(len(input_files), (os.cpu_count() or 1))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            with tqdm(total=len(input_files), desc="Processing files", unit="file") as pbar:
                for input_file in input_files:
                    future = executor.submit(
                        self._process_single_file,
                        input_file,
                        output_dir,
                        method,
                        password,
                        extract_tables,
                        **kwargs
                    )
                    futures[future] = input_file
                
                for future in as_completed(futures):
                    input_file = futures[future]
                    try:
                        result = future.result()
                        results[input_file] = result
                        if not result['success']:
                            failed.append((input_file, result.get('error', 'Unknown error')))
                            if self._debug:
                                logging.error(f"Failed to process {input_file}: {result.get('error')}")
                    except Exception as e:
                        failed.append((input_file, str(e)))
                        if self._debug:
                            logging.error(f"Failed to process {input_file}: {e}")
                    finally:
                        pbar.update(1)
        
        # Print summary if in debug mode
        if self._debug:
            successful = len([r for r in results.values() if r['success']])
            logging.info(f"\nProcessing Summary:")
            logging.info(f"Total files: {len(input_files)}")
            logging.info(f"Successful: {successful}")
            logging.info(f"Failed: {len(failed)}")
            
            if failed:
                logging.info("\nFailed files:")
                for file, error in failed:
                    logging.info(f"  {file}: {error}")
        
        return {
            'results': results,
            'failed': failed
        }
    
    def _get_unique_output_path(self, input_file: str, output_dir: Optional[str] = None) -> str:
        """Generate unique output path with counter if file exists"""
        # Convert input file to absolute path
        input_path = Path(input_file).resolve()
        
        # Use current directory if none specified, ensure it's a Path
        output_dir = Path(output_dir or '.').resolve()
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get base name without extension
        base_name = input_path.stem
        
        # Start with basic output path
        counter = 0
        while True:
            if counter == 0:
                output_name = f"{base_name}.txt"
            else:
                output_name = f"{base_name}_{counter}.txt"
                
            output_path = output_dir / output_name
            
            if not output_path.exists():
                if self._debug:
                    logging.debug(f"Using output path: {output_path}")
                return str(output_path)
            
            counter += 1

    def _process_single_file(self, input_file: str,
                            output_dir: Optional[str] = None,
                            method: Optional[str] = None,
                            password: Optional[str] = None,
                            extract_tables: bool = False,
                            **kwargs) -> Dict[str, Any]:
        """Process a single document"""
        result = {
            'success': False,
            'text': '',
            'tables': [],
            'metadata': {},
            'input_file': input_file
        }
        
        try:
            # Generate unique output path
            output_path = self._get_unique_output_path(input_file, output_dir)
            
            if self._debug:
                logging.info(f"Processing {input_file} -> {output_path}")
            
            # Extract text
            text = self.manager.extract(
                input_file,
                method=method,
                password=password,
                extract_tables=extract_tables,
                **kwargs
            )
            
            if text:
                result['text'] = text
                result['success'] = True
                
                # Save to file
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    result['output_path'] = output_path
                    
                    if self._debug:
                        logging.info(f"Saved text to {output_path}")
                        
                except Exception as e:
                    logging.error(f"Failed to write output file {output_path}: {e}")
                    result['success'] = False
                    result['error'] = str(e)
                    
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logging.error(error_msg)
            result['error'] = error_msg
            result['success'] = False
            
        return result
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        metadata = {}
        try:
            file_info = Path(file_path)
            metadata.update({
                'filename': file_info.name,
                'size': file_info.stat().st_size,
                'modified': file_info.stat().st_mtime
            })
            
            # Extract document-specific metadata
            if file_info.suffix.lower() == '.pdf':
                metadata.update(self._extract_pdf_metadata(file_path))
            elif file_info.suffix.lower() == '.epub':
                metadata.update(self._extract_epub_metadata(file_path))
                
        except Exception as e:
            if self._debug:
                logging.error(f"Metadata extraction failed: {e}")
                
        return metadata

def main():
    import glob
    
    """Command-line interface entry point"""
    parser = argparse.ArgumentParser(
        description="Document Text Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s input.pdf
              %(prog)s -o output_dir/ *.pdf
              %(prog)s -m pymupdf -p password input.pdf
              %(prog)s -t -j output.json *.pdf
        """)
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help="Input files to process (supports wildcards)"
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        help="Output directory for extracted text files (default: current directory)",
        default='.'  # Default to current directory
    )
    
    parser.add_argument(
        '-m', '--method',
        help="Preferred extraction method"
    )
    
    parser.add_argument(
        '-p', '--password',
        help="Password for encrypted documents"
    )
    
    parser.add_argument(
        '-t', '--tables',
        action='store_true',
        help="Extract tables (PDF only)"
    )
    
    parser.add_argument(
        '-j', '--json',
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        help="Maximum number of worker threads"
    )
    
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Expand file patterns
        input_files = []
        for pattern in args.files:
            matched_files = glob.glob(pattern)
            if matched_files:
                input_files.extend(matched_files)
            else:
                logging.warning(f"No files found matching pattern: {pattern}")
        
        if not input_files:
            logging.error("No input files found")
            return 1
            
        # Initialize processor
        processor = DocumentProcessor(debug=args.debug)
        
        try:
            results = processor.process_files(
                input_files,
                output_dir=args.output_dir,
                method=args.method,
                password=args.password,
                extract_tables=args.tables,
                max_workers=args.workers
            )
            
            # Handle results
            if args.json:
                import json
                with open(args.json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            
            return 0 if not results.get('failed') else 1
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return 130
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    # Set up signal handlers
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(130))
    
    sys.exit(main())
