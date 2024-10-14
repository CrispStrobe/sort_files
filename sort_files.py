#!/usr/bin/env python3
import os
import sys
import re
import argparse
import subprocess
import shutil
import tempfile
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
import platform
import time
from types import MappingProxyType

# Attempt to import OpenAI client
try:
    from openai import OpenAI
except ImportError:
    print("OpenAI client library is not installed. Please install it using 'pip install openai'.")
    sys.exit(1)

class ProgressBar:
    """
    A class to handle progress bar operations using tqdm.
    
    If tqdm is unavailable, it gracefully degrades without displaying a progress bar.
    """
    def __init__(self, total, desc="Processing Files", enable=True):
        """
        Initialize the ProgressBar.
        
        Parameters:
            total (int): Total number of tasks/files to process.
            desc (str): Description to display alongside the progress bar.
            enable (bool): Whether to enable the progress bar.
        """
        self.enable = enable
        self.lock = threading.Lock()
        self.bar = None
        if self.enable:
            try:
                from tqdm import tqdm
                self.bar = tqdm(total=total, desc=desc, unit="file")
                logging.debug("Progress bar initialized using tqdm.")
            except ImportError:
                logging.warning("tqdm is not installed. Progress bar will be disabled.")
                self.bar = None
    
    def update(self, n=1):
        """
        Update the progress bar by a specified increment.
        
        Parameters:
            n (int): Number of increments to add.
        """
        if self.enable and self.bar:
            with self.lock:
                self.bar.update(n)
    
    def close(self):
        """
        Close the progress bar.
        """
        if self.enable and self.bar:
            with self.lock:
                self.bar.close()
                logging.debug("Progress bar closed.")

# Global lock for thread-safe file operations
file_lock = threading.Lock()
thread_local = threading.local()

# Configuration Constants
MODEL_NAME = "cas/spaetzle-v85-7b"  # Ensure this is defined before usage
MAX_RETRY_ATTEMPTS = 5
SUPPORTED_EXTENSIONS = (
    '.csv', '.doc', '.docx', '.eml', '.epub', '.gif', '.jpg', '.jpeg', '.json', 
    '.html', '.htm', '.mp3', '.msg', '.odt', '.ogg', '.pdf', '.png', '.pptx', 
    '.ps', '.rtf', '.tiff', '.tif', '.txt', '.wav', '.xlsx', '.xls', '.ppt', 
    '.ods', '.odp', '.md'
)
OCR_PREFERRED_ORDER = ['tesseract', 'kraken']  # Define preferred OCR methods

# Define required Tesseract languages
REQUIRED_TESS_LANGS = ['eng', 'deu', 'ara']

# Custom FileHandler to flush immediately
class FlushImmediatelyFileHandler(logging.FileHandler):
    """
    Custom FileHandler that flushes the buffer immediately after each log record.
    """
    def emit(self, record):
        super().emit(record)
        self.flush()

def get_openai_client():
    if not hasattr(thread_local, "client"):
        try:
            thread_local.client = OpenAI(
                base_url="http://localhost:11434/v1/",
                api_key="ollama"
            )
            logging.debug("OpenAI client initialized successfully for thread.")
        except Exception as e:
            logging.critical(f"Failed to initialize OpenAI client in thread: {e}")
            raise
    return thread_local.client

def setup_logging(verbose, debug):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Setup stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    if debug:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Setup file handler with thread-safe rotation
    log_filename = None
    file_level = None

    if verbose and not debug:
        log_filename = "script_debug.log"
        file_level = logging.DEBUG
    elif not verbose and not debug:
        log_filename = "script.log"
        file_level = logging.INFO
    else:
        log_filename = None

    if log_filename:
        try:
            file_handler = FlushImmediatelyFileHandler(log_filename, mode='a')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Failed to initialize FlushImmediatelyFileHandler: {e}. Falling back to RotatingFileHandler.")
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5)
                file_handler.setLevel(file_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as rot_e:
                logging.critical(f"Failed to initialize RotatingFileHandler: {rot_e}. No file logging will be available.")


    # Suppress external library logs unless in debug mode
    openai_logger = logging.getLogger('openai')
    if debug:
        openai_logger.setLevel(logging.DEBUG)
    else:
        openai_logger.setLevel(logging.WARNING)


# Define a shutdown flag for graceful termination
shutdown_flag = threading.Event()

# Signal handler to set the shutdown flag
def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_flag.set()

# Register signal handlers for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Attempt to import essential and optional libraries and set availability flags
essential_libraries = {
    'textract': False,
    'openai': True  # Already imported above
}

# Import pdfminer if available
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfparser import PDFSyntaxError
    optional_libraries_pdfminer = True
    logging.debug("pdfminer is available for PDF extraction.")
except ImportError:
    optional_libraries_pdfminer = False
    logging.warning("pdfminer.six is not installed. PDFMiner-based extraction will be skipped.")
except Exception as e:
    optional_libraries_pdfminer = False
    logging.error(f"Unexpected error importing pdfminer: {e}")
    logging.warning("PDFMiner-based extraction will be skipped.")

# Import textract
try:
    import textract
    essential_libraries['textract'] = True
    logging.debug("textract is available for text extraction.")
except ImportError:
    logging.warning("textract is not installed. PDF extraction will fallback to PyPDF2 if available.")

# Import optional libraries with warnings if not available
optional_libraries = {
    'tqdm': False,
    'ebooklib': False,
    'beautifulsoup4': False,
    'docx': False,
    'mobi': False,
    'magic': False,
    'PyPDF2': False,
    'pdf2image': False,
    'pytesseract': False,
    'pdfminer': optional_libraries_pdfminer,
    'pillow': False 
}

# Attempt to import Pillow (PIL)
try:
    from PIL import Image
    optional_libraries['pillow'] = True
    logging.debug("Pillow (PIL) is available for image processing.")
except ImportError:
    optional_libraries['pillow'] = False
    logging.warning("Pillow (PIL) is not installed. Image processing will be skipped or limited.")

try:
    from ebooklib import epub
    from ebooklib import ITEM_DOCUMENT as EPUB_ITEM_DOCUMENT
    optional_libraries['ebooklib'] = True
    logging.debug("ebooklib is available for EPUB processing.")
except ImportError:
    logging.warning("ebooklib is not installed. EPUB and MOBI file processing will be skipped.")

try:
    from bs4 import BeautifulSoup
    optional_libraries['beautifulsoup4'] = True
    logging.debug("beautifulsoup4 is available for HTML parsing.")
except ImportError:
    logging.warning("beautifulsoup4 is not installed. HTML and EPUB text extraction will be limited.")

try:
    import docx
    optional_libraries['docx'] = True
    logging.debug("python-docx is available for DOCX processing.")
except ImportError:
    logging.warning("python-docx is not installed. DOCX file processing will be skipped.")

try:
    import mobi
    optional_libraries['mobi'] = True
    logging.debug("mobi library is available for MOBI processing.")
except ImportError:
    logging.warning("mobi library is not installed. MOBI and AZW file processing will be skipped.")

try:
    import magic
    optional_libraries['magic'] = True
    logging.debug("python-magic is available for MIME type detection.")
except ImportError:
    logging.warning("python-magic is not installed. MIME type detection will be skipped.")

try:
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PdfReadError
    optional_libraries['PyPDF2'] = True
    logging.debug("PyPDF2 is available for PDF extraction.")
except ImportError:
    logging.warning("PyPDF2 is not installed. PDF files will be skipped.")
    optional_libraries['PyPDF2'] = False

try:
    from pdf2image import convert_from_path
    optional_libraries['pdf2image'] = True
    logging.debug("pdf2image is available for converting PDFs to images.")
except ImportError:
    logging.warning("pdf2image is not installed. OCR on specific pages will be skipped.")
    optional_libraries['pdf2image'] = False

try:
    import pytesseract
    optional_libraries['pytesseract'] = True
    logging.debug("pytesseract is available for OCR.")
except ImportError:
    logging.warning("pytesseract is not installed. OCR functionality will be limited.")
    optional_libraries['pytesseract'] = False

def prompt_install_tool(tool_name, install_instructions, auto_install=False, no_install=False):
    """
    Prompt the user to install a missing tool.

    Parameters:
        tool_name (str): The name of the tool to install.
        install_instructions (list): The command to install the tool.
        auto_install (bool): If True, attempt to install without prompting.
        no_install (bool): If True, do not attempt to install even if prompted.

    Returns:
        bool: True if installation succeeded, False otherwise.
    """
    if no_install:
        logging.info(f"Installation of {tool_name} is disabled via --no-install flag.")
        return False
    if auto_install:
        user_input = 'y'
    else:
        try:
            user_input = input(f"{tool_name} is not installed. Do you want to install it now? [y/N]: ").strip().lower()
        except EOFError:
            logging.warning(f"Input not available. Skipping installation of {tool_name}.")
            return False

    if user_input == 'y':
        try:
            subprocess.run(install_instructions, check=True)
            logging.info(f"Successfully installed {tool_name}.")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {tool_name}: {e}")
    else:
        logging.info(f"{tool_name} is required for OCR functionalities. Proceeding without it.")
    return False


def find_tessdata_dirs():
    """
    Search for tessdata directories in common locations.
    Returns a list of found tessdata directories.
    """
    common_paths = [
        '/usr/local/share/tessdata',
        '/usr/share/tesseract-ocr/4.00/tessdata',
        '/usr/share/tesseract-ocr/tessdata',
        '/opt/homebrew/share/tessdata',
        '/usr/local/Cellar/tesseract/5.4.1/share/tessdata',  # Adjust based on your Tesseract version
    ]
    tessdata_dirs = []

    # First, check if TESSDATA_PREFIX is set in the environment
    env_prefix = os.environ.get('TESSDATA_PREFIX')
    if env_prefix and os.path.isdir(env_prefix):
        tessdata_dirs.append(env_prefix)
        logging.debug(f"Found tessdata directory from TESSDATA_PREFIX: {env_prefix}")
    
    # Search common paths if not found in environment
    if not tessdata_dirs:
        for path in common_paths:
            if os.path.isdir(path):
                tessdata_dirs.append(path)
                logging.debug(f"Found tessdata directory: {path}")
    
    # Optionally, search the entire system (commented out for performance)
    # for root, dirs, files in os.walk('/'):
    #     if 'tessdata' in dirs:
    #         tessdata_dir = os.path.join(root, 'tessdata')
    #         tessdata_dirs.append(tessdata_dir)
    #         logging.debug(f"Found tessdata directory via system-wide search: {tessdata_dir}")
    
    return tessdata_dirs

def check_tesseract_langs(required_langs):
    """
    Check if the required Tesseract language data files are present.
    """
    try:
        langs = subprocess.check_output(['tesseract', '--list-langs'], stderr=subprocess.STDOUT).decode('utf-8')
        missing_langs = [lang for lang in required_langs if lang not in langs]
        if missing_langs:
            logging.error(f"Missing Tesseract language data files: {', '.join(missing_langs)}")
            return False
        logging.debug("All required Tesseract language data files are present.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking Tesseract languages: {e}")
        return False
    except FileNotFoundError:
        logging.error("Tesseract is not installed or not found in PATH.")
        return False
    except Exception as e:
        logging.error(f"Unexpected error checking Tesseract languages: {e}")
        return False

def get_installed_tesseract_langs():
    """
    Retrieve a list of installed Tesseract languages.
    """
    try:
        langs_output = subprocess.check_output(['tesseract', '--list-langs'], stderr=subprocess.STDOUT).decode('utf-8')
        langs = langs_output.splitlines()
        if len(langs) > 1:
            return langs[1:]
        return []
    except Exception as e:
        logging.error(f"Error retrieving installed Tesseract languages: {e}")
        return []

# Function to check and install required Tesseract languages
def check_and_install_tesseract_langs(required_langs):
    """
    Check if the required Tesseract language data files are present.
    If not, attempt to find tessdata directories and prompt the user to install missing languages.
    """
    tessdata_dirs = find_tessdata_dirs()

    if not tessdata_dirs:
        logging.warning("Could not find 'tessdata' directory. Please ensure Tesseract OCR is correctly installed.")
        return False

    # Check for each required language
    missing_langs = []
    for lang in required_langs:
        lang_file = f"{lang}.traineddata"
        found = False
        for tessdata_dir in tessdata_dirs:
            if os.path.isfile(os.path.join(tessdata_dir, lang_file)):
                found = True
                break
        if not found:
            missing_langs.append(lang)

    if not missing_langs:
        logging.debug("All required Tesseract language data files are present.")
        return True

    logging.warning(f"Missing Tesseract language data files: {', '.join(missing_langs)}")

    # Determine OS
    current_os = platform.system()
    install_cmd = None
    if current_os == "Darwin":  # macOS
        logging.info("Detected macOS.")
        if shutil.which('brew') is None:
            logging.error("Homebrew is not installed. Please install Homebrew to proceed.")
            return False
        # Construct brew install commands for missing languages
        brew_langs = [f"tesseract-lang-{lang}" for lang in missing_langs]
        install_cmd = ['brew', 'install'] + brew_langs
    elif current_os == "Linux":
        logging.info("Detected Linux.")
        # Attempt to detect package manager
        if shutil.which('apt-get'):
            install_cmd = ['sudo', 'apt-get', 'install', '-y'] + [f"tesseract-ocr-{lang}" for lang in missing_langs]
        elif shutil.which('yum'):
            install_cmd = ['sudo', 'yum', 'install', '-y'] + [f"tesseract-langpack-{lang}" for lang in missing_langs]
        else:
            logging.error("Unsupported package manager. Please install the required Tesseract language data manually.")
            return False
    else:
        logging.error(f"Unsupported operating system: {current_os}. Please install the required Tesseract language data manually.")
        return False

    # Prompt user for installation
    user_input = 'n'
    try:
        user_input = input(f"Do you want to install the missing Tesseract language data ({', '.join(missing_langs)})? [y/N]: ").strip().lower()
    except EOFError:
        logging.warning("Input not available. Skipping installation of Tesseract languages.")
        return False

    if user_input == 'y':
        try:
            logging.info(f"Installing Tesseract language data: {' '.join(missing_langs)}")
            subprocess.run(install_cmd, check=True)
            logging.info("Tesseract language data installation completed.")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing Tesseract language data: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during installation: {e}")
    else:
        logging.info("User opted not to install the missing Tesseract language data. OCR functionality may be limited.")
    return False

# Text extraction functions for various file types
def extract_text_with_pdfminer(pdf_path, verbose=False):
    if not optional_libraries['pdfminer']:
        logging.warning(f"pdfminer is not available. Skipping PDFMiner extraction for {pdf_path}.")
        return ""
    logging.debug(f"Extracting text from PDF using pdfminer: {pdf_path}")
    from pdfminer.pdfparser import PDFSyntaxError
    try:
        text = pdfminer_extract_text(pdf_path, maxpages=4)
        logging.debug(f"Successfully extracted text using pdfminer ({len(text)} characters)")
        if verbose:
            logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
        return text[:2000]
    except PDFSyntaxError as e:
        logging.error(f"PDF syntax error in {pdf_path}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path} with pdfminer: {e}")
        return ""


def perform_ocr_on_image(image, ocr_available=None, tools_available=None, optional_libraries=None, verbose=False):
    """
    Perform OCR on an image using available OCR tools.
    
    Parameters:
        image (PIL.Image.Image): PIL Image object.
        ocr_available (dict): Availability of OCR tools.
        tools_available (dict): Dictionary indicating availability of tools.
        optional_libraries (dict): Dictionary indicating availability of optional libraries.
        verbose (bool): Enable verbose logging.
    
    Returns:
        str: Extracted text from OCR or empty string if OCR fails.
    """
    text = ""
    if ocr_available is None:
        ocr_available = {}
    if tools_available is None:
        tools_available = {}
    if optional_libraries is None:
        optional_libraries = {}
    
    # Determine available OCR methods based on preferred order
    available_ocr_methods = [method for method in OCR_PREFERRED_ORDER if ocr_available.get(method, False)]
    if not available_ocr_methods:
        logging.warning("No OCR tools are available to perform OCR on the image.")
        return text
    
    for ocr_method in available_ocr_methods:
        if ocr_method == 'tesseract' and image:
            lang = '+'.join([lang for lang in REQUIRED_TESS_LANGS if lang in get_installed_tesseract_langs()])
            logging.debug(f"Performing OCR on image with Tesseract using languages: {lang}")
            try:
                page_text = pytesseract.image_to_string(image, lang=lang)
                if page_text.strip():
                    text += page_text + "\n"
                    logging.debug(f"Successfully extracted text using Tesseract ({len(page_text)} characters)")
                    if verbose:
                        logging.debug(f"Extracted Text (first 500 chars): {page_text[:500]}")
                    break  # Successful OCR, no need to try other methods
            except Exception as e:
                logging.error(f"Error performing OCR with Tesseract on image: {e}")
                continue  # Try next OCR method
        elif ocr_method == 'kraken' and image:
            logging.debug("Performing OCR on image with Kraken.")
            page_text = perform_ocr_with_kraken(image, tools_available=tools_available, verbose=verbose)
            if page_text.strip():
                text += page_text + "\n"
                break  # Successful OCR, no need to try other methods
        elif ocr_method == 'tesseract' and not image:
            # If Pillow is not available, skip Tesseract in this function
            logging.debug("Image object is not available. Skipping Tesseract in perform_ocr_on_image.")
            continue
    return text


def extract_text_from_image(image_path, perform_ocr_if_needed, ocr_available, verbose=False, tools_available=None, optional_libraries=None):
    """
    Extract text from an image file using textract and fallback OCR methods.
    
    Parameters:
        image_path (str): Path to the image file.
        perform_ocr_if_needed (bool): Whether to perform OCR if textract fails.
        ocr_available (dict): Availability of OCR tools.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
        optional_libraries (dict): Dictionary indicating availability of optional libraries.
    
    Returns:
        str: Extracted text or empty string if extraction fails.
    """
    text = ""
    if tools_available is None:
        tools_available = {}
    if optional_libraries is None:
        optional_libraries = {}
    
    # Attempt text extraction with textract first
    if tools_available.get('textract', False):
        logging.debug(f"Extracting text from image using textract: {image_path}")
        try:
            result = textract.process(image_path)
            if result:
                text = result.decode('utf-8').strip()
                if text:
                    logging.debug(f"Successfully extracted text using textract ({len(text)} characters)")
                    if verbose:
                        logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
                    logging.info(f"Extracted text from {image_path} using textract.")
                    return text[:2000]
        except Exception as e:
            logging.error(f"Error extracting text from {image_path} using textract: {e}")
    
    # Proceed with OCR if needed
    if perform_ocr_if_needed and (ocr_available.get('tesseract', False) or ocr_available.get('kraken', False)):
        logging.debug("Attempting OCR due to textract failure or insufficiency.")
        
        image = None
        if tools_available.get('pillow', False):
            try:
                image = Image.open(image_path)
            except Exception as e:
                logging.error(f"Error opening image {image_path} for OCR: {e}")
                image = None
        
        if image:
            text = perform_ocr_on_image(
                image, 
                ocr_available, 
                tools_available=tools_available, 
                optional_libraries=optional_libraries, 
                verbose=verbose
            )
        else:
            # Fall back to direct Tesseract subprocess if Pillow is not available or image failed to open
            logging.debug("Pillow is not available or failed to open image. Performing OCR via Tesseract subprocess.")
            try:
                lang = '+'.join([lang for lang in REQUIRED_TESS_LANGS if lang in get_installed_tesseract_langs()])
                cmd = ['tesseract', image_path, 'stdout', '-l', lang]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                text = result.stdout.strip()
                logging.debug(f"Successfully extracted text using Tesseract subprocess ({len(text)} characters)")
                if verbose:
                    logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error performing OCR with Tesseract subprocess on {image_path}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during OCR with Tesseract subprocess on {image_path}: {e}")
        
        # Validate the extracted text
        if is_text_meaningful(text, file_path=image_path):
            logging.info(f"Extracted meaningful text from {image_path} using OCR.")
            return text[:2000]
        else:
            logging.warning(f"OCR extraction from {image_path} did not yield meaningful text.")
    
    logging.warning(f"Failed to extract meaningful text from image: {image_path}.")
    return ""


def extract_text_from_docx(docx_path, verbose=False, tools_available=None):
    """
    Extract text from a DOCX file using python-docx.
    Falls back to textract if python-docx is unavailable or fails.
    
    Parameters:
        docx_path (str): Path to the DOCX file.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if extraction fails.
    """
    if tools_available is None:
        tools_available = {}
    
    if not tools_available.get('docx', False):
        logging.warning(f"DOCX processing is skipped for {docx_path} as python-docx is not available.")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(docx_path, verbose=verbose)
        else:
            logging.error("Textract is also unavailable. Cannot extract text from DOCX.")
            return ""
    
    logging.debug(f"Extracting text from DOCX: {docx_path}")
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
            if len(text) > 3000:
                break
        logging.debug(f"Extracted {len(text)} characters from DOCX")
        if verbose:
            logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
    except Exception as e:
        logging.error(f"Error extracting text from {docx_path}: {e}")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            text = extract_text_with_textract(docx_path, verbose=verbose)
    return text[:2000]


def extract_file_metadata(file_path, verbose=False):
    logging.debug(f"Extracting file metadata for: {file_path}")
    metadata = {}
    try:
        if optional_libraries['magic']:
            file_type = magic.from_file(file_path, mime=True)
            metadata['file_type'] = file_type
        else:
            metadata['file_type'] = "Unknown"

        metadata['file_size'] = os.path.getsize(file_path)

        creation_time = os.path.getctime(file_path)
        modification_time = os.path.getmtime(file_path)
        metadata['creation_time'] = datetime.fromtimestamp(creation_time).isoformat()
        metadata['modification_time'] = datetime.fromtimestamp(modification_time).isoformat()

        logging.debug(f"Extracted metadata: {metadata}")

    except Exception as e:
        logging.error(f"Error extracting metadata from {file_path}: {e}")

    return metadata

def is_text_meaningful(text, min_length=200, min_unique_words=20, alnum_ratio_threshold=0.3, max_repetition_ratio=0.3, file_path=""):
    if len(text) < min_length:
        logging.debug(f"{file_path} - Text length {len(text)} is below the minimum threshold of {min_length}.")
        return False

    alnum_count = sum(c.isalnum() for c in text)
    alnum_ratio = alnum_count / len(text)
    if alnum_ratio < alnum_ratio_threshold:
        logging.debug(f"{file_path} - Alphanumeric ratio {alnum_ratio:.2f} is below the threshold of {alnum_ratio_threshold}.")
        return False

    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        logging.debug(f"{file_path} - No words found in text.")
        return False

    unique_words = set(words)
    if len(unique_words) < min_unique_words:
        logging.debug(f"{file_path} - Unique words count {len(unique_words)} is below the threshold of {min_unique_words}.")
        return False

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    max_repetition = max(word_counts.values()) / len(words)
    if max_repetition > max_repetition_ratio:
        logging.debug(f"{file_path} - Max word repetition ratio {max_repetition:.2f} exceeds the threshold of {max_repetition_ratio}.")
        return False

    return True


def extract_text_from_mobi(mobi_path, verbose=False, tools_available=None):
    """
    Extract text from a MOBI file using the mobi library.
    Falls back to textract if mobi is unavailable or fails.
    
    Parameters:
        mobi_path (str): Path to the MOBI file.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if extraction fails.
    """
    if tools_available is None:
        tools_available = {}
    
    if not tools_available.get('mobi', False):
        logging.warning(f"MOBI processing is skipped for {mobi_path} as mobi library is not available.")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(mobi_path, verbose=verbose)
        else:
            logging.error("Textract is also unavailable. Cannot extract text from MOBI.")
            return ""
    
    logging.debug(f"Extracting text from MOBI: {mobi_path}")
    text = ""
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = mobi.extract(mobi_path, tempdir=tempdir)
            logging.debug(f"mobi.extract returned filepath: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                logging.debug(f"Successfully extracted text using mobi library ({len(text)} characters)")
                if verbose:
                    logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
    except Exception as e:
        logging.error(f"Error extracting text from {mobi_path} using mobi library: {e}")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(mobi_path, verbose=verbose)
    return text[:2000]


def extract_text_from_djvu(djvu_path, djvutxt_installed, verbose=False, tools_available=None):
    """
    Extract text from a DJVU file using djvutxt.
    Falls back to textract if djvutxt is unavailable or fails.
    
    Parameters:
        djvu_path (str): Path to the DJVU file.
        djvutxt_installed (bool): Whether djvutxt is installed.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if extraction fails.
    """
    if tools_available is None:
        tools_available = {}
    
    if not djvutxt_installed:
        logging.warning(f"Skipping DJVU file {djvu_path} as 'djvutxt' is not available.")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(djvu_path, verbose=verbose)
        else:
            logging.error("Textract is also unavailable. Cannot extract text from DJVU.")
            return ""
    
    logging.debug(f"Extracting text from DJVU: {djvu_path}")
    text = ""
    try:
        result = subprocess.run(['djvutxt', djvu_path], capture_output=True, text=True, check=True)
        text = result.stdout[:2000]
        logging.debug(f"Successfully extracted {len(text)} characters from DJVU")
        if verbose:
            logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting text from {djvu_path}: {e}")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(djvu_path, verbose=verbose)
    except FileNotFoundError:
        logging.error("djvutxt command not found. Please install DjVuLibre package.")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(djvu_path, verbose=verbose)
    return text[:2000]


def extract_text_from_azw(azw_path, verbose=False, tools_available=None):
    """
    Extract text from an AZW file using mobi library and EbookLib.
    Falls back to textract if necessary.
    
    Parameters:
        azw_path (str): Path to the AZW file.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if extraction fails.
    """
    if tools_available is None:
        tools_available = {}
    
    if not tools_available.get('mobi', False):
        logging.warning(f"AZW processing is skipped for {azw_path} as mobi library is not available.")
        if tools_available.get('textract', False):
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(azw_path, verbose=verbose)
        else:
            logging.error("Textract is also unavailable. Cannot extract text from AZW.")
            return ""
    
    logging.debug(f"Extracting text from AZW: {azw_path}")
    text = ""
    if tools_available.get('textract', False):
        logging.debug("Attempting to use textract for AZW extraction...")
        try:
            text = textract.process(azw_path).decode('utf-8')
            if text.strip():
                logging.debug(f"Successfully extracted text using textract ({len(text)} characters)")
                if verbose:
                    logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
                logging.info(f"Extracted text from {azw_path} using textract.")
                return text[:2000]
        except Exception as e:
            logging.error(f"Error extracting text from {azw_path} using textract: {e}")
    
    if tools_available.get('ebooklib', False):
        logging.debug("Attempting to use EbookLib for AZW extraction...")
        try:
            book = epub.read_epub(azw_path)
            for item in book.get_items():
                if item.get_type() == EPUB_ITEM_DOCUMENT:
                    if tools_available.get('beautifulsoup4', False):
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text += soup.get_text() + "\n"
                    else:
                        logging.warning("beautifulsoup4 is not available. Skipping HTML parsing.")
                    if len(text) > 3000:
                        break
            if text.strip():
                logging.debug(f"Successfully extracted text using EbookLib ({len(text)} characters)")
                if verbose:
                    logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
                logging.info(f"Extracted text from {azw_path} using EbookLib.")
                return text[:2000]
        except Exception as e:
            logging.error(f"Error extracting text from {azw_path} using EbookLib: {e}")
    
    if tools_available.get('mobi', False):
        logging.debug("Attempting to use mobi library for AZW extraction...")
        tempdir = None  # Ensure tempdir is initialized
        try:
            tempdir, filepath = mobi.extract(azw_path)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    logging.debug(f"Successfully extracted text using mobi library ({len(text)} characters)")
                    if verbose:
                        logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
            finally:
                if tempdir:
                    shutil.rmtree(tempdir)
            if text.strip():
                return text[:2000]
            else:
                logging.warning(f"No meaningful text extracted from {azw_path} using mobi library.")
                if tools_available.get('textract', False):
                    logging.info("Attempting to extract text using textract as a fallback.")
                    return extract_text_with_textract(azw_path, verbose=verbose)
        except Exception as e:
            logging.error(f"Error extracting text from {azw_path} using mobi library: {e}")
            if tempdir:
                try:
                    shutil.rmtree(tempdir)
                except Exception as cleanup_e:
                    logging.error(f"Error removing temporary directory {tempdir}: {cleanup_e}")
            if tools_available.get('textract', False):
                logging.info("Attempting to extract text using textract as a fallback.")
                return extract_text_with_textract(azw_path, verbose=verbose)
    return text[:2000]


def extract_text_from_pdf(pdf_path, perform_ocr_if_needed, ocr_available, verbose=False, tools_available=None):
    """
    Extract text from a PDF file using textract, pdfminer, and PyPDF2.
    Falls back to OCR if necessary.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        perform_ocr_if_needed (bool): Whether to perform OCR if text extraction is insufficient.
        ocr_available (dict): Availability of OCR tools.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if extraction fails.
    """
    if tools_available is None:
        tools_available = {}
    text = ""
    extraction_methods = []
    
    if tools_available.get('textract', False):
        extraction_methods.append('textract')
    if tools_available.get('pdfminer', False):
        extraction_methods.append('pdfminer')
    if tools_available.get('PyPDF2', False):
        extraction_methods.append('PyPDF2')
    
    for method in extraction_methods:
        if shutdown_flag.is_set():
            logging.info(f"Shutdown initiated. Stopping extraction for {pdf_path}.")
            return ""
        if method == 'textract':
            logging.debug(f"Extracting text from PDF using textract: {pdf_path}")
            try:
                text = textract.process(pdf_path).decode('utf-8')
                if text.strip():
                    logging.debug(f"Successfully extracted text using textract ({len(text)} characters)")
                    if verbose:
                        logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
                    logging.info(f"Extracted text from {pdf_path} using textract.")
                    return text[:2000]
            except Exception as e:
                logging.error(f"Error extracting text from {pdf_path} using textract: {e}")
        elif method == 'pdfminer':
            text = extract_text_with_pdfminer(pdf_path, verbose=verbose)
            if is_text_meaningful(text, file_path=pdf_path):
                logging.info(f"Extracted meaningful text from {pdf_path} using pdfminer.")
                return text[:2000]
            else:
                logging.warning(f"{pdf_path} - Text extracted by pdfminer is not meaningful.")
        elif method == 'PyPDF2':
            logging.debug(f"Attempting to extract text from PDF using PyPDF2: {pdf_path}")
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PdfReader(file)
                    for page_num in range(min(4, len(reader.pages))):
                        logging.debug(f"Reading page: {page_num}")
                        page = reader.pages[page_num]
                        extracted_text = page.extract_text() or ''
                        if extracted_text:
                            text += extracted_text
                        if len(text) > 3000:
                            break
                if is_text_meaningful(text, file_path=pdf_path):
                    logging.debug(f"Successfully extracted text using PyPDF2 ({len(text)} characters)")
                    if verbose:
                        logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
                    logging.info(f"Extracted meaningful text from {pdf_path} using PyPDF2.")
                    return text[:2000]
            except (PdfReadError, ValueError, TypeError) as e:
                logging.error(f"Error extracting text from {pdf_path} with PyPDF2: {e}.")
    
    # If all extraction methods failed, attempt OCR
    if perform_ocr_if_needed and (ocr_available.get('tesseract', False) or ocr_available.get('kraken', False)):
        logging.debug("Attempting OCR due to insufficient text extraction.")
        text = perform_ocr_on_first_pages(pdf_path, num_pages=3, verbose=verbose, tools_available=tools_available)
        if is_text_meaningful(text, file_path=pdf_path):
            logging.info(f"Extracted meaningful text from {pdf_path} using OCR.")
            return text[:2000]
        else:
            logging.warning(f"OCR extraction from {pdf_path} did not yield meaningful text.")
    
    return text[:2000]


def perform_ocr_with_kraken(image, tools_available=None, verbose=False):
    """
    Perform OCR using Kraken on a given image.
    
    Parameters:
        image (PIL.Image.Image): PIL Image object.
        tools_available (dict): Dictionary indicating availability of tools.
        verbose (bool): Enable verbose logging.
    
    Returns:
        str: Extracted text from Kraken OCR or empty string if OCR fails.
    """
    if tools_available is None:
        tools_available = {}
    if not tools_available.get('kraken', False):
        logging.warning("Kraken OCR is not available. Skipping Kraken OCR.")
        return ""
    try:
        from kraken import binarization, pageseg, ocr
        if verbose:
            logging.debug("Performing OCR using Kraken.")
        # Binarize the image
        bin_img = binarization.nlbin(image)
        # Segment the image into lines
        regions = pageseg.segment(bin_img)
        # Perform OCR
        text = ocr.recognize(bin_img, regions=regions)
        if not isinstance(text, str):
            logging.error("Kraken OCR did not return a string.")
            return ""
        logging.debug(f"Successfully extracted text using Kraken ({len(text)} characters)")
        if verbose:
            logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
        return text
    except ImportError:
        logging.error("Kraken OCR dependencies are missing. Please install Kraken and its dependencies.")
    except Exception as e:
        logging.error(f"Error performing OCR with Kraken: {e}")
    return ""


def perform_ocr_on_first_pages(pdf_path, num_pages=3, verbose=False, tools_available=None):
    """
    Perform OCR on the first few pages of a PDF file using available OCR tools.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        num_pages (int): Number of pages to perform OCR on.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if OCR fails.
    """
    if tools_available is None:
        tools_available = {}
    if not (tools_available.get('pdf2image', False) and (tools_available.get('tesseract', False) or tools_available.get('kraken', False))):
        logging.warning("pdf2image and/or no OCR tool is available. OCR on specific pages will be skipped.")
        return ""
    logging.debug(f"Performing OCR on the first {num_pages} pages of: {pdf_path}")
    text = ""
    try:
        # Dynamically find tessdata directories and set TESSDATA_PREFIX
        tessdata_dirs = find_tessdata_dirs()
        if tessdata_dirs:
            TESSDATA_PREFIX = tessdata_dirs[0]  # Use the first found tessdata directory
            os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
            logging.debug(f"Set TESSDATA_PREFIX to: {TESSDATA_PREFIX}")
        else:
            logging.warning("Could not find 'tessdata' directory. Skipping setting the environment variable for pytesseract.")
            TESSDATA_PREFIX = None

        # Determine available OCR methods based on preferred order
        available_ocr_methods = [method for method in OCR_PREFERRED_ORDER if tools_available.get(method, False)]
        if not available_ocr_methods:
            logging.warning("No OCR tools are available to perform OCR.")
            return ""

        # Check if required Tesseract languages are installed
        if 'tesseract' in available_ocr_methods and not check_tesseract_langs(REQUIRED_TESS_LANGS):
            logging.warning("Required Tesseract languages are missing. Tesseract OCR will be skipped.")
            available_ocr_methods.remove('tesseract')

        # Determine languages to use
        installed_langs = get_installed_tesseract_langs()
        installed_langs = [lang for lang in REQUIRED_TESS_LANGS if lang in installed_langs]
        if not installed_langs and 'tesseract' in available_ocr_methods:
            installed_langs = REQUIRED_TESS_LANGS  # Fallback to all required languages
            logging.warning("No specific Tesseract languages found. Using all required languages.")
        lang = '+'.join(installed_langs) if installed_langs else ''

        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=num_pages)
        for i, image in enumerate(images):
            for ocr_method in available_ocr_methods:
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown initiated. Stopping OCR on {pdf_path}.")
                    return text[:2000]
                if ocr_method == 'tesseract':
                    logging.debug(f"Performing OCR on page {i+1} with Tesseract and languages: {lang}")
                    try:
                        page_text = pytesseract.image_to_string(image, lang=lang)
                        if isinstance(page_text, str) and page_text.strip():
                            text += page_text + "\n"
                            break  # Successful OCR, move to next image
                    except Exception as e:
                        logging.error(f"Error performing OCR with Tesseract on page {i+1}: {e}")
                        continue  # Try next OCR method
                elif ocr_method == 'kraken':
                    logging.debug(f"Performing OCR on page {i+1} with Kraken.")
                    page_text = perform_ocr_with_kraken(image, tools_available=tools_available, verbose=verbose)
                    if isinstance(page_text, str) and page_text.strip():
                        text += page_text + "\n"
                        break  # Successful OCR, move to next image
            if len(text) > 2000:
                break
        if text.strip():
            logging.debug(f"Successfully extracted text through OCR on first {num_pages} pages ({len(text)} characters)")
            if verbose:
                logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
            logging.info(f"Successfully performed OCR on first {num_pages} pages of {pdf_path}.")
    except Exception as e:
        logging.error(f"Error performing OCR on {pdf_path}: {e}")
    return text[:2000]


def extract_text_with_textract(file_path, verbose=False):
    if not essential_libraries['textract']:
        logging.warning(f"Textract is not available. Skipping extraction for {file_path}.")
        return ""
    logging.debug(f"Extracting text with textract: {file_path}")
    try:
        result = textract.process(file_path)
        if result is None:
            logging.error(f"Textract returned None for {file_path}.")
            return ""
        if not isinstance(result, bytes):
            logging.error(f"Unexpected return type from textract for {file_path}: {type(result)}")
            return ""
        text = result.decode('utf-8')
        logging.debug(f"Successfully extracted {len(text)} characters with textract")
        if verbose:
            logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
        logging.info(f"Extracted text from {file_path} using textract.")
        return text[:2000]
    except Exception as e:
        logging.error(f"Error extracting text from {file_path} with textract: {e}")
        return ""

def extract_text_from_epub(epub_path, textract_available, tools_available, verbose=False):
    """
    Extract text from an EPUB file using EbookLib and BeautifulSoup4.
    Falls back to textract if EbookLib or BeautifulSoup4 is unavailable or fails.
    
    Parameters:
        epub_path (str): Path to the EPUB file.
        textract_available (bool): Whether textract is available for fallback.
        tools_available (dict): Dictionary indicating availability of tools.
        verbose (bool): Enable verbose logging.
    
    Returns:
        str: Extracted text up to 2000 characters or empty string if extraction fails.
    """
    if not tools_available.get('ebooklib', False):
        logging.warning(f"EPUB processing is skipped for {epub_path} as ebooklib is not available.")
        if textract_available:
            logging.info("Attempting to extract text using textract as a fallback.")
            return extract_text_with_textract(epub_path, verbose=verbose)
        else:
            logging.error("Textract is also unavailable. Cannot extract text from EPUB.")
            return ""
    
    logging.debug(f"Extracting text from EPUB: {epub_path}")
    text = ""
    try:
        book = epub.read_epub(epub_path)
        for item in book.get_items():
            if item.get_type() == EPUB_ITEM_DOCUMENT:
                if tools_available.get('beautifulsoup4', False):
                    try:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text += soup.get_text(separator="\n")
                    except Exception as e:
                        logging.error(f"Error parsing HTML content in {epub_path}: {e}")
                else:
                    logging.warning("beautifulsoup4 is not available. Skipping HTML parsing for this item.")
                if len(text) > 3000:
                    break
        if text.strip():
            logging.debug(f"Successfully extracted {len(text)} characters from EPUB using ebooklib.")
            if verbose:
                logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
            logging.info(f"Extracted text from {epub_path} using ebooklib.")
            return text[:2000]
        else:
            logging.warning(f"No meaningful text extracted from {epub_path} using ebooklib.")
            if textract_available:
                logging.info("Attempting to extract text using textract as a fallback.")
                return extract_text_with_textract(epub_path, verbose=verbose)
    except Exception as e:
        logging.error(f"Error extracting text from {epub_path} with ebooklib: {e}")
        if textract_available:
            logging.info("Attempting to extract text using textract as a fallback.")
            try:
                text = extract_text_with_textract(epub_path, verbose=verbose)
                if text.strip():
                    logging.debug(f"Successfully extracted text using textract from EPUB ({len(text)} characters)")
                    if verbose:
                        logging.debug(f"Extracted Text (first 500 chars): {text[:500]}")
                    logging.info(f"Extracted text from {epub_path} using textract.")
                    return text[:2000]
            except Exception as inner_e:
                logging.error(f"Error extracting text from {epub_path} using textract: {inner_e}")
    return ""



# Metadata handling functions
def clean_author_name(author_name):
    """Remove titles and punctuations from the author name."""
    author_name = re.sub(r'\bDr\.?\b', '', author_name, flags=re.IGNORECASE)  # Remove titles like Dr, Dr.
    author_name = re.sub(r'\s*,\s*', ' ', author_name).strip()  # Remove extra commas and spaces
    return author_name

def valid_author_name(author_name):
    """Check if the author name is valid, i.e., contains at least a first name and a last name,
    and does not contain special characters other than a-z, A-Z, hyphens, apostrophes, and maybe '.'.
    Also, ensure the name does not contain placeholders like 'lastname' or 'surname'."""
    parts = author_name.strip().split()
    if len(parts) <= 1:
        return False
    if not re.match(r'^[\w\s.\'-]+$', author_name, re.UNICODE):
        return False
    if "lastname" in author_name.lower() or "surname" in author_name.lower():
        return False
    return True

def sort_author_names(author_names, openai_client, max_attempts=5, verbose=False):
    for attempt in range(1, max_attempts + 1):
        if verbose:
            logging.debug(f"Attempt {attempt} to sort author names: {author_names}")
        formatted_author_names = author_names.replace('&', ',')
        prompt = (
            f"You will be given an author name that you must put into the format 'Lastname Surname'. "
            f"So, you must first make an educated guess if the given input is already in this format. If so, return it back. "
            f"If not and it is more plausibly in the format 'Surname(s) Lastname', you must reformat it. "
            f"Example: 'Jan Christian Gertz' must become 'Gertz Jan Christian' and 'Michael M. Meier' must become 'Meier Michael M'. "
            f"No comma after the Lastname! "
            f"If you are given multiple person names, only keep the first and omit all others. "
            f"If it is impossible to come up with a correct name, return <AUTHOR>n a</AUTHOR>. "
            f"You must give the output in the format: <AUTHOR>Lastname Surname(s)</AUTHOR>. "
            f"Here are the name parts: <AUTHOR>{formatted_author_names}</AUTHOR>"
        )
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.5,
                max_tokens=500,
                messages=messages
            )
        except Exception as e:
            logging.error(f"Error querying Ollama server for author names: {e}")
            return "n a"
        reformatted_name = response.choices[0].message.content.strip()
        logging.debug(f"Response from Ollama server: {reformatted_name}")
        name_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_name)
        if name_match:
            ordered_name = name_match.group(1).strip().split(",")[0].strip()
            ordered_name = clean_author_name(ordered_name)
            logging.debug(f"Ordered name after cleaning: '{ordered_name}'")
            if valid_author_name(ordered_name):
                return ordered_name
            else:
                logging.warning(f"Invalid author name format detected: '{ordered_name}', retrying...")
        else:
            logging.warning("Failed to extract a valid name, retrying...")
    logging.error("Maximum retry attempts reached for sorting author names.")
    return "n a"


def parse_metadata(content, verbose=False):
    title_match = re.search(r'<TITLE>(.*?)</TITLE>', content)
    year_match = re.search(r'<YEAR>(.*?)</YEAR>', content)
    author_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', content)
    logging.debug(f"Parsing metadata: {content}")
    if not title_match:
        logging.warning(f"No match for title in {content}.")
        return None
    if not author_match:
        logging.warning(f"No match for author in {content}.")
        return None
    if not year_match:
        logging.info(f"No match for year in {content}. Assigning 'Unknown'.")
        year = "Unknown"
    else:
        year = sanitize_filename(year_match.group(1).strip())
    title = sanitize_filename(title_match.group(1).strip())
    author = author_match.group(1).strip()
    logging.debug(f"Parsed metadata - Title: '{title}', Year: '{year}', Author: '{author}'")
    if any(placeholder in (title.lower(), author.lower(), year.lower()) for placeholder in ["unknown", "n a", ""]):
        logging.error("Error: Found 'unknown', 'n a', or empty strings in title, year, or author.")
        return None
    return {'author': author, 'year': year, 'title': title}

# Communication with Ollama server
def send_to_ollama_server(text, filename, openai_client, max_attempts=4, verbose=False):
    attempt = 1
    while attempt <= max_attempts and not shutdown_flag.is_set():
        logging.debug(f"Consulting Ollama server on file: {filename} (Attempt: {attempt})")
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        prompt = (
            f"Extract the first author name (ignore other authors), year of publication, and title from the following text, considering the filename '{base_filename}' which may contain clues. "
            f"I need the output in the following format: \n"
            f"<TITLE>The publication title</TITLE> \n<YEAR>2023</YEAR> \n<AUTHOR>Lastname Surname</AUTHOR> \n\n"
            f"Here is the extracted text:\n{text}"
        )
        messages = [{"role": "user", "content": prompt}]
        
        if verbose:
            logging.debug(f"Sending prompt to Ollama server: {prompt}")
        
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=250,
                messages=messages,
                timeout=300  # Set a timeout of 300 seconds
            )
        except Exception as e:
            logging.error(f"Error communicating with Ollama server for {filename}: {e}")
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
                continue
            return ""
        if verbose:
            logging.debug(f"Raw content received from server: (start){response.choices[0].message.content}(end)")
        output = response.choices[0].message.content.strip()
        if verbose:
            logging.debug(f"Metadata content received from server: {output}")
        if ("lastname" in output.lower() or "surname" in output.lower()) and attempt < max_attempts:
            logging.warning(f"{filename} - Detected placeholders in response. Retrying...")
            attempt += 1
            time.sleep(2 ** attempt)
            continue
        # Validate response format
        if re.match(r'^<TITLE>.*?</TITLE>\s*<YEAR>\d{4}</YEAR>\s*<AUTHOR>.*?</AUTHOR>$', output, re.DOTALL):
            return output
        else:
            logging.warning(f"Unexpected response format from Ollama server: {output}.")
            #attempt += 1
            #time.sleep(2 ** attempt)
            return output
    logging.error(f"{filename} - Maximum retry attempts reached for sending to Ollama server.")
    return ""


def send_metadata_to_ollama(metadata_info, filename, openai_client, max_attempts=4, verbose=False):
    attempt = 1
    while attempt <= max_attempts:
        logging.debug(f"Sending file metadata to Ollama server for file: {filename} (Attempt: {attempt})")
        metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata_info.items()])
        prompt = (
            f"Unable to extract meaningful text from the file '{os.path.splitext(os.path.basename(filename))[0]}'. "
            f"Here is the file metadata that might contain clues:\n{metadata_str}\n\n"
            f"Based on this metadata, extract the first author name (ignore other authors), year of publication, and title of the publication. "
            f"Provide the output in the following format:\n"
            f"<TITLE>The publication title</TITLE>\n"
            f"<YEAR>2023</YEAR>\n"
            f"<AUTHOR>Lastname Surname</AUTHOR>"
        )
        messages = [{"role": "user", "content": prompt}]
        logging.debug(f"Sending prompt to Ollama server: {prompt}")
        
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=250,
                messages=messages
            )
        except Exception as e:
            logging.error(f"Error sending metadata to Ollama server for {filename}: {e}")
            return ""
        output = response.choices[0].message.content.strip()
        logging.debug(f"Raw content received from server: (start){output}(end)")
        if ("lastname" in output.lower() or "surname" in output.lower()) and attempt < max_attempts:
            logging.warning("Detected placeholders in response. Retrying...")
            attempt += 1
            continue
        return output
    logging.error("Maximum retry attempts reached for sending metadata to Ollama server.")
    return ""


# Filename sanitization to ensure safe filesystem operations
def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    if " " in name:
        name = re.sub(r'(\w)\.', r'\1', name)
        parts = name.split()
        new_parts = []
        current_initials = ""
        for part in parts:
            if len(part) == 1:
                current_initials += part
            else:
                if current_initials:
                    new_parts.append(current_initials)
                    current_initials = ""
                new_parts.append(part)
        if current_initials:
            new_parts.append(current_initials)
        name = " ".join(new_parts)
    return name.strip().replace('/', '')


# Execute the generated rename commands script
def execute_rename_commands(script_path):
    try:
        # Ensure the file is closed before executing
        with open(script_path, 'r') as script_file:
            pass  # Just to ensure it's accessible and can be opened
        subprocess.run(['bash', script_path], check=True)
        logging.info(f"Successfully executed rename commands from {script_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing rename commands: {e}. Please check the rename script for issues.")
    except FileNotFoundError:
        logging.error(f"Rename script '{script_path}' not found.")
    except PermissionError:
        logging.error(f"Permission denied while executing the rename script '{script_path}'. Ensure it is executable.")
    except Exception as e:
        logging.error(f"Unexpected error during rename command execution: {e}")


# Main processing function for each file
def process_file(file_path, openai_client, max_attempts=5, verbose=False, tools_available=None, optional_libraries=None):
    """
    Process a single file to extract metadata and prepare rename commands.
    
    Parameters:
        file_path (str): Path to the file to process.
        openai_client: Initialized OpenAI client for communication.
        max_attempts (int): Maximum number of processing attempts.
        verbose (bool): Enable verbose logging.
        tools_available (dict): Dictionary indicating availability of tools.
        optional_libraries (dict): Dictionary indicating availability of optional libraries.
    
    Returns:
        dict or None: Extracted metadata dictionary or None if processing fails.
    """
    if tools_available is None:
        tools_available = {}
    if optional_libraries is None:
        optional_libraries = {}

    logging.debug(f"Processing file: {file_path}")
    if shutdown_flag.is_set():
        logging.info(f"Shutdown initiated. Skipping processing for {file_path}.")
        return None
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        ocr_performed = False
        try:
            logging.info(f"Starting processing for: {file_path} (Attempt {attempt})")
            
            for extraction_attempt in range(1, max_attempts + 1):
                if shutdown_flag.is_set():
                    logging.info(f"Shutdown initiated. Stopping extraction for {file_path}.")
                    return None
  
                logging.debug(f"Processing file: {file_path} (Attempt: {extraction_attempt})")
                if file_path.lower().endswith('.pdf'):
                    # Perform OCR after 2 failed attempts instead of 4
                    perform_ocr_if_needed = extraction_attempt >= 2 and not ocr_performed
                    text = extract_text_from_pdf(
                        file_path, 
                        perform_ocr_if_needed=perform_ocr_if_needed, 
                        ocr_available={'tesseract': tools_available.get('tesseract', False),
                                       'kraken': tools_available.get('kraken', False)}, 
                        verbose=verbose,
                        tools_available=tools_available
                    )
                    if perform_ocr_if_needed and text:
                        ocr_performed = True
                elif file_path.lower().endswith('.epub'):
                    text = extract_text_from_epub(
                        file_path, 
                        textract_available=tools_available.get('textract', False), 
                        tools_available=tools_available,
                        verbose=verbose
                    )
                elif file_path.lower().endswith('.docx'):
                    text = extract_text_from_docx(
                        file_path, 
                        verbose=verbose,
                        tools_available=tools_available
                    )
                elif file_path.lower().endswith(('.azw', '.azw3')):
                    text = extract_text_from_azw(
                        file_path, 
                        verbose=verbose,
                        tools_available=tools_available
                    )
                elif file_path.lower().endswith('.djvu'):
                    text = extract_text_from_djvu(
                        file_path, 
                        djvutxt_installed=tools_available.get('djvutxt', False), 
                        verbose=verbose,
                        tools_available=tools_available
                    )
                elif file_path.lower().endswith('.mobi'):
                    text = extract_text_from_mobi(
                        file_path, 
                        verbose=verbose, 
                        tools_available=tools_available
                    )
                elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # Check if any OCR tool is available
                    ocr_tools_available = tools_available.get('tesseract', False) or tools_available.get('kraken', False)
                    if not (tools_available.get('pillow', False) or ocr_tools_available):
                        logging.warning(f"No OCR tools available to process image file {file_path}. Skipping.")
                        text = ""
                    else:
                        text = extract_text_from_image(
                            file_path,
                            perform_ocr_if_needed=extraction_attempt >= 1,
                            ocr_available={'tesseract': tools_available.get('tesseract', False),
                                           'kraken': tools_available.get('kraken', False)},
                            verbose=verbose,
                            tools_available=tools_available,
                            optional_libraries=optional_libraries
                        )
                elif file_path.lower().endswith(('.doc', '.xls', '.xlsx', '.ppt', '.pptx', 
                                                '.odt', '.ods', '.odp', '.html', '.xml', 
                                                '.rtf', '.md', '.txt')):
                    text = extract_text_with_textract(file_path, verbose=verbose)
                else:
                    logging.warning(f"Unsupported file format for {file_path}. Skipping.")
                    return None
                
                logging.debug(f"Extracted text from {file_path}: {text[:500]}...")
                if is_text_meaningful(text, file_path=file_path):
                    logging.debug("Text extraction successful and meaningful.")
                    metadata_content = send_to_ollama_server(text, file_path, openai_client, verbose=verbose)
                    break
                else:
                    logging.warning(f"Attempt {extraction_attempt}: Extracted text is not meaningful for {file_path}.")
                    word_pattern = r'\b\w+\b'
                    unique_words = len(set(re.findall(word_pattern, text.lower())))
                    logging.debug(f"Text length: {len(text)}, Unique words: {unique_words}")
                    
                    if perform_ocr_if_needed and (tools_available.get('tesseract', False) or tools_available.get('kraken', False)):
                        logging.info("Invoking OCR due to non-meaningful text extraction.")
                        if file_path.lower().endswith('.pdf'):
                            text = perform_ocr_on_first_pages(
                                file_path, 
                                num_pages=3, 
                                verbose=verbose, 
                                tools_available=tools_available
                            )
                            ocr_performed = True
                        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                            text = extract_text_from_image(
                                file_path,
                                perform_ocr_if_needed=True,
                                ocr_available={'tesseract': tools_available.get('tesseract', False),
                                               'kraken': tools_available.get('kraken', False)},
                                tools_available=tools_available,
                                optional_libraries=optional_libraries,
                                verbose=verbose
                            )
                        if is_text_meaningful(text, file_path=file_path):
                            logging.debug("OCR extraction successful and meaningful.")
                            metadata_content = send_to_ollama_server(text, file_path, openai_client, verbose=verbose)
                            break
                        else:
                            logging.warning("OCR extraction did not yield meaningful text.")
                    elif extraction_attempt == max_attempts:
                        logging.info(f"All extraction attempts failed for {file_path}. Proceeding to extract metadata from file metadata.")
                        metadata_info = extract_file_metadata(file_path, verbose=verbose)
                        metadata_content = send_metadata_to_ollama(metadata_info, file_path, openai_client, verbose=verbose)
            
            if metadata_content:
                metadata = parse_metadata(metadata_content, verbose=verbose)
                if metadata:
                    corrected_authors = sort_author_names(metadata['author'], openai_client, verbose=verbose)
                    logging.debug(f"Corrected author: '{corrected_authors}'")
                    metadata['author'] = corrected_authors
                    if valid_author_name(metadata['author']):
                        return metadata
                    else:
                        logging.warning(f"Author's name invalid after sorting: '{metadata['author']}'. Retrying...")
                else:
                    logging.warning(f"Metadata parsing failed for {file_path}. Retrying...")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}. Retrying...")
        logging.info(f"Retrying processing for {file_path} (Attempt {attempt}/{max_attempts})...")
    logging.error(f"Failed to process {file_path} after {max_attempts} attempts.")
    with file_lock:
        try:
            with open("unparseables.lst", "a") as unparseable_file:
                unparseable_file.write(f"{file_path}\n")
                unparseable_file.flush()
        except Exception as e:
            logging.error(f"Failed to write to unparseables.lst: {e}")
    return None


# Escape special characters in filenames for shell commands
def escape_special_chars(filename):
    """
    Safely escape special characters in filenames for shell commands.
    
    Tries to use shlex.quote for comprehensive escaping.
    Falls back to regex-based escaping if shlex is unavailable.
    
    Parameters:
        filename (str): The filename to escape.
    
    Returns:
        str: Escaped filename.
    """
    try:
        import shlex
        return shlex.quote(filename)
    except ImportError:
        logging.warning("shlex module not available. Falling back to regex-based escaping.")
        return re.sub(r'([$`"\\])', r'\\\1', filename)

# Main function to process all files in the directory
def main(directory, verbose, force, singletask, debug, NO_INSTALL):
    global optional_libraries
    rename_script_path = "rename_commands.sh"
    
    # Initialize external_tools dictionary with all relevant tools
    external_tools = {
        'tesseract': False,
        'kraken': False, 
        'djvutxt': False,
        'textract': essential_libraries.get('textract', False)
    }

    # Check for Kraken availability
    if shutil.which('kraken'):
        external_tools['kraken'] = True
        logging.debug("Kraken OCR is available.")
    else:
        logging.warning("Kraken OCR is not installed or not found in PATH. Kraken OCR functionality will be skipped.")
        # Prompt user to install Kraken
        install_commands = ['pip', 'install', 'kraken']
        if prompt_install_tool('Kraken OCR', install_commands, auto_install=True, no_install=NO_INSTALL):
                external_tools['kraken'] = True
                if not shutil.which('kraken'):
                    logging.error("Kraken OCR installation failed or 'kraken' is not in PATH.")
                    external_tools['kraken'] = False

    # Check for Tesseract availability
    if shutil.which('tesseract'):
        external_tools['tesseract'] = True
        logging.debug("Tesseract OCR is available.")
    else:
        logging.warning("Tesseract OCR is not installed or not found in PATH. Tesseract OCR functionality will be skipped.")
        # Prompt user to install Tesseract
        current_os = platform.system()
        install_commands = []
        if current_os == "Darwin":
            install_commands = ['brew', 'install', 'tesseract']
        elif current_os == "Linux":
            if shutil.which('apt-get'):
                install_commands = ['sudo', 'apt-get', 'install', '-y', 'tesseract-ocr']
            elif shutil.which('yum'):
                install_commands = ['sudo', 'yum', 'install', '-y', 'tesseract']
        
        if install_commands:
            if prompt_install_tool('Tesseract OCR', install_commands, auto_install=True, no_install=NO_INSTALL):
                external_tools['tesseract'] = True
                if not shutil.which('tesseract'):
                    logging.error("Tesseract OCR installation failed or 'tesseract' is not in PATH.")
                    external_tools['tesseract'] = False
        else:
            logging.error("Unsupported OS or package manager. Please install Tesseract OCR manually.")

    # Check for djvutxt availability
    external_tools['djvutxt'] = shutil.which('djvutxt') is not None
    if not external_tools['djvutxt']:
        logging.warning("djvutxt is not installed or not found in PATH. DJVU files will be skipped.")
    else:
        logging.debug("djvutxt is available.")

    if not os.path.exists(directory):
        logging.error("The specified directory does not exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not files:
        logging.info("No supported files found in the specified directory.")
        sys.exit(0)
    
    try:
        with file_lock:
            with open(rename_script_path, "w") as mv_file:
                mv_file.write("#!/bin/bash\n")
                mv_file.write('set -e\n')
                mv_file.flush()
    
        # Dynamically find tessdata directories and set TESSDATA_PREFIX
        tessdata_dirs = find_tessdata_dirs()

        if tessdata_dirs:
            TESSDATA_PREFIX = tessdata_dirs[0]  # Use the first found tessdata directory
            os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX
            logging.debug(f"Set TESSDATA_PREFIX to: {TESSDATA_PREFIX}")
        else:
            logging.warning("Could not find 'tessdata' directory. Please set the TESSDATA_PREFIX environment variable manually.")
            TESSDATA_PREFIX = None  # Handle cases where tessdata is not found

        tools_available = MappingProxyType({
            'tesseract': external_tools.get('tesseract', False),
            'kraken': external_tools.get('kraken', False),
            'djvutxt': external_tools.get('djvutxt', False),
            'textract': essential_libraries.get('textract', False),
            'pdfminer': optional_libraries.get('pdfminer', False),
            'PyPDF2': optional_libraries.get('PyPDF2', False),
            'pillow': optional_libraries.get('pillow', False),
            'ebooklib': optional_libraries.get('ebooklib', False),
            'beautifulsoup4': optional_libraries.get('beautifulsoup4', False),
            'docx': optional_libraries.get('docx', False),
            'mobi': optional_libraries.get('mobi', False),
            'pytesseract': optional_libraries.get('pytesseract', False),
            'pdf2image': optional_libraries.get('pdf2image', False)
        })

        # Log tool availability
        tool_messages = {
            'tesseract': "Tesseract OCR is available.",
            'kraken': "Kraken OCR is available.",
            'djvutxt': "djvutxt is available.",
            'textract': "textract is available.",
            'pdfminer': "pdfminer is available.",
            'PyPDF2': "PyPDF2 is available.",
            'pillow': "Pillow (PIL) is available.",
            'ebooklib': "ebooklib is available.",
            'beautifulsoup4': "beautifulsoup4 is available.",
            'docx': "python-docx is available.",
            'mobi': "mobi library is available.",
            'pytesseract': "pytesseract is available.",
            'pdf2image': "pdf2image is available."
        }

        for tool, available in tools_available.items():
            if available:
                logging.debug(tool_messages.get(tool, f"{tool} is available."))
            else:
                logging.warning(tool_messages.get(tool, f"{tool} is not available."))

        # Check and install missing Tesseract languages
        if tools_available['tesseract']:
            check_and_install_tesseract_langs(REQUIRED_TESS_LANGS)

        openai_client = get_openai_client()

        if singletask:
            max_workers = 1
            logging.info("Single-task mode enabled. Processing files sequentially.")
        else:
            max_workers = min(32, os.cpu_count() + 4)
            logging.info(f"Processing files with {max_workers} concurrent workers.")
        
        # Initialize ProgressBar
        progress_bar = ProgressBar(total=len(files), desc="Processing Files", enable=optional_libraries.get('tqdm', False))

        if singletask:
            # Sequential processing
            for filename in files:
                if shutdown_flag.is_set():
                    logging.info("Shutdown flag detected. Stopping further processing.")
                    break
                file_path = os.path.join(directory, filename)
                try:
                    metadata = process_file(file_path, openai_client, max_attempts=MAX_RETRY_ATTEMPTS, verbose=verbose, tools_available=tools_available, optional_libraries=optional_libraries)
                    if metadata:
                        author = metadata['author']
                        title = metadata['title']
                        year = metadata['year']
                        if not author or not title:
                            logging.warning(f"Missing author or title for {file_path}. Skipping renaming.")
                            with file_lock:
                                with open("unparseables.lst", "a") as unparseable_file:
                                    unparseable_file.write(f"{file_path}\n")
                                    unparseable_file.flush()
                            continue
                        first_author = sanitize_filename(author.split(", ")[0])
                        target_dir = os.path.join(directory, first_author)
                        if year == "Unknown":
                            logging.info(f"Year is unknown for {file_path}. Assigning 'UnknownYear' in filename.")
                            year = "UnknownYear"
                        else:
                            year = sanitize_filename(year)
                        sanitized_title = sanitize_filename(title)
                        file_extension = filename.split('.')[-1]
                        new_filename = f"{year} {sanitized_title}.{file_extension}"
                        new_file_path = os.path.join(target_dir, new_filename)
                        escaped_file_path = escape_special_chars(file_path)
                        escaped_target_dir = escape_special_chars(target_dir)
                        escaped_new_file_path = escape_special_chars(new_file_path)
                        with file_lock:
                            with open(rename_script_path, "a") as mv_file:
                                mv_file.write(f"mkdir -p {escaped_target_dir}\n")
                                mv_file.write(f"mv {escaped_file_path} {escaped_new_file_path}\n")
                                mv_file.flush()
                        logging.debug(f"Rename command added for: {file_path}")
                    else:
                        with file_lock:
                            with open("unparseables.lst", "a") as unparseable_file:
                                unparseable_file.write(f"{file_path}\n")
                                unparseable_file.flush()
                        logging.info(f"Added to unparseables: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {e}")
                    with file_lock:
                        with open("unparseables.lst", "a") as unparseable_file:
                            unparseable_file.write(f"{file_path}\n")
                            unparseable_file.flush()
                finally:
                    progress_bar.update(1)
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        process_file, 
                        os.path.join(directory, filename), 
                        openai_client,  # Pass the OpenAI client
                        max_attempts=MAX_RETRY_ATTEMPTS, 
                        verbose=verbose, 
                        tools_available=tools_available,
                        optional_libraries=optional_libraries
                    ): filename
                    for filename in files
                }
                try:
                    for future in as_completed(future_to_file):
                        if shutdown_flag.is_set():
                            logging.info("Shutdown flag detected. Breaking out of the processing loop.")
                            for fut in future_to_file:
                                fut.cancel()
                            break
                        filename = future_to_file[future]
                        file_path = os.path.join(directory, filename)
                        try:
                            metadata = future.result()
                            if metadata:
                                author = metadata['author']
                                title = metadata['title']
                                year = metadata['year']
                                if not author or not title:
                                    logging.warning(f"Missing author or title for {file_path}. Skipping renaming.")
                                    with file_lock:
                                        with open("unparseables.lst", "a") as unparseable_file:
                                            unparseable_file.write(f"{file_path}\n")
                                            unparseable_file.flush()
                                    continue
                                first_author = sanitize_filename(author.split(", ")[0])
                                target_dir = os.path.join(directory, first_author)
                                if year == "Unknown":
                                    logging.info(f"Year is unknown for {file_path}. Assigning 'UnknownYear' in filename.")
                                    year = "UnknownYear"
                                else:
                                    year = sanitize_filename(year)
                                sanitized_title = sanitize_filename(title)
                                file_extension = filename.split('.')[-1]
                                new_filename = f"{year} {sanitized_title}.{file_extension}"
                                new_file_path = os.path.join(target_dir, new_filename)
                                escaped_file_path = escape_special_chars(file_path)
                                escaped_target_dir = escape_special_chars(target_dir)
                                escaped_new_file_path = escape_special_chars(new_file_path)
                                with file_lock:
                                    with open(rename_script_path, "a") as mv_file:
                                        mv_file.write(f"mkdir -p {escaped_target_dir}\n")
                                        mv_file.write(f"mv {escaped_file_path} {escaped_new_file_path}\n")
                                        mv_file.flush()
                                logging.debug(f"Rename command added for: {file_path}")
                            else:
                                with file_lock:
                                    with open("unparseables.lst", "a") as unparseable_file:
                                        unparseable_file.write(f"{file_path}\n")
                                        unparseable_file.flush()
                                logging.info(f"Added to unparseables: {file_path}")
                        except Exception as e:
                            logging.error(f"Failed to process {file_path}: {e}")
                            with file_lock:
                                with open("unparseables.lst", "a") as unparseable_file:
                                    unparseable_file.write(f"{file_path}\n")
                                    unparseable_file.flush()
                        finally:
                            progress_bar.update(1)
                except Exception as e:
                    logging.error(f"Error during concurrent processing: {e}")
                finally:
                    if shutdown_flag.is_set():
                        logging.info("Cancelling pending tasks...")
                        for future in future_to_file:
                            future.cancel()
    
        progress_bar.close()

        with file_lock:
            try:
                os.chmod(rename_script_path, 0o755)
                logging.debug(f"Set execute permissions for {rename_script_path}.")
            except Exception as e:
                logging.error(f"Error setting execute permissions for {rename_script_path}: {e}")
        logging.info(f"\nRename commands have been written to {rename_script_path}")
        logging.info("Please review the commands before executing them.")
        
        if force:
            logging.info("Force option detected. Executing rename commands...")
            execute_rename_commands(rename_script_path)
        else:
            logging.info(f"To execute the rename commands, run: bash {rename_script_path}")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Initiating graceful shutdown...")
        shutdown_flag.set()
        progress_bar.close()
    except Exception as e:
        logging.critical(f"Unhandled exception in main workflow: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if progress_bar and progress_bar.enable:
            progress_bar.close()
        logging.info("Script terminated.")


# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse various document formats to extract metadata and generate rename commands."
    )
    parser.add_argument(
        "directory", 
        nargs='?', 
        default=".", 
        help="Directory containing document files (default: current directory)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output (detailed logs in logfile)"
    )

    parser.add_argument(
        "--no-install", 
        action="store_true", 
        help="Do not prompt for installing missing tools. Proceed without them."
    )

    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode: detailed logs on stdout only, no logfile"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Automatically execute rename commands"
    )
    parser.add_argument(
        "--singletask", 
        action="store_true", 
        help="Disable concurrent processing and process files sequentially"
    )
    args = parser.parse_args()
    DIRECTORY = args.directory
    VERBOSE = args.verbose
    DEBUG = args.debug
    FORCE = args.force
    SINGLETASK = args.singletask
    NO_INSTALL = args.no_install
    setup_logging(VERBOSE, DEBUG)

    main(DIRECTORY, VERBOSE, FORCE, SINGLETASK, DEBUG, NO_INSTALL)
