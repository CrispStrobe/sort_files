import os
import sys
import re
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI
from tqdm import tqdm
import argparse
import threading
import ctypes
from ebooklib import epub
from bs4 import BeautifulSoup
import docx
import mobi
import shutil
import textract

modelname = "cas/spaetzle-v85-7b"
#modelname = "cas/llama3.1-8b-spaetzle-v109"

verbose = False

def terminate_thread(thread):
    """Terminate a thread from another thread."""
    if not thread.is_alive():
        return

    tid = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")

# we could also use textract for docx, but sometimes this might work better
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
            if len(text) > 3000:
                break
    except Exception as e:
        print(f"Error extracting text from {docx_path}: {e}")
    return text[:2000]

def extract_text_from_mobi(mobi_path):
    text = ""
    try:
        tempdir, filepath = mobi.extract(mobi_path)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        shutil.rmtree(tempdir)
    except Exception as e:
        print(f"Error extracting text from {mobi_path} using mobi library: {e}")
        print("Attempting to use EbookLib for MOBI extraction...")
        try:
            # Fallback to using EbookLib
            book = epub.read_epub(mobi_path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
                if len(text) > 3000:
                    break
        except Exception as inner_e:
            print(f"Error extracting text from {mobi_path} using EbookLib: {inner_e}")
    return text[:2000]

def extract_text_from_pdf(pdf_path, timeout=5):
    text = ""
    def target():
        nonlocal text
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num in range(min(5, len(reader.pages))):
                    if verbose:
                        print("reading page: ", page_num, "\n")
                    page = reader.pages[page_num]
                    extracted_text = page.extract_text() or ''
                    if extracted_text:
                        text += extracted_text
                    if len(text) > 3000:
                        break
        except (PdfReadError, ValueError, TypeError) as e:
            print(f"Error extracting text from {pdf_path} with PyPDF2: {e}.")
            text = ""

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"Timeout extracting text from {pdf_path} with PyPDF2. Attempting OCR.")
        terminate_thread(thread)  # Ensure thread is terminated properly
        text = ""
        print("\ncleaned.\n")

    return text[:2000]

def extract_text_with_textract(file_path):
    try:
        text = textract.process(file_path).decode('utf-8')
        return text[:2000]
    except Exception as e:
        print(f"Error extracting text from {file_path} with textract: {e}")
        return ""

def extract_text_from_epub(epub_path):
    text = ""
    try:
        book = epub.read_epub(epub_path)
        for item in book.get_items():
            if item.media_type == 'application/xhtml+xml':
                soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                text += soup.get_text(separator="\n")
                if len(text) > 3000:
                    break
    except Exception as e:
        print(f"Error extracting text from {epub_path} with EbookLib: {e}")
    return text[:2000]

def perform_ocr(pdf_path):
    if verbose:
       print("performing OCR...\n")
    text = ""
    try:
        pages = convert_from_path(pdf_path, 300, first_page=1, last_page=5)
        for page in pages:
            text += pytesseract.image_to_string(page)
            if len(text) > 2000:
                break
    except Exception as e:
        print(f"Error performing OCR on {pdf_path}: {e}")
    return text[:2000]

def clean_author_name(author_name):
    """Remove titles and punctuations from the author name."""
    author_name = re.sub(r'\bDr\.?\b', '', author_name)  # Remove titles like Dr, Dr.
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

def sort_author_names(author_names, attempt=1, verbose=False):
    if attempt > 5:
        if verbose:
            print("Maximum retry attempts reached for sorting names.")
        return "n a"

    author_names = author_names.replace('&', ',')

    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )

    if verbose:
       print(f"Querying the server to correct author name: {author_names}.\n")

    prompt = (f"You will be given an author name that you must put into the format 'Lastname Surname'."
              f"So, you must first make an educated guess if the given input is already in this format. If so, return it back."
              f"If not and it is more plausibly in the format 'Surname(s) Lastname', you must reformat it."
              f"Example: 'Jan Christian Gertz' must become 'Gertz Jan Christian' and 'Michael M. Meier' must become 'Meier Michael M'."
              f"No comma after the Lastname!"
              f"If you are given multiple person names, only keep the first and omit all others."
              f"If it is impossible to come up with a correct name, return <AUTHOR>n a</AUTHOR>."
              f"You must give the output in the format: <AUTHOR>Lastname Surname(s)</AUTHOR>."
              f"Here are the name parts: <AUTHOR>{author_names}</AUTHOR>")

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=modelname,
        temperature=0.5,
        max_tokens=500,
        messages=messages
    )

    reformatted_name = response.choices[0].message.content.strip()
    name = re.search(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_name)

    if name:
        ordered_name = name.group(1).strip()
        # Split to take only the first author if multiple are present
        ordered_name = ordered_name.split(",")[0].strip()
        if verbose:
            print(f"Ordered name from server: '{ordered_name}'")
        ordered_name = clean_author_name(ordered_name)  # Clean author name after receiving from server
        if verbose:
            print(f"Ordered name after cleaning: '{ordered_name}'")
        if not valid_author_name(ordered_name) or ordered_name.lower() == "n a":
            if verbose:
                print(f"Invalid author name format detected: '{ordered_name}', retrying...")
            return sort_author_names(author_names, attempt + 1, verbose=verbose)
        return ordered_name
    else:
        if verbose:
            print("Failed to extract a valid name, retrying...")
        return sort_author_names(author_names, attempt + 1, verbose=verbose)

def send_to_ollama_server(text, filename, attempt=1, verbose=False):
    if verbose:
       print("consulting ollama server on file:", filename, "in attempt:", attempt, "\n")

    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    prompt = (f"Extract the first author name (ignore other authors), year of publication, and title from the following text, considering the filename '{base_filename}' which may contain clues. "
              "I need the output in the following format: \n"
              "<TITLE>The publication title</TITLE> \n<YEAR>2023</YEAR> \n <AUTHOR>Lastname Surname</AUTHOR> \n\n"
              "Here is the extracted text:\n" + text)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model = modelname,
        temperature=0.7,
        max_tokens=250,
        messages=messages
    )

    output = response.choices[0].message.content.strip()
    if verbose:
        print(f"Metadata content received from server: {output}")

    if ("lastname" in output.lower() or "surname" in output.lower()) and attempt < 4:
        return send_to_ollama_server(text, filename, attempt + 1, verbose=verbose)
    return output

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).replace('/', '')

def parse_metadata(content):
    global verbose
    title_match = re.search(r'<TITLE>(.*?)</TITLE>', content)
    year_match = re.search(r'<YEAR>(.*?)</YEAR>', content)
    author_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', content)

    if verbose:
       print("parsing metadata:", content, "\n")

    if not title_match:
        if verbose:
            print(f"\nNo match for title in {content}.\n")
        return None

    if not author_match:
        if verbose:
            print(f"\nNo match for author in {content}.\n")
        return None

    if not year_match:
        print(f"\nNo match for year in {content}. Continuing without year.\n")

    title = sanitize_filename(title_match.group(1).strip())
    year = sanitize_filename(year_match.group(1).strip() if year_match else "")
    author = author_match.group(1).strip()

    if verbose:
        print(f"Parsed metadata - Title: '{title}', Year: '{year}', Author: '{author}'")

    if any(placeholder in (title.lower(), author.lower()) for placeholder in ["unknown", "n a", ""]) or year.lower() == "unknown" or year == "n a":
        if verbose:
            print("Error: found 'unknown' or 'n a' or '' in title, year or author!\n")
        return None

    return {'author': author, 'year': year, 'title': title}

def process_file(file_path, attempt=1):
    if verbose:
        print("processing file:", file_path, " in attempt:", attempt, "\n")

    if attempt > 4:
        if verbose:
            print(f"Maximum retry attempts reached for {file_path}.")
        return None

    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.epub'):
            text = extract_text_from_epub(file_path)
        elif file_path.lower().endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_path.lower().endswith(('.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                                         '.odt', '.ods', '.odp', '.jpg', '.jpeg', '.png', 
                                         '.gif', '.html', '.xml', '.rtf', '.md', '.txt')):
            text = extract_text_with_textract(file_path)
        elif file_path.lower().endswith('.mobi'):
            text = extract_text_from_mobi(file_path)
        else:
            if verbose:
                print(f"Unsupported file format for {file_path}.")
            return None

        if verbose:
           print("returned from extraction.\n")
    except Exception as e:
        if verbose:
            print(f"Error extracting text from {file_path}: {e}. Attempting OCR next.")
        text = ""

    if verbose:
       print("extracted text: ", text, ".\n")

    if not text.strip():
        if file_path.lower().endswith('.pdf'):
            try:
                if verbose:
                    print("attempting ocr next...\n")
                text = perform_ocr(file_path)
            except Exception as e:
                if verbose:
                    print(f"Error extracting text from {file_path} with OCR: {e}")
                return None

    try:
        metadata_content = send_to_ollama_server(text, file_path, attempt)
    except Exception as e:
        if verbose:
            print(f"Error sending text to server for {file_path}: {e}")
        return None

    metadata = parse_metadata(metadata_content)
    if metadata:
        corrected_authors = sort_author_names(metadata['author'])
        if verbose:
            print(f"Corrected author: '{corrected_authors}'")
        metadata['author'] = corrected_authors
        if not valid_author_name(metadata['author']):
            if verbose:
                print(f"Author's name still invalid after sorting, retrying for {file_path}. Attempt {attempt}")
            return process_file(file_path, attempt + 1)
        return metadata
    else:
        if verbose:
            print(f"Error: Metadata parsing failed or incomplete for {file_path}. \nMetadata content: {metadata_content}, retrying... Attempt {attempt}")
        return process_file(file_path, attempt + 1)

def main(directory, verbose):
    if not os.path.exists(directory):
        print("The specified directory does not exist")
        sys.exit(1)

    supported_extensions = ('.pdf', '.epub', '.docx', '.doc', '.xls', '.xlsx', '.ppt', '.pptx', 
                            '.odt', '.ods', '.odp', '.jpg', '.jpeg', '.png', '.gif', '.html', 
                            '.xml', '.rtf', '.md', '.txt', '.mobi')
    
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported_extensions)]

    with open("rename_commands.sh", "w") as mv_file, open("unparseables.lst", "w") as unparseable_file:
        mv_file.write("#!/bin/sh\n")
        mv_file.flush()
        for filename in tqdm(files):
            file_path = os.path.join(directory, filename)
            try:
                metadata = process_file(file_path)
                if metadata:
                    first_author = sanitize_filename(metadata['author'].split(", ")[0])
                    target_dir = os.path.join(directory, first_author)
                    new_file_path = os.path.join(target_dir, f"{metadata['year']} {sanitize_filename(metadata['title'])}.{filename.split('.')[-1]}")

                    mv_file.write(f"mkdir -p \"{target_dir}\"\n")
                    mv_file.write(f"mv \"{file_path}\" \"{new_file_path}\"\n")
                    mv_file.flush()
                else:
                    unparseable_file.write(f"{file_path}\n")
                    unparseable_file.flush()
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                unparseable_file.write(f"{file_path}\n")
                unparseable_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDFs and EPUBs and extract metadata.")
    parser.add_argument("directory", help="Directory containing PDF and EPUB files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    DIRECTORY = args.directory
    verbose = args.verbose

    main(DIRECTORY, verbose)
