import os
import sys
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
#from PIL import Image
from openai import OpenAI
from tqdm import tqdm

modelname="cas/spaetzle-v58:latest"
#model="cas/occiglot-7b-de-en-instruct-q4-k-m",
#modelname = "mistral:7b-instruct-q4_K_M"

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page_num in range(min(5, len(reader.pages))):
            page = reader.pages[page_num]
            extracted_text = page.extract_text() or ''
            if extracted_text:
                text += extracted_text
            if len(text) > 2000:
                break
    return text[:2000]

def perform_ocr(pdf_path):
    text = ""
    pages = convert_from_path(pdf_path, 300, first_page=1, last_page=5)
    for page in pages:
        text += pytesseract.image_to_string(page)
        if len(text) > 2000:
            break
    return text[:2000]

def sort_author_names(author_names):
    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )
    # Building a proper prompt for the OpenAI API
    prompt = ("Please reorder the following author names into the format 'Lastname Surname', if they are not already. "
              "Each name should be in the format <AUTHOR>Lastname Surname</AUTHOR>. Here are the names: " +
              ' '.join(f"<AUTHOR>{name}</AUTHOR>" for name in author_names.split(", ")))
    
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=modelname,
        temperature=0.5,
        max_tokens=500,  # Adjusted to allow for larger responses
        messages=messages
    )
    
    # Extracting and reformating the response to only include names
    reformatted_names = response.choices[0].message.content.strip()
    
    # Parsing the output to ensure it is in the correct format
    # This regex extracts text within <AUTHOR> tags, assuming the API follows the instructions properly
    names = re.findall(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_names)
    
    # Joining names with a comma if multiple authors are present
    ordered_names = ", ".join(names)
    #print ("\nordered names:", ordered_names)
    return ordered_names


def send_to_ollama_server(text, attempt=1):
    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )
    prompt = ("Extract the author name, year of publication, and title from the following text which is an "
              "OCR extract from an academic publication. I need the output in the following format: "
              "<TITLE>The publication title</TITLE>, <YEAR>2023</YEAR>, <AUTHOR>Lastname Surname</AUTHOR>. "
              "You must make very sure you have the correct order of Lastname Surname. If your result looks more like Surname Lastname, reorder the names!"
              "If there are several authors, specify the first two (omitting others, if any) like this: <AUTHOR>Lastname Surname, Lastname Surname</AUTHOR>"
              "Now here is the extracted text:\n" + text)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model = modelname,
        temperature=0.7,
        max_tokens=250,
        messages=messages
    )
    #print ("\nresponse: ", response)
    output = response.choices[0].message.content.strip()
    #response.choices[0].message.content.strip()
    #print ("\noutput: ", output)

    if "Lastname Surname" in output and attempt < 3:
        return send_to_ollama_server(text, attempt + 1)
    return output

def parse_metadata(content):
    # Search for the first occurrence of title, year, and author
    title_match = re.search(r'<TITLE>(.*?)</TITLE>', content)
    year_match = re.search(r'<YEAR>(.*?)</YEAR>', content)
    author_match = re.search(r'<AUTHOR>(.*?)</AUTHOR>', content)

    # Check all matches were successful and that placeholders are not part of the actual data
    if not title_match or not year_match or not author_match:
        return None  # If any field is missing, return None

    title = title_match.group(1)
    year = year_match.group(1)
    author = author_match.group(1)

    if "lastname" in author.lower() or "surname" in author.lower():
        return None  # Check for placeholder text in author names

    # Return the first found metadata
    return {'author': author, 'year': year, 'title': title}


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        text = perform_ocr(pdf_path)

    metadata_content = send_to_ollama_server(text)
    metadata = parse_metadata(metadata_content)
    if metadata and 'author' in metadata:
            corrected_authors = sort_author_names(metadata['author'])
            metadata['author'] = corrected_authors
            return metadata
    else:
        print(f"Error: Metadata parsing failed or incomplete for {pdf_path}. Metadata content: {metadata_content}")
        return None

def main(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            try:
                metadata = process_pdf(pdf_path)
                if metadata:
                    sanitized_name = sanitize_filename(f"{metadata['author']} {metadata['year']} {metadata['title']}")
                    #print ("\nnew name: ", sanitized_name)
                    with open("rename_commands.sh", "a") as mv_file:
                        mv_file.write(f"mv \"{pdf_path}\" \"{sanitized_name}.pdf\"\n")
                else:
                    with open("unparseables.lst", "a") as unparseable_file:
                        unparseable_file.write(f"{pdf_path}\n")
            except Exception as e:
                print(f"Failed to process {pdf_path}: {e}")
                with open("unparseables.lst", "a") as unparseable_file:
                    unparseable_file.write(f"{pdf_path}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_pdfs.py <directory of PDFs>")
        sys.exit(1)

    DIRECTORY = sys.argv[1]
    main(DIRECTORY)
