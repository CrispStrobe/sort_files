import os
import sys
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI
from tqdm import tqdm

modelname="cas/spaetzle-v58:latest"

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

def valid_author_name(author_names):
    """ Check if the author names are valid, i.e., each name has at least two words. """
    #return all(len(name.split()) > 1 for name in author_names.split(", "))
    """Check if the author name is valid, i.e., contains at least a first name and a last name."""
    return len(author_names.strip().split()) > 1

def sort_author_names(author_names, attempt=1):
    if attempt > 5:
        print("Maximum retry attempts reached for sorting names.")
        return author_names
    #print ("\nsorting author names:", author_names)
    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )

    prompt = (f"You will be given an author name that you must put into the format 'Lastname Surname'."
              f"So, you must first make an educated guess if the given input is already in this format. If so, return it back."
              f"If not and it is more plausibly in the format 'Surname(s) Lastname', you must reformat it."
              f"Example: 'Jan Christian Gertz' must become 'Gertz Jan Christian' and 'Michael M. Meier' must become 'Meier Michael M'."
              f"No comma after the Lastname!"
              f"If you are given multiple person names, only keep the first and omit all others."
              f"You must give the output in the format: <AUTHOR>Lastname Surname(s)</AUTHOR>."
              f"Here are the name parts: <AUTHOR>{author_names}</AUTHOR>")

    messages = [{"role": "user", "content": prompt}]
    #print ("sorting prompt:", messages)
    response = client.chat.completions.create(
        model=modelname,
        temperature=0.5,
        max_tokens=500,
        messages=messages
    )
    
    reformatted_name = response.choices[0].message.content.strip()
    #print ("\nmodel output:", reformatted_name)
    #names = re.findall(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_names)
    name = re.search(r'<AUTHOR>(.*?)</AUTHOR>', reformatted_name)
    #print ("\nonly the name part:", name) 
    #ordered_names = ", ".join(names)
    #print ("\sorted author names:", ordered_names)
    #if "lastname" in ordered_names.lower() or "surname" in ordered_names.lower() or not valid_author_name(ordered_names):
    #    print("\nRetrying author names sorting due to detection of placeholder names or invalid format: ", ordered_names)
    #    return sort_author_names(author_names, attempt + 1)
    #return ordered_names
    if name:
        ordered_name = name.group(1)
        #print ("ordered name:", ordered_name)
        if not valid_author_name(ordered_name):
            print(f"Invalid author name format detected: '{ordered_name}', retrying...")
            return sort_author_names(author_names, attempt + 1)
        return ordered_name
    else:
        print("Failed to extract a valid name, retrying...")
        return sort_author_names(author_names, attempt + 1)

def send_to_ollama_server(text, filename, attempt=1):
    client = OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    prompt = (f"Extract the first author nam (ignore other authors), year of publication, and title from the following text, considering the filename '{base_filename}' which may contain clues. "
              "I need the output in the following format: "
              "<TITLE>The publication title</TITLE>, <YEAR>2023</YEAR>, <AUTHOR>Lastname Surname</AUTHOR>. "
              "Here is the extracted text:\n" + text)
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

    if ("lastname" in output.lower() or "surname" in output.lower()) and attempt < 4:
        return send_to_ollama_server(text, filename, attempt + 1)
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

    if "unknown" in (title.lower(), year.lower(), author.lower()) or len(author.split()) <= 1:
        return None
    
    if "lastname" in author.lower() or "surname" in author.lower():
        return None  # Check for placeholder text in author names

    # Return the first found metadata
    return {'author': author, 'year': year, 'title': title}


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def process_pdf(pdf_path, attempt=1):
    if attempt > 4:  # Limiting the retries to prevent infinite loops
        print(f"Maximum retry attempts reached for {pdf_path}.")
        return None

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        text = perform_ocr(pdf_path)

    metadata_content = send_to_ollama_server(text, pdf_path, attempt)
    metadata = parse_metadata(metadata_content)
    if metadata:
        corrected_authors = sort_author_names(metadata['author'])
        metadata['author'] = corrected_authors
        # After sorting, check if the author's name still has one word only
        if len(metadata['author'].split()) <= 1:
            print(f"Author's name still has insufficient details after sorting, retrying for {pdf_path}. Attempt {attempt}")
            return process_pdf(pdf_path, attempt + 1)
        return metadata
    else:
        print(f"Error: Metadata parsing failed or incomplete for {pdf_path}. Metadata content: {metadata_content}, retrying... Attempt {attempt}")
        return process_pdf(pdf_path, attempt + 1)

def main(directory):
    if not os.path.exists(directory):
        print("The specified directory does not exist")
        sys.exit(1)

    with open("rename_commands.sh", "w") as mv_file, open("unparseables.lst", "w") as unparseable_file:
        mv_file.write("#!/bin/sh\n")
        mv_file.flush()
        for filename in tqdm(os.listdir(directory)):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(directory, filename)
                try:
                    metadata = process_pdf(pdf_path)
                    if metadata:
                        first_author = metadata['author'].split(", ")[0]
                        target_dir = os.path.join(directory, first_author)
                        new_file_path = os.path.join(target_dir, f"{metadata['year']} {sanitize_filename(metadata['title'])}.pdf")
                        
                        mv_file.write(f"mkdir -p \"{target_dir}\"\n")
                        mv_file.write(f"mv \"{pdf_path}\" \"{new_file_path}\"\n")
                        mv_file.flush()
                    else:
                        unparseable_file.write(f"{pdf_path}\n")
                        unparseable_file.flush()
                except Exception as e:
                    print(f"Failed to process {pdf_path}: {e}")
                    unparseable_file.write(f"{pdf_path}\n")
                    unparseable_file.flush()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_pdfs.py <directory of PDFs>")
        sys.exit(1)

    DIRECTORY = sys.argv[1]
    main(DIRECTORY)
