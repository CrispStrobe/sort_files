# PDF File Sorting Script

This quickly hacked Python script organizes all PDFs and EPUBs in a given directory by parsing their content. It moves each file into a directory named after the first author (formatted as "Lastname Surname") and renames the file in the format "year title.pdf".

## Features

- **Content Parsing**: Extracts metadata like author names, publication year, and title from PDF and EPUB files.
- **File Organization**: Moves PDFs and EPUBs into author-named directories and renames them based on year and title.
- **OCR Capability**: Performs on-the-fly OCR as needed using pytesseract.
- **Verbose Mode**: Optional verbose output for debugging purposes.

## Requirements

- **Python**: Ensure Python is installed on your system.
- **Python Libraries and System Dependencies**: The script requires specific python libraries which can be installed via pip:
  ```bash
  pip install PyPDF2 pdf2image pytesseract tqdm openai EbookLib
  ```
  Also, Poppler and Tesseract are necessary for PDF to image conversion and OCR functionality:
  ```bash
  sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng 
  ```
  For **Windows** (without WSL), there is an extra version. For it to work, install some alternative libraries: 
  ```terminal
  pip install PyPDF2 pytesseract pymupdf openai tqdm ebooklib beautifulsoup4
  ```
- **Ollama Setup**:
  - **Model Installation**: The script uses a local Ollama instance with a specific LLM model. The default is `cas/spaetzle-v60-7b`, suitable for English and German text. Change the model as per your requirements.
    ```bash
    ollama pull cas/spaetzle-v60-7b
    ```

## Usage

Run the script in the directory containing the PDFs and EPUBs you want to sort:

```bash
python parse_pdfs.py ./ --verbose
```

This command sorts all PDFs in the current directory with verbose output enabled for debugging. If you want to run it without verbose output, simply omit the `--verbose` flag:

```bash
python parse_pdfs.py ./
```

Note that the results will naturally be imperfect. Therefore, the renaming is not done automatically. Rather, a script "rename_commands.sh" is created. You should check it manually for inconsistencies (you can just delete or edit those). And then make that shell script executable and start it:

```bash
chmod +x ./rename_commands.sh
./rename_commands.sh 
```

### Optional Arguments

- `--verbose`: Enable verbose output for debugging.

## Notes

- The script will sanitize filenames by removing problematic characters to ensure compatibility with various filesystems.
- If no year can be parsed from the PDF content, the year in the filename will be left blank.
- The script retries metadata extraction multiple times to ensure accuracy, especially for author names.

Customize the script's model or configurations as necessary to fit your specific needs.
