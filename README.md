# Document Sorting Script

This Python script organizes various document types in a given directory by parsing their content. It moves each file into a directory named after the first author (formatted as "Lastname Surname") and renames the file in the format "year title.extension".

## Features

- **Content Parsing**: Extracts metadata like author names, publication year, and title from various file types.
- **File Organization**: Moves files into author-named directories and renames them based on year and title.
- **OCR Capability**: Performs on-the-fly OCR for PDFs and images using pytesseract.
- **Wide Format Support**: Handles PDF, EPUB, DOCX, DOC, XLS, XLSX, PPT, PPTX, ODT, ODS, ODP, JPG, JPEG, PNG, GIF, HTML, XML, RTF, MD, TXT, and MOBI files.
- **Verbose Mode**: Optional verbose output for debugging purposes.

## Requirements

- **Python**: Ensure Python is installed on your system.
- **Python Libraries**: The script requires specific Python libraries which can be installed via pip:

  ```bash
  pip install PyPDF2 pdf2image pytesseract tqdm openai EbookLib python-docx mobi textract
  ```

- **System Dependencies**: 
  For Ubuntu or Debian-based systems:
  ```bash
  sudo apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig
  ```
  For macOS with Homebrew:
  ```bash
  brew install libxml2 libxslt antiword unrtf poppler tesseract swig
  ```

- **Ollama Setup**:
  - **Model Installation**: The script uses a local Ollama instance with a specific LLM model. The default is `cas/spaetzle-v85-7b`, suitable for English and German text. Change the model as per your requirements.
    ```bash
    ollama pull cas/spaetzle-v85-7b
    ```

## Usage

Run the script in the directory containing the files you want to sort:

```bash
python sort_files.py ./
```

Or optionally with verbose output enabled for debugging by adding the `--verbose` flag:

```bash
python sort_files.py ./ --verbose
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
- If no year can be parsed from the file content, the year in the filename will be left blank.
- The script retries metadata extraction multiple times to ensure accuracy, especially for author names.
- Files that cannot be parsed are listed in "unparseables.lst".
- Customize the script's model or configurations as necessary to fit your specific needs.

## Limitations

- The script assumes it has read access to all files and write access to create new directories and files.
- For very large files, the script might be slow or use a lot of memory as it loads entire files into memory.
- The accuracy of metadata extraction depends on the quality of the input files and the capabilities of the Ollama model used.
