# Document Sorting Script

This Python script invokes a local [ollama](https://github.com/ollama/ollama) instance to organize various document types in a given directory by parsing their content. It moves each file into a directory named after the first author (formatted as "Lastname Surname") and renames the file in the format "year title.extension".

## Features

- **Content Parsing**: Extracts metadata like author names, publication year, and title from various file types.
- **File Organization**: Moves files into author-named directories and renames them based on year and title.
- **OCR Capability**: Performs on-the-fly OCR for PDFs and images using pytesseract.
- **Wide Format Support**: Handles PDF, EPUB, DOCX, DOC, XLS, XLSX, PPT, PPTX, ODT, ODS, ODP, JPG, JPEG, PNG, GIF, HTML, XML, RTF, MD, TXT, MOBI, AZW, AZW3, and DJVU files.
- **Verbose Mode**: Optional verbose output for debugging purposes.
- **Multiple Extraction Methods**: Uses various libraries to extract text from different file formats, with fallback options for better success rates.

## Requirements

- **Python**: Ensure Python 3.6+ is installed on your system.
- **Python Libraries**: The script requires specific Python libraries which can be installed via pip:
  ```bash
  pip install PyPDF2 pdf2image pytesseract tqdm openai EbookLib python-docx mobi textract beautifulsoup4
  ```
- **System Dependencies**: 
  For Ubuntu or Debian-based systems:
  ```bash
  sudo apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig djvulibre-bin
  ```
  For macOS with Homebrew:
  ```bash
  brew install libxml2 libxslt antiword unrtf poppler tesseract swig djvulibre
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

### Optional Arguments
- `--verbose`: Enable verbose output for debugging.
- `--force`: Automatically execute rename commands after processing (use with caution).

Note that the results will naturally be imperfect. Therefore, the renaming is not done automatically. Rather, a script "rename_commands.sh" is created. You should check it manually for inconsistencies (you can just delete or edit those). And then make that shell script executable and start it:
```bash
chmod +x ./rename_commands.sh
./rename_commands.sh
```

## Notes

- The script will sanitize filenames by removing problematic characters to ensure compatibility with various filesystems.
- If no year can be parsed from the file content, the year in the filename will be left blank.
- The script retries metadata extraction multiple times to ensure accuracy, especially for author names.
- Files that cannot be parsed are listed in "unparseables.lst".
- Customize the script's model or configurations as necessary to fit your specific needs.
- For some filetapes, the script uses multiple methods (mobi library, ebooklib, and textract) to attempt text extraction.

## Limitations

- The script assumes it has read access to all files and write access to create new directories and files.
- For very large files, the script might be slow or use a lot of memory as it loads entire files into memory.
- The accuracy of metadata extraction depends on the quality of the input files and the capabilities of the Ollama model used.

## Troubleshooting

- If you encounter issues with specific file formats, ensure you have the necessary libraries and system dependencies installed.
- Make sure you have the relevant Python packages, like `mobi` and `ebooklib`, installed.
- If Ollama server communication fails, check that the server is running and accessible at `http://localhost:11434/v1/`, or change that in the script according to your setup.

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request on the project repository.

