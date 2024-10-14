# Document Sorting Script

This Python script leverages a local [Ollama](https://github.com/ollama/ollama) instance to organize various document types within a specified directory. By parsing the content of each document, the script moves files into directories named after the first author (formatted as "Lastname Surname") and renames each file following the format "year title.extension".

## Features

- **Comprehensive Content Parsing**: Extracts metadata such as author names, publication year, and title from a wide array of file types.
- **Automated File Organization**: Automatically moves files into author-specific directories and renames them based on extracted metadata.
- **Advanced OCR Capabilities**: Performs Optical Character Recognition (OCR) on PDFs and image files using both `pytesseract` and `Kraken` for enhanced text extraction.
- **Extensive Format Support**: Handles numerous file formats including:
  - **Documents**: PDF, EPUB, DOCX, DOC, XLS, XLSX, PPT, PPTX, ODT, ODS, ODP, HTML, XML, RTF, MD, TXT
  - **Images**: JPG, JPEG, PNG, GIF
  - **E-books**: MOBI, AZW, AZW3
  - **Others**: DJVU, CSV, JSON, WAV, MP3, etc.
- **Multiple Extraction Methods**: Utilizes various libraries (`textract`, `pdfminer`, `PyPDF2`, `EbookLib`, `python-docx`, `mobi`, and more) with fallback options to maximize text extraction success rates.
- **Verbose and Debug Modes**: Offers detailed logging for debugging and tracking purposes.
- **Concurrent Processing**: Processes multiple files in parallel to speed up operations, with an option to disable concurrency.
- **Graceful Shutdown**: Handles interruptions gracefully, ensuring no data corruption or incomplete operations.
- **Error Handling and Logging**: Logs all operations, warnings, and errors to facilitate troubleshooting. Unprocessable files are listed in `unparseables.lst`.
- **Safe Filename Sanitization**: Cleans filenames to remove problematic characters, ensuring compatibility across different filesystems.
- **Customizable Rename Commands**: Generates a `rename_commands.sh` script with the necessary commands to reorganize and rename files, which can be reviewed and executed manually or automatically.

## Requirements

### Software Dependencies

- **Python**: Ensure Python 3.6+ is installed on your system.

### Python Libraries

Install the required Python libraries using `pip`:

```bash
pip install PyPDF2 pdf2image pytesseract tqdm openai EbookLib python-docx mobi textract beautifulsoup4 kraken
```

### System Dependencies

For Ubuntu or Debian-based Systems:
```bash
sudo apt-get update
sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig djvulibre-bin kraken
```

For macOS with Homebrew:
```bash
brew update
brew install libxml2 libxslt antiword unrtf poppler tesseract swig djvulibre kraken
```

### Additional Notes:

- **Tesseract OCR Languages**: The script, as is, requires specific Tesseract language data files (eng, deu, ara). Ensure these are installed. The script can assist in installing missing language packs. Modify the script if you have other needs.

### Ollama Setup:

1. **Model Installation**: The script uses a local Ollama instance with the cas/spaetzle-v85-7b model by default, suitable for English and German text. You can change the model as needed in the code.

```bash
ollama pull cas/spaetzle-v85-7b
```

2. **Ollama Server**: Ensure the Ollama server is running and accessible at http://localhost:11434/v1/. Modify the script if your setup differs.

## Installation

1. Clone the Repository:
```bash
git clone https://github.com/yourusername/document-sorting-script.git
cd document-sorting-script
```

2. Install Python Dependencies:
```bash
pip install -r requirements.txt
```

3. Install System Dependencies:
   Follow the instructions under System Dependencies based on your operating system.

4. Set Up Ollama:
   - Install Ollama: Follow the Ollama installation guide if not already installed.
   - Pull the Required Model:
   ```bash
   ollama pull cas/spaetzle-v85-7b
   ```

## Usage

Run the script in the directory containing the files you want to sort:

```bash
python sort_files.py ./
```

### Optional Arguments

- `--verbose`: Enable verbose output for detailed logging.
- `--force`: Automatically execute the generated rename commands without manual intervention.
- `--singletask`: Disable concurrent processing and process files sequentially.
- `--debug`: Enable debug mode, which provides detailed logs on stdout only and disables file logging.
- `--no-install`: Do not prompt for installing missing tools. The script will proceed without attempting installations.

### Examples

1. Basic Usage:
```bash
python sort_files.py ./
```

2. Enable Verbose Logging:
```bash
python sort_files.py ./ --verbose
```

3. Force Execute Rename Commands:
```bash
python sort_files.py ./ --force
```

4. Process Files Sequentially:
```bash
python sort_files.py ./ --singletask
```

5. Enable Debug Mode:
```bash
python sort_files.py ./ --debug
```

6. Disable Automatic Installation of Missing Tools:
```bash
python sort_files.py ./ --no-install
```

## Execution Flow

1. **Text Extraction**: The script attempts to extract text from each file using appropriate libraries and OCR tools if necessary.
2. **Metadata Extraction**: Extracted text is sent to the local Ollama instance to obtain metadata (author, year, title).
3. **Rename Commands Generation**: Based on the metadata, the script generates `rename_commands.sh`, containing `mkdir` and `mv` commands to reorganize and rename files.
4. **Optional Execution**: If the `--force` flag is used, the script executes the `rename_commands.sh` automatically. Otherwise, you should review and execute it manually:

```bash
chmod +x ./rename_commands.sh
./rename_commands.sh
```

## Notes

- **Filename Sanitization**: The script removes problematic characters from filenames to ensure compatibility across different filesystems.
- **Handling Missing Metadata**: If no year can be parsed from the file content, the year in the filename will be set to "UnknownYear".
- **Retry Mechanism**: The script retries metadata extraction multiple times to enhance accuracy, especially for author names.
- **Unprocessable Files**: Files that cannot be parsed are listed in `unparseables.lst` for manual review.
- **Customization**: You can customize the Ollama model or other configurations within the script to better fit your specific needs.
- **Multiple Extraction Methods**: For certain file types, the script employs multiple libraries (e.g., mobi, ebooklib, textract) to maximize text extraction success rates.

## Limitations

- **Access Permissions**: The script assumes it has read access to all files and write access to create new directories and files.
- **Performance with Large Files**: Processing very large files may be slow or consume significant memory as the script loads entire files into memory.
- **Dependency on Ollama Model**: The accuracy of metadata extraction heavily relies on the quality and capabilities of the Ollama model used.
- **Incomplete Metadata**: In cases where metadata cannot be fully extracted, the script may leave certain fields blank or use placeholders.

## Troubleshooting

1. **Missing Python Libraries**: If you encounter issues with specific file formats, ensure you have the necessary Python libraries installed. Re-run the installation steps or install missing libraries using pip.

2. **System Dependencies**: Verify that all required system dependencies are installed. Refer to the System Dependencies section for your operating system.

3. **Ollama Server Issues**:
   - Server Not Running: Ensure the Ollama server is up and running. Start it if necessary.
   - Incorrect API Endpoint: The script communicates with Ollama at http://localhost:11434/v1/ by default. Modify the script if your server is hosted elsewhere.

4. **OCR Failures**:
   - Missing Language Packs: Ensure that the required Tesseract language packs (eng, deu, ara) are installed.
   - Kraken OCR Issues: Verify that Kraken OCR is correctly installed and accessible in your system's PATH.

5. **Permission Errors**: If you encounter permission-related errors when executing the rename script, ensure that you have the necessary permissions or run the script with elevated privileges.

6. **Unprocessable Files**: Review the `unparseables.lst` file to identify and manually handle files that the script couldn't process.

## Contributing

Contributions, bug reports, and feature requests are welcome!

## License

MIT License

## Acknowledgements

Among others,
- Ollama for providing the local language model framework.
- Developers of the various Python libraries utilized in this script.
