# Document Archive System

This is an intelligent document archiving system built with LangChain that allows you to:
- Archive various document types (PDF, DOCX, TXT, etc.)
- Search through documents using semantic search
- Ask questions about your documents using AI

## Setup

1. Create a virtual environment and activate it:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install the required packages:
```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Google API key:
```
GOOGLE_API_KEY=your-api-key-here
```

Note: This project now uses Donut (a vision-language model) for structured extraction from document images/PDFs. To use Donut and the image/PDF handling features, install the additional dependencies below (these are included in `requirements.txt`):

- transformers
- torch
- pillow
- pdf2image
- pytesseract (optional, for OCR fallback)

On Windows, install the Tesseract engine separately if you want OCR support (pytesseract is only a Python wrapper). Download the installer from: https://github.com/tesseract-ocr/tesseract

## Usage

Run the main script:
```powershell
python main.py
```

The script provides an interactive menu with the following options:
1. Add documents to the archive
2. Search through archived documents
3. Ask questions about the archived documents
4. Exit

### Adding Documents
When adding documents, provide the full path to each document, separated by commas.

### Searching Documents
Enter a search query to find relevant content in your archived documents. The system will return the most semantically similar passages.

### Asking Questions
You can ask questions about your archived documents, and the system will use AI to generate answers based on the content.

## Document Support
The system supports the following document types:
- PDF files (.pdf)
- Word documents (.doc, .docx)
- Text files (.txt)
- Other document types (using Unstructured library)

## Storage
Documents are processed and stored as vector embeddings in the `vector_store` directory within your archive folder.