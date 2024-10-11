import os
import PyPDF2
import docx

def extract_text(file_path):
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    
    # Handling .pdf files
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    
    # Handling .docx files
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    
    # Raise error for unsupported file formats
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only '.pdf' and '.docx' are allowed.")

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])
