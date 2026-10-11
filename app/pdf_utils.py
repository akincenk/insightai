import fitz

def extract_clean_text_from_pdf(pdf_file_path):
    text = ""
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    cleaned = ' '.join(text.replace('\n', ' ').split())
    return cleaned