from app.ingestion.pdf_parser import parse_pdf
from app.ingestion.text_cleaner import clean_text


pdf_path = "data/documents/SCRIPT FOR DISCOVERY.pdf"

pages = parse_pdf(pdf_path)

for page in pages[:2]:
    cleaned = clean_text(page["text"])

    print(f"\nPage {page['page_number']}:")
    print(cleaned[:500])