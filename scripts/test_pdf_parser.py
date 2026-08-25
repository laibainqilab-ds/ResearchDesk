from app.ingestion.pdf_parser import parse_pdf


pdf_path = "data/documents/SCRIPT FOR DISCOVERY.pdf"

pages = parse_pdf(pdf_path)

print(f"Number of pages: {len(pages)}")

for page in pages:
    print(f"\nPage {page['page_number']}:")
    print(page["text"][:500])