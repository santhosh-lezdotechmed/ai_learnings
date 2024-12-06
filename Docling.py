# Import necessary libraries
from pathlib import Path
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Define the main function
def main(pdf_path: str):

    # Validate the input PDF path
    input_doc = Path(pdf_path)
    if not input_doc.is_file():
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")

    # Setup the pipeline options
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
    )
    pipeline_options.table_structure_options.do_cell_matching = True

    # OCR options (using EasyOCR)
    pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True)

    # Create the document converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # Convert the document and export to markdown
    doc = converter.convert(input_doc).document
    markdown_content = doc.export_to_markdown()

    # Print the converted Markdown content
    print(markdown_content)

# Execute the function
if __name__ == "__main__":
    pdf_file_path = "/home/Redacted.pdf"  # Replace with the actual path to your PDF
    main(pdf_file_path)
