from PyPDF2 import PdfReader, PdfWriter

def crop_pdf(input_pdf_path, output_pdf_path, start_page, end_page):
    """
    Crop a PDF file to include only the specified page range.

    :param input_pdf_path: Path to the input PDF file.
    :param output_pdf_path: Path to save the cropped PDF file.
    :param start_page: The starting page number (1-based index).
    :param end_page: The ending page number (1-based index).
    """
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Ensure the page range is valid
    if start_page < 1 or end_page > len(reader.pages) or start_page > end_page:
        raise ValueError("Invalid page range specified.")

    # Add the specified pages to the writer
    for i in range(start_page - 1, end_page):
        writer.add_page(reader.pages[i])

    # Write the cropped PDF to the output file
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)

# Example usage
# crop_pdf("input.pdf", "output.pdf", 2, 5)

if __name__ == '__main__':
    crop_pdf('test.pdf', '文献综述.pdf', 10, 30)