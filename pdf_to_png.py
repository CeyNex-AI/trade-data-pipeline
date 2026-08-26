import fitz  # PyMuPDF
from pathlib import Path


def pdf_to_png(pdf_path, output_folder, dpi=300):
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    for page_number, page in enumerate(doc, start=1):
        # Render page at the specified DPI
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Numerical ordering: page_001.png, page_002.png, ...
        output_path = output_folder / f"page_{page_number:03d}.png"

        pix.save(output_path)
        print(f"Saved: {output_path}")

    doc.close()
    print(f"\nConverted {len(doc)} pages.")


if __name__ == "__main__":
    pdf_to_png(
        pdf_path="national-export-strategy-of-sri-lanka.pdf",
        output_folder="pages",
        dpi=150
    )