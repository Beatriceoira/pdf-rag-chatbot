"""Create a minimal synthetic PDF for testing."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter


def create_minimal_pdf(text: str, path: Path | None = None) -> bytes:
    """Create a minimal valid PDF containing the given text.

    Returns the PDF as bytes. If path is given, also writes to disk.
    """
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    # Use pypdf to add text content directly
    # For a minimal test PDF, we use a pre-built simple PDF
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, text)
    c.showPage()
    c.save()
    buffer.seek(0)
    pdf_bytes = buffer.read()

    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes)

    return pdf_bytes


def create_test_pdfs(directory: Path) -> dict[str, Path]:
    """Create sample PDFs for integration testing.

    Returns dict mapping filename → path.
    """
    directory.mkdir(parents=True, exist_ok=True)

    files = {
        "company_policy.pdf": (
            "The company provides 20 vacation days per year.\n"
            "Employees must request time off at least 2 weeks in advance.\n"
            "Sick leave is available up to 10 days per year.\n"
        ),
        "product_manual.pdf": (
            "Product X has a 30-day return policy.\n"
            "Warranty covers manufacturing defects for 1 year.\n"
            "Contact support at support@example.com for issues.\n"
        ),
        "employee_handbook.pdf": (
            "The dress code is business casual.\n"
            "Office hours are 9 AM to 5 PM, Monday through Friday.\n"
            "Remote work is available on Wednesdays.\n"
        ),
    }

    results = {}
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        for name, content in files.items():
            path = directory / name
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            lines = content.strip().split("\n")
            y = 750
            for line in lines:
                c.drawString(100, y, line)
                y -= 20
            c.showPage()
            c.save()
            buffer.seek(0)
            path.write_bytes(buffer.read())
            results[name] = path
    except ImportError:
        # Fallback: create a minimal valid PDF manually
        for name, content in files.items():
            path = directory / name
            create_minimal_pdf(content, path)
            results[name] = path

    return results
