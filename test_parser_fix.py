#!/usr/bin/env python3
"""Test script to verify the PDF parser fix."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from src.ingestion.parser import parse_pdf_from_bytes


def test_parser():
    """Test the PDF parser with a simple PDF."""
    # Create a simple PDF in memory
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000117 00000 n
0000000210 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
298
%%EOF"""

    print(f"Testing with PDF content of length: {len(pdf_content)} bytes")

    try:
        docs, meta = parse_pdf_from_bytes(pdf_content, "test.pdf")
        print(f"Success! Parsed {len(docs)} documents")
        print(f"Metadata: {meta}")
        if docs:
            print(f"First doc content: {docs[0].page_content}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parser()