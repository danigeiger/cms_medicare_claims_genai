import re
from pathlib import Path

from pypdf import PdfReader


# Find the root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of the CMS codebook
CODEBOOK_PATH = PROJECT_ROOT / "docs" / "cms_ffs_claims_codebook.pdf"


def extract_pdf_text(pdf_path: Path) -> list[str]:
    """Extract text from each page of a PDF."""

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return pages


def parse_variable_entry(text: str) -> dict:
    """Parse a CMS codebook variable entry into structured fields."""

    fields = [
        "LABEL",
        "DESCRIPTION",
        "SHORT NAME",
        "LONG NAME",
        "TYPE",
        "LENGTH",
        "SOURCE",
        "VALUES",
        "COMMENT",
    ]

    result = {}

    # Split the page into lines and remove blank lines
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Remove the two-line CMS page header
    lines = lines[2:]

    # Remove the navigation footer
    lines = [
        line
        for line in lines
        if line != "^ Back to TOC ^"
    ]

    # The first remaining line is the CMS variable name
    result["variable"] = lines[0]

    # Combine everything after the variable name
    remaining_text = "\n".join(lines[1:])

    # Extract each labeled section
    for i, field in enumerate(fields):
        next_fields = fields[i + 1:]

        if next_fields:
            next_pattern = "|".join(
                re.escape(f) for f in next_fields
            )

            pattern = (
                rf"{re.escape(field)}:\s*(.*?)"
                rf"(?=\n?(?:{next_pattern}):)"
            )

        else:
            pattern = rf"{re.escape(field)}:\s*(.*)$"

        match = re.search(
            pattern,
            remaining_text,
            re.DOTALL,
        )

        if match:
            value = " ".join(
                match.group(1).split()
            )

            key = field.lower().replace(" ", "_")

            result[key] = value

    return result


if __name__ == "__main__":
    # Extract all 411 pages from the CMS codebook
    pages = extract_pdf_text(CODEBOOK_PATH)

    print(f"PDF: {CODEBOOK_PATH.name}")
    print(f"Pages extracted: {len(pages)}")

    # PDF page 19 contains the ADMTG_DGNS_CD example
    example = parse_variable_entry(pages[18])

    print("\n--- Parsed variable ---\n")

    for key, value in example.items():
        print(f"{key}: {value}")