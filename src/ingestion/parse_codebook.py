import json
import re
from collections import Counter
from pathlib import Path
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CODEBOOK_PATH = (PROJECT_ROOT/"docs"/"cms_ffs_claims_codebook.pdf")

OUTPUT_PATH = (PROJECT_ROOT/"knowledge_base"/"processed"/"cms_codebook_variables.json")



def extract_pdf_text(pdf_path: Path) -> list[str]:
    """Extract text from every page of the CMS codebook."""

    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return pages



# Step 2: Clean repeating PDF page content

def clean_page_text(text: str) -> str:
    """Remove repeating CMS page headers and navigation text."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    cleaned_lines = []

    for line in lines:

        # Remove repeating CMS header
        if line.startswith(
            "Chronic Conditions Warehouse Virtual Research Data Center"):
            continue

        # Remove repeating codebook/version header
        if line.startswith("Medicare FFS Claims (version L) Codebook"):
            continue

        # Remove navigation footer
        if line == "^ Back to TOC ^":
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)



# Step 3: Detect the beginning of a CMS variable

def is_variable_start(
    lines: list[str],
    index: int,
) -> bool:
    """
    Determine whether a line begins a CMS variable entry.

    A real variable entry begins with a SAS-style variable
    name followed by a LABEL field.
    """

    if index + 1 >= len(lines):
        return False

    current_line = lines[index].strip()
    next_line = lines[index + 1].strip()

    variable_pattern = (r"^[A-Z][A-Z0-9_]*(?:\[[A-Z0-9]+\])?$")

    return (re.fullmatch(variable_pattern, current_line,)
        is not None
        and next_line.startswith("LABEL:"))


# Step 4: Collect complete variable entries
def collect_variable_entries(pages: list[str],) -> list[str]:
    """
    Combine pages and identify complete CMS variable entries.

    Joining the pages before identifying boundaries allows
    variable definitions to continue across PDF page breaks.
    """

    cleaned_pages = [clean_page_text(page)for page in pages]

    full_text = "\n".join(cleaned_pages)

    lines = [
        line.strip()
        for line in full_text.splitlines()
        if line.strip()
    ]

    start_indices = []

    # Find every real variable-entry start
    for i in range(len(lines) - 1):

        if is_variable_start(lines, i):
            start_indices.append(i)

    entries = []

    # Everything between one variable start and the next
    # belongs to the current variable.
    for i, start in enumerate(start_indices):

        if i + 1 < len(start_indices):
            end = start_indices[i + 1]

        else:
            end = len(lines)

        entry_lines = lines[start:end]

        entry_text = "\n".join(entry_lines)

        entries.append(entry_text)

    return entries




# Step 5: Parse one variable into structured fields

def parse_variable_entry(text: str) -> dict:
    """Parse one CMS variable entry into structured fields."""

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

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    result = {
        "variable": lines[0]
    }

    remaining_text = "\n".join(lines[1:])

    for i, field in enumerate(fields):

        next_fields = fields[i + 1:]

        if next_fields:

            next_pattern = "|".join(
                re.escape(next_field)
                for next_field in next_fields
            )

            pattern = (
                rf"{re.escape(field)}:\s*(.*?)"
                rf"(?=\n?(?:{next_pattern}):)"
            )

        else:

            pattern = (
                rf"{re.escape(field)}:\s*(.*)$"
            )

        match = re.search(
            pattern,
            remaining_text,
            re.DOTALL,
        )

        if match:

            value = " ".join(
                match.group(1).split()
            )

            key = (
                field
                .lower()
                .replace(" ", "_")
            )

            result[key] = value

    return result




# Step 6: Parse all CMS variables

def parse_all_variables(
    entries: list[str],
) -> list[dict]:
    """Parse all collected CMS variable entries."""

    variables = []

    for entry in entries:

        parsed = parse_variable_entry(entry)

        variables.append(parsed)

    return variables



# Step 7: Validate the parsed knowledge base


def validate_variables(
    variables: list[dict],
) -> bool:
    """
    Validate parsed CMS variable records.

    Returns True when no critical validation problems
    are detected.
    """

    print("\n--- Validation Report ---\n")

    print(
        f"Total parsed variables: {len(variables)}"
    )



    # Check for duplicate variable names

    variable_names = [
        variable.get("variable")
        for variable in variables
    ]

    counts = Counter(variable_names)

    duplicates = [
        name
        for name, count in counts.items()
        if count > 1
    ]

    print(
        f"Duplicate variable names: {len(duplicates)}"
    )

    if duplicates:

        for name in duplicates:
            print(f"  - {name}")



    # Check important fields

    required_fields = [
        "variable",
        "label",
        "description",
        "type",
        "source",
    ]

    critical_problem = False

    for field in required_fields:

        missing = [
            variable.get("variable", "<UNKNOWN>")
            for variable in variables
            if not variable.get(field)
        ]

        print(
            f"Missing {field}: {len(missing)}"
        )

        if missing:

            critical_problem = True

            # Print only the first 10 so the terminal
            # does not become overwhelming.
            for name in missing[:10]:
                print(f"  - {name}")

            if len(missing) > 10:

                print(
                    f"  ... and "
                    f"{len(missing) - 10} more"
                )


    # Check optional VALUES and COMMENT fields

    missing_values = [
        variable.get("variable", "<UNKNOWN>")
        for variable in variables
        if not variable.get("values")
    ]

    missing_comments = [
        variable.get("variable", "<UNKNOWN>")
        for variable in variables
        if not variable.get("comment")
    ]

    print(
        f"Missing values field: "
        f"{len(missing_values)}"
    )

    print(
        f"Missing comment field: "
        f"{len(missing_comments)}"
    )



    # Display first and last variables as a sanity check

    if variables:

        print(
            f"First variable: "
            f"{variables[0].get('variable')}"
        )

        print(
            f"Last variable: "
            f"{variables[-1].get('variable')}"
        )


    # Final result

    if duplicates:
        critical_problem = True

    if critical_problem:

        print(
            "\nValidation result: "
            "ISSUES FOUND"
        )

        return False

    print(
        "\nValidation result: PASSED"
    )

    return True




# Step 8: Save the structured knowledge base

def save_variables_to_json(
    variables: list[dict],
    output_path: Path,
) -> None:
    """Save parsed CMS variables as formatted JSON."""

    # Create the directory if it does not exist
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            variables,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nJSON saved to:\n{output_path}"
    )



# Main ingestion pipeline


if __name__ == "__main__":

    print(
        "\n=== CMS Codebook Ingestion Pipeline ===\n"
    )

    # Step 1
    pages = extract_pdf_text(
        CODEBOOK_PATH
    )

    print(
        f"PDF: {CODEBOOK_PATH.name}"
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # Step 2
    entries = collect_variable_entries(
        pages
    )

    print(
        f"Variable entries found: {len(entries)}"
    )

    # Step 3
    variables = parse_all_variables(
        entries
    )

    print(
        f"Variable entries parsed: {len(variables)}"
    )

    # Step 4
    validation_passed = validate_variables(
        variables
    )

    # Step 5
    if validation_passed:

        save_variables_to_json(
            variables,
            OUTPUT_PATH,
        )

    else:

        print(
            "\nJSON was NOT created because "
            "validation found problems."
        )