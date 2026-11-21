
import io
import logging
import re
from collections import Counter
from math import ceil

import pdfplumber
import pymupdf
import pytesseract
from PIL import Image


# --------------------------------------------------
# 1. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. BASIC TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving
    paragraph and line structure.
    """

    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Join words broken by PDF line wrapping.
    text = re.sub(
        r"(?<=\w)-\n(?=[a-z])",
        "",
        text
    )

    lines = []

    for line in text.split("\n"):

        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        lines.append(line)

    text = "\n".join(lines)

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# --------------------------------------------------
# 3. HEADING AND SUBJECT DETECTION
# --------------------------------------------------

def is_heading(line: str) -> bool:
    """
    Identify common document headings.
    """

    line = line.strip()

    if not line:
        return False

    patterns = [
        r"^unit\s+\d+",
        r"^module\s+\d+",
        r"^chapter\s+[\w\d]+",
        r"^section\s+\d+",
        r"^subject\s*:",
        r"^course\s+objectives?",
        r"^course\s+outcomes?",
        r"^references\s*:?",
        r"^textbooks\s*:?",
        r"^contents\s*:?",
        r"^syllabus\s*:?"
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            line,
            re.IGNORECASE
        ):
            return True

    # Short uppercase headings.
    if (
        len(line) < 90
        and line.isupper()
        and len(line.split()) <= 12
    ):
        return True

    return False


def is_new_subject(line: str) -> bool:
    """
    Detect known subject titles.

    Add more subject names here when needed.
    """

    line = line.strip()

    if not line:
        return False

    patterns = [
        r"^Introduction to Machine Learning$",
        r"^Operating Systems$",
        r"^Object Oriented Programming$",
        r"^Cloud Computing$",
        r"^Introduction to Industrial Management.*$"
    ]

    return any(
        re.match(
            pattern,
            line,
            re.IGNORECASE
        )
        for pattern in patterns
    )


# --------------------------------------------------
# 4. REMOVE REPEATED HEADERS AND FOOTERS
# --------------------------------------------------

def remove_repeated_headers(pages_data: list) -> list:
    """
    Detect repeated lines near page boundaries
    and remove them from the document.
    """

    if len(pages_data) < 2:
        return pages_data

    line_counts = Counter()

    for page_data in pages_data:

        lines = page_data["text"].splitlines()

        candidate_lines = (
            lines[:5] + lines[-3:]
        )

        normalized_candidates = {
            re.sub(
                r"\s+",
                " ",
                line
            ).strip().lower()

            for line in candidate_lines

            if len(line.strip()) >= 10
        }

        line_counts.update(normalized_candidates)

    threshold = max(
        2,
        ceil(len(pages_data) * 0.4)
    )

    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= threshold
    }

    logger.info(
        "Detected %s repeated header/footer lines.",
        len(repeated_lines)
    )

    for page_data in pages_data:

        lines = page_data["text"].splitlines()

        boundary_indices = set(
            range(min(5, len(lines)))
        )

        if len(lines) > 8:

            boundary_indices.update(
                range(len(lines) - 3, len(lines))
            )

        cleaned_lines = []

        for index, line in enumerate(lines):

            normalized = re.sub(
                r"\s+",
                " ",
                line
            ).strip().lower()

            if (
                index in boundary_indices
                and normalized in repeated_lines
            ):
                continue

            cleaned_lines.append(line)

        page_data["text"] = clean_text(
            "\n".join(cleaned_lines)
        )

    return pages_data


# --------------------------------------------------
# 5. STANDARD DIGITAL PDF EXTRACTION
# --------------------------------------------------

def extract_standard_pdf(pdf_path: str) -> list:
    """
    Extract selectable text using PyMuPDF.
    """

    pages_data = []

    with pymupdf.open(pdf_path) as doc:

        for page_num, page in enumerate(doc):

            text = clean_text(
                page.get_text("text")
            )

            pages_data.append({
                "page": page_num + 1,
                "text": text,
                "extraction_method": "digital"
            })

    return pages_data


# --------------------------------------------------
# 6. OPTIONAL PDF TABLE EXTRACTION
# --------------------------------------------------

def extract_pdf_with_tables(pdf_path: str) -> list:
    """
    Extract normal text and tables using pdfplumber.

    This is an optional extraction function.
    The active smart extraction pipeline below
    does not call this function yet.
    """

    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_num, page in enumerate(pdf.pages):

            page_content = []

            text = page.extract_text()

            if text:
                page_content.append(
                    clean_text(text)
                )

            tables = page.extract_tables()

            for table_num, table in enumerate(tables):

                page_content.append(
                    f"Table {table_num + 1}:"
                )

                for row in table:

                    clean_row = []

                    for cell in row:

                        if cell is None:
                            cell = ""

                        cell = clean_text(
                            str(cell).replace("\n", " ")
                        )

                        clean_row.append(cell)

                    page_content.append(
                        " | ".join(clean_row)
                    )

            final_text = "\n".join(page_content)

            pages_data.append({
                "page": page_num + 1,
                "text": clean_text(final_text),
                "extraction_method": "digital_with_tables"
            })

    return pages_data


# --------------------------------------------------
# 7. OCR EXTRACTION
# --------------------------------------------------

def perform_ocr(page) -> str:
    """
    Convert a PDF page into an image
    and extract text using Tesseract.
    """

    pix = page.get_pixmap(
        dpi=300,
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    img = Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    )

    text = pytesseract.image_to_string(
        img,
        config="--psm 3"
    )

    return clean_text(text)


def extract_scanned_pdf(pdf_path: str) -> list:
    """
    Run OCR on every page of a scanned PDF.
    """

    pages_data = []

    with pymupdf.open(pdf_path) as doc:

        for page_num, page in enumerate(doc):

            logger.info(
                "Running OCR on page %s",
                page_num + 1
            )

            text = perform_ocr(page)

            pages_data.append({
                "page": page_num + 1,
                "text": text,
                "extraction_method": "ocr"
            })

    return pages_data


# --------------------------------------------------
# 8. OCR QUALITY CHECK
# --------------------------------------------------

def needs_ocr(text: str) -> bool:
    """
    Decide whether digital extraction produced
    sufficiently useful text.
    """

    if not text:
        return True

    text = text.strip()

    if len(text) < 30:
        return True

    meaningful_chars = sum(
        char.isalnum()
        for char in text
    )

    ratio = meaningful_chars / len(text)

    return ratio < 0.50


# --------------------------------------------------
# 9. HYBRID SMART EXTRACTION
# --------------------------------------------------

def smart_extract_pdf(pdf_path: str) -> list:
    """
    First attempt digital extraction.
    Use OCR when digital text is weak.
    """

    pages_data = []

    with pymupdf.open(pdf_path) as doc:

        for page_num, page in enumerate(doc):

            text = clean_text(
                page.get_text("text")
            )

            method = "digital"

            if needs_ocr(text):

                logger.info(
                    "Page %s: Weak digital text. Running OCR.",
                    page_num + 1
                )

                try:

                    ocr_text = perform_ocr(page)

                    if len(ocr_text) > len(text):

                        text = ocr_text
                        method = "ocr"

                    else:

                        method = "digital_fallback"

                except Exception:

                    logger.exception(
                        "OCR failed on page %s",
                        page_num + 1
                    )

                    method = "digital_fallback"

            else:

                logger.info(
                    "Page %s: Digital extraction successful.",
                    page_num + 1
                )

            pages_data.append({
                "page": page_num + 1,
                "text": text,
                "extraction_method": method
            })

    pages_data = remove_repeated_headers(
        pages_data
    )

    return pages_data


# --------------------------------------------------
# 10. CONVERT PAGE DATA INTO ONE STRING
# --------------------------------------------------

def pages_to_text(pages_data: list) -> str:
    """
    Convert extracted page data into one string.
    """

    full_text = []

    for page_data in pages_data:

        page_num = page_data["page"]
        text = page_data["text"]

        full_text.append(
            f"\n--- Page {page_num} ---\n{text}"
        )

    return "\n".join(full_text)


# --------------------------------------------------
# 11. EXTRACT PDF AND RETURN PLAIN TEXT
# --------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:

    pages_data = smart_extract_pdf(pdf_path)

    return pages_to_text(pages_data)


# --------------------------------------------------
# 12. EXTRACT PARAGRAPHS
# --------------------------------------------------

def extract_paragraphs(text: str) -> list:
    """
    Preserve paragraph, heading and numbered-list
    boundaries instead of joining everything blindly.
    """

    lines = text.splitlines()

    paragraphs = []
    current = []

    list_pattern = re.compile(
        r"^\s*(?:\d+[.)]\s+|[•●▪\-]\s+)"
    )

    def save_current():

        nonlocal current

        if current:

            paragraphs.append(
                " ".join(current).strip()
            )

            current = []

    for line in lines:

        line = line.strip()

        if not line:

            save_current()
            continue

        # Keep headings separate.
        if is_heading(line):

            save_current()
            paragraphs.append(line)
            continue

        # Keep each numbered or bulleted item separate.
        if list_pattern.match(line):

            save_current()
            paragraphs.append(line)
            continue

        current.append(line)

    save_current()

    return paragraphs


# --------------------------------------------------
# 13. SPLIT TEXT INTO SENTENCES
# --------------------------------------------------

def split_sentences(text: str) -> list:
    """
    Split sentences while protecting common
    abbreviations, decimal numbers and list markers.
    """

    if not text.strip():
        return []

    protected = text

    protected_tokens = {}

    def protect(pattern, replacement_function=None):

        nonlocal protected

        def replace_match(match):

            token = f"PROTECTEDTOKEN{len(protected_tokens)}X"

            original = match.group(0)

            if replacement_function:
                original = replacement_function(match)

            protected_tokens[token] = original

            return token

        protected = re.sub(
            pattern,
            replace_match,
            protected,
            flags=re.IGNORECASE
        )

    # Protect common abbreviations.
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.",
        "Sr.", "Jr.", "e.g.", "i.e.", "etc.",
        "vs.", "No.", "Fig.", "Dept.", "Ed."
    ]

    for abbreviation in abbreviations:

        protect(
            re.escape(abbreviation)
        )

    # Protect decimal points, e.g. 3.14.
    protect(
        r"(?<=\d)\.(?=\d)"
    )

    # Protect initials, e.g. A. Kumar.
    protect(
        r"\b[A-Z]\.(?=\s*[A-Z])"
    )

    # Protect numbered list markers, e.g. 1. First item.
    protect(
        r"(?m)^\s*\d+\.(?=\s)"
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        protected
    )

    results = []

    for sentence in sentences:

        # Restore protected text.
        for token, original in protected_tokens.items():

            sentence = sentence.replace(
                token,
                original
            )

        sentence = sentence.strip()

        if sentence:
            results.append(sentence)

    return results


# --------------------------------------------------
# 14. CREATE DOCUMENT SEGMENTS
# --------------------------------------------------

def create_segments(pages_data: list) -> list:
    """
    Convert pages into ordered segments.

    Each segment stores:
    - text
    - page number
    - subject
    - segment type
    - extraction method
    """

    segments = []

    current_subject = None

    for page_data in pages_data:

        page_number = page_data["page"]

        extraction_method = page_data.get(
            "extraction_method",
            "unknown"
        )

        paragraphs = extract_paragraphs(
            page_data["text"]
        )

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Detect a new subject.
            if is_new_subject(paragraph):

                current_subject = paragraph

                segments.append({
                    "text": paragraph,
                    "page": page_number,
                    "subject": current_subject,
                    "type": "subject_heading",
                    "extraction_method": extraction_method
                })

                continue

            # Detect a normal heading.
            if is_heading(paragraph):

                segments.append({
                    "text": paragraph,
                    "page": page_number,
                    "subject": current_subject,
                    "type": "heading",
                    "extraction_method": extraction_method
                })

                continue

            sentences = split_sentences(paragraph)

            for sentence in sentences:

                if not sentence.strip():
                    continue

                segments.append({
                    "text": sentence,
                    "page": page_number,
                    "subject": current_subject,
                    "type": "content",
                    "extraction_method": extraction_method
                })

    logger.info(
        "Created %s text segments.",
        len(segments)
    )

    return segments


# --------------------------------------------------
# 15. DYNAMIC CHUNK SETTINGS
# --------------------------------------------------

def get_dynamic_chunk_settings(segments: list) -> dict:
    """
    Calculate chunk size from average segment length.
    """

    lengths = [
        len(segment["text"])
        for segment in segments
        if segment["text"].strip()
    ]

    if not lengths:

        return {
            "chunk_size": 850,
            "overlap_chars": 127
        }

    average_length = (
        sum(lengths) / len(lengths)
    )

    if average_length < 150:

        chunk_size = 700

    elif average_length < 350:

        chunk_size = 850

    elif average_length < 600:

        chunk_size = 1000

    else:

        chunk_size = 1200

    overlap_chars = int(
        chunk_size * 0.15
    )

    settings = {
        "chunk_size": chunk_size,
        "overlap_chars": overlap_chars
    }

    logger.info(
        "Dynamic chunk settings: %s",
        settings
    )

    return settings


# --------------------------------------------------
# 16. SPLIT LONG SEGMENTS
# --------------------------------------------------

def split_long_segment(
    text: str,
    max_size: int
) -> list:
    """
    Split a long segment at word boundaries.
    """

    if len(text) <= max_size:
        return [text]

    words = text.split()

    result = []

    current = []
    current_length = 0

    for word in words:

        required_length = len(word)

        if current:

            required_length += 1

        if (
            current
            and current_length + required_length > max_size
        ):

            result.append(
                " ".join(current)
            )

            current = []
            current_length = 0

        current.append(word)

        current_length += len(word) + 1

    if current:

        result.append(
            " ".join(current)
        )

    return result


# --------------------------------------------------
# 17. DOCUMENT-LEVEL CHUNKING
# --------------------------------------------------

def chunk_segments(
    segments: list,
    chunk_size: int,
    overlap_chars: int
) -> list:
    """
    Create chunks across the complete document.

    Chunks can span multiple pages, but they cannot
    mix different subjects.
    """

    chunks = []

    current = []
    current_length = 0
    current_subject = None

    def save_chunk(items):

        if not items:
            return

        pages = sorted({
            item["page"]
            for item in items
        })

        methods = sorted({
            item["extraction_method"]
            for item in items
        })

        chunks.append({
            "text": "\n".join(
                item["text"]
                for item in items
            ),
            "page_start": min(pages),
            "page_end": max(pages),
            "pages": pages,
            "subject": items[0].get("subject"),
            "extraction_method": (
                methods[0]
                if len(methods) == 1
                else "mixed"
            )
        })

    for segment in segments:

        text = segment["text"]

        subject = segment.get("subject")

        # Start a new chunk when the subject changes.
        if current and subject != current_subject:

            save_chunk(current)

            current = []
            current_length = 0

        current_subject = subject

        pieces = split_long_segment(
            text,
            chunk_size
        )

        for piece in pieces:

            required_length = len(piece) + 1

            # Check whether the piece fits.
            if (
                current
                and current_length + required_length > chunk_size
            ):

                previous = current.copy()

                save_chunk(previous)

                # Build overlap from complete content segments.
                overlap = []
                overlap_length = 0

                for item in reversed(previous):

                    if item.get("type") != "content":
                        continue

                    item_length = (
                        len(item["text"]) + 1
                    )

                    if (
                        overlap_length + item_length
                        > overlap_chars
                    ):
                        break

                    overlap.insert(0, item)

                    overlap_length += item_length

                current = overlap

                current_length = sum(
                    len(item["text"]) + 1
                    for item in current
                )

            current.append({
                **segment,
                "text": piece
            })

            current_length += required_length

    # Save the last chunk.
    if current:

        save_chunk(current)

    # Add unique chunk IDs.
    for chunk_id, chunk in enumerate(chunks):

        chunk["chunk_id"] = chunk_id

    logger.info(
        "Created %s final chunks.",
        len(chunks)
    )

    return chunks


# --------------------------------------------------
# 18. COMPLETE PDF PROCESSING PIPELINE
# --------------------------------------------------

def extract_and_chunk_pdf(pdf_path: str) -> list:
    """
    Complete pipeline:

    1. Extract PDF pages.
    2. Clean extracted text.
    3. Detect paragraphs and headings.
    4. Create segments.
    5. Calculate dynamic chunk settings.
    6. Create document-level chunks.
    """

    # Step 1: Extract pages.
    pages_data = smart_extract_pdf(
        pdf_path
    )

    # Step 2: Create document segments.
    segments = create_segments(
        pages_data
    )

    # Step 3: Calculate dynamic settings.
    settings = get_dynamic_chunk_settings(
        segments
    )

    chunk_size = settings["chunk_size"]

    overlap_chars = settings["overlap_chars"]

    # Step 4: Create final chunks.
    chunks = chunk_segments(
        segments=segments,
        chunk_size=chunk_size,
        overlap_chars=overlap_chars
    )

    # Step 5: Attach settings to every chunk.
    for chunk in chunks:

        chunk["chunk_size_setting"] = chunk_size

        chunk["overlap_setting"] = overlap_chars

    logger.info(
        "PDF processing completed. Pages: %s, Segments: %s, Chunks: %s",
        len(pages_data),
        len(segments),
        len(chunks)
    )

    return chunks
