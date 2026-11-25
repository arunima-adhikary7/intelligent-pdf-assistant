
import io
import logging
import re
from collections import Counter
from math import ceil
from typing import Any

import pdfplumber
import pymupdf
import pytesseract
from PIL import Image


# ==================================================
# 1. LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# 2. TEXT CLEANING
# ==================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by hyphen at line endings.
    text = re.sub(
        r"(?<=\w)-\n(?=[a-z])",
        "",
        text
    )

    lines = []

    for line in text.split("\n"):

        line = re.sub(r"[ \t]+", " ", line).strip()

        lines.append(line)

    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_cell(value: Any) -> str:

    if value is None:
        return ""

    return clean_text(
        str(value).replace("\n", " ")
    )


# ==================================================
# 3. HEADING DETECTION
# ==================================================

def is_heading(line: str) -> bool:

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
        r"^syllabus\s*:?",
    ]

    if any(
        re.match(pattern, line, re.IGNORECASE)
        for pattern in patterns
    ):
        return True

    return (
        len(line) < 90
        and line.isupper()
        and len(line.split()) <= 12
    )


def is_new_subject(line: str) -> bool:

    patterns = [
        r"^Introduction to Machine Learning$",
        r"^Operating Systems$",
        r"^Object Oriented Programming$",
        r"^Cloud Computing$",
        r"^Introduction to Industrial Management.*$",
    ]

    return any(
        re.match(
            pattern,
            line.strip(),
            re.IGNORECASE
        )
        for pattern in patterns
    )


# ==================================================
# 4. REMOVE REPEATED HEADERS AND FOOTERS
# ==================================================

def remove_repeated_headers(pages_data):

    if len(pages_data) < 2:
        return pages_data

    counts = Counter()

    for page_data in pages_data:

        lines = page_data["text"].splitlines()

        candidates = lines[:5] + lines[-3:]

        normalized = {
            re.sub(r"\s+", " ", line).strip().lower()
            for line in candidates
            if len(line.strip()) >= 10
        }

        counts.update(normalized)

    threshold = max(
        2,
        ceil(len(pages_data) * 0.4)
    )

    repeated = {
        line
        for line, count in counts.items()
        if count >= threshold
    }

    for page_data in pages_data:

        lines = page_data["text"].splitlines()

        boundary_indices = set(
            range(min(5, len(lines)))
        )

        if len(lines) > 8:

            boundary_indices.update(
                range(len(lines) - 3, len(lines))
            )

        cleaned = []

        for index, line in enumerate(lines):

            normalized = re.sub(
                r"\s+",
                " ",
                line
            ).strip().lower()

            if (
                index in boundary_indices
                and normalized in repeated
            ):
                continue

            cleaned.append(line)

        page_data["text"] = clean_text(
            "\n".join(cleaned)
        )

    return pages_data


# ==================================================
# 5. TABLE NORMALIZATION
# ==================================================

def normalize_table_rows(table):

    rows = []

    for row in table or []:

        normalized_row = [
            normalize_cell(cell)
            for cell in (row or [])
        ]

        if any(normalized_row):
            rows.append(normalized_row)

    if not rows:
        return []

    column_count = max(
        len(row) for row in rows
    )

    rows = [
        row + [""] * (column_count - len(row))
        for row in rows
    ]

    return rows


# ==================================================
# 6. SPECIAL HANDLING FOR TURNAROUND-TIME TABLE
# ==================================================

def normalize_turnaround_table(rows):

    if not rows:
        return rows

    header_text = " ".join(rows[0]).lower()

    is_turnaround_table = (
        "service" in header_text
        and (
            "turn around" in header_text
            or "turnaround" in header_text
            or "maximum" in header_text
        )
    )

    if not is_turnaround_table:
        return rows

    logger.info(
        "Detected policyholder servicing turnaround-time table."
    )

    normalized = [
        [
            "Service Category",
            "Service",
            "Maximum Turn Around Time"
        ]
    ]

    current_category = ""

    for row in rows[1:]:

        first = row[0].strip() if len(row) > 0 else ""
        second = row[1].strip() if len(row) > 1 else ""

        # A category row normally has a name in the
        # first column and an empty second column.
        if first and not second:

            current_category = first

            continue

        # A normal service row has service text and
        # turnaround time.
        if first or second:

            normalized.append([
                current_category,
                first,
                second
            ])

    return normalized


# ==================================================
# 7. TABLE CONVERSION TO MARKDOWN
# ==================================================

def table_to_markdown(table):

    rows = normalize_table_rows(table)

    if not rows:
        return ""

    rows = normalize_turnaround_table(rows)

    column_count = max(
        len(row) for row in rows
    )

    rows = [
        row + [""] * (column_count - len(row))
        for row in rows
    ]

    markdown_rows = []

    for index, row in enumerate(rows):

        markdown_rows.append(
            "| " + " | ".join(row) + " |"
        )

        if index == 0:

            markdown_rows.append(
                "| " + " | ".join(
                    ["---"] * column_count
                ) + " |"
            )

    return "\n".join(markdown_rows)


# ==================================================
# 8. TABLE EXTRACTION
# ==================================================

def score_extracted_tables(tables):

    total_score = 0

    for table in tables:

        if not table:
            continue

        rows = normalize_table_rows(table)

        if not rows:
            continue

        column_count = max(
            len(row) for row in rows
        )

        nonempty_cells = sum(
            bool(cell.strip())
            for row in rows
            for cell in row
        )

        paired_rows = sum(
            sum(bool(cell.strip()) for cell in row) >= 2
            for row in rows
        )

        score = (
            len(rows) * 2
            + nonempty_cells
            + paired_rows * 3
            + column_count * 5
        )

        # Give a preference to the target table
        # when its header is detected.
        header_text = " ".join(rows[0]).lower()

        if (
            "service" in header_text
            and (
                "turn around" in header_text
                or "turnaround" in header_text
                or "maximum" in header_text
            )
        ):
            score += 100

        total_score += score

    return total_score


def extract_tables_from_page(page):

    strategies = [

        # Strategy 1: Detect both vertical and horizontal lines.
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "edge_min_length": 10
        },

        # Strategy 2: Vertical lines and text-based rows.
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "min_words_horizontal": 1
        },

        # Strategy 3: Text-based columns and horizontal lines.
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "min_words_vertical": 2
        },

        # Strategy 4: Text-based table detection.
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "min_words_vertical": 2,
            "min_words_horizontal": 1
        },

        # Strategy 5: Default pdfplumber settings.
        None
    ]

    best_tables = []
    best_score = -1

    for strategy_index, settings in enumerate(strategies):

        try:

            if settings is None:

                tables = page.extract_tables()

            else:

                tables = page.extract_tables(
                    table_settings=settings
                )

            valid_tables = []

            for table in tables or []:

                rows = normalize_table_rows(table)

                if not rows:
                    continue

                # Ignore single-column extraction results.
                if len(rows[0]) < 2:
                    continue

                valid_tables.append(rows)

            strategy_score = score_extracted_tables(
                valid_tables
            )

            logger.info(
                "Table strategy %s: %s tables, score=%s",
                strategy_index + 1,
                len(valid_tables),
                strategy_score
            )

            if (
                valid_tables
                and strategy_score > best_score
            ):

                best_score = strategy_score
                best_tables = valid_tables

        except Exception:

            logger.exception(
                "Table extraction strategy %s failed.",
                strategy_index + 1
            )

    if best_tables:

        logger.info(
            "Selected table extraction with score %s",
            best_score
        )

    return best_tables


# ==================================================
# 9. EXTRACT TABLES FROM ALL PDF PAGES
# ==================================================

def extract_page_tables(pdf_path):

    page_tables = {}

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                tables = extract_tables_from_page(page)

                if not tables:
                    continue

                entries = []

                for index, table in enumerate(
                    tables,
                    start=1
                ):

                    markdown = table_to_markdown(table)

                    if not markdown:
                        continue

                    logger.info(
                        "PAGE %s TABLE %s:\n%s",
                        page_number,
                        index,
                        markdown
                    )

                    entries.append({

                        "table_id": index,

                        "text": markdown

                    })

                if entries:

                    page_tables[page_number] = entries

                    logger.info(
                        "Page %s: extracted %s table(s)",
                        page_number,
                        len(entries)
                    )

    except Exception:

        logger.exception(
            "Could not extract tables from PDF."
        )

    return page_tables


# ==================================================
# 10. OCR
# ==================================================

def perform_ocr(page):

    pix = page.get_pixmap(
        dpi=300,
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    image = Image.open(
        io.BytesIO(pix.tobytes("png"))
    )

    text = pytesseract.image_to_string(
        image,
        config="--psm 3"
    )

    return clean_text(text)


def needs_ocr(text):

    if not text or len(text.strip()) < 30:
        return True

    meaningful = sum(
        char.isalnum()
        for char in text
    )

    ratio = meaningful / max(len(text), 1)

    return ratio < 0.50


# ==================================================
# 11. SMART PDF EXTRACTION
# ==================================================

def smart_extract_pdf(pdf_path):

    pages_data = []

    # Extract tables separately using pdfplumber.
    page_tables = extract_page_tables(pdf_path)

    with pymupdf.open(pdf_path) as doc:

        for page_number, page in enumerate(
            doc,
            start=1
        ):

            tables = page_tables.get(
                page_number,
                []
            )

            # Extract complete digital page text.
            # Do not use mismatched table bounding boxes:
            # they can accidentally remove normal text.
            text = clean_text(
                page.get_text(
                    "text",
                    sort=True
                )
            )

            method = "digital"

            # OCR fallback for weak digital text.
            if needs_ocr(text):

                logger.info(
                    "Digital text is weak on page %s. Running OCR.",
                    page_number
                )

                try:

                    ocr_text = perform_ocr(page)

                    if len(ocr_text.strip()) > len(text.strip()):

                        text = ocr_text
                        method = "ocr"

                    else:

                        method = "digital_fallback"

                except Exception:

                    logger.exception(
                        "OCR failed on page %s",
                        page_number
                    )

                    method = "digital_fallback"

            pages_data.append({

                "page": page_number,

                "text": clean_text(text),

                "tables": tables,

                "extraction_method": method

            })

            logger.info(
                "Page %s: extracted %s characters using %s",
                page_number,
                len(text),
                method
            )

    return remove_repeated_headers(pages_data)


# ==================================================
# 12. EXTRACT PDF AS PLAIN TEXT
# ==================================================

def pages_to_text(pages_data):

    output = []

    for page_data in pages_data:

        page_number = page_data["page"]

        output.append(
            f"\n--- Page {page_number} ---"
        )

        output.append(
            page_data.get("text", "")
        )

        for table in page_data.get("tables", []):

            output.append(
                f"\n[TABLE {table['table_id']}]\n"
                f"{table['text']}"
            )

    return "\n".join(output)


def extract_pdf_text(pdf_path):

    pages_data = smart_extract_pdf(pdf_path)

    return pages_to_text(pages_data)


# ==================================================
# 13. FAQ-AWARE PARAGRAPH EXTRACTION
# ==================================================

FAQ_QUESTION_PATTERN = re.compile(
    r"^\s*(?:Q\s*\.|Question\s*[:.]?)\s*",
    re.IGNORECASE
)

FAQ_ANSWER_PATTERN = re.compile(
    r"^\s*(?:Ans\s*\.?|Answer\s*[:.]?)\s*",
    re.IGNORECASE
)

LIST_PATTERN = re.compile(
    r"^\s*(?:"
    r"\d+\s*[.)]"
    r"|[•●▪◦‣]"
    r"|[-–—]\s+"
    r")\s*"
)


def extract_paragraphs(text):

    blocks = []

    current = []
    faq_current = []
    list_current = []

    inside_faq = False
    inside_list = False

    def save_current():

        nonlocal current

        if current:

            block = "\n".join(current).strip()

            if block:
                blocks.append(block)

            current = []

    def save_list():

        nonlocal list_current, inside_list

        if list_current:

            block = "\n".join(list_current).strip()

            if block:
                blocks.append(block)

            list_current = []

        inside_list = False

    def save_faq():

        nonlocal faq_current, inside_faq

        if faq_current:

            block = "\n".join(faq_current).strip()

            if block:
                blocks.append(block)

            faq_current = []

        inside_faq = False

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:

            if inside_faq:

                faq_current.append("")

            else:

                save_list()
                save_current()

            continue

        # Start a new FAQ.
        if FAQ_QUESTION_PATTERN.match(line):

            save_faq()
            save_list()
            save_current()

            faq_current = [line]
            inside_faq = True

            continue

        # Close FAQ when a heading is detected.
        if inside_faq and is_heading(line):

            save_faq()

            blocks.append(line)

            continue

        # Keep the answer and its bullets together.
        if inside_faq:

            faq_current.append(line)

            continue

        # Handle headings.
        if is_heading(line):

            save_list()
            save_current()

            blocks.append(line)

            continue

        # Handle bullet and numbered lists.
        if LIST_PATTERN.match(line):

            save_current()

            inside_list = True

            list_current.append(line)

            continue

        # Preserve wrapped lines in a list.
        if inside_list:

            list_current.append(line)

            continue

        current.append(line)

    save_faq()
    save_list()
    save_current()

    return blocks


# ==================================================
# 14. SENTENCE SPLITTING
# ==================================================

def split_sentences(text):

    if not text.strip():
        return []

    protected = text

    tokens = {}

    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.",
        "Prof.", "Sr.", "Jr.",
        "e.g.", "i.e.", "etc.",
        "vs.", "No.", "Fig.", "Dept."
    ]

    def protect(pattern):

        nonlocal protected

        def replace_match(match):

            token = f"PROTECTEDTOKEN{len(tokens)}X"

            tokens[token] = match.group(0)

            return token

        protected = re.sub(
            pattern,
            replace_match,
            protected,
            flags=re.IGNORECASE
        )

    for abbreviation in abbreviations:

        protect(re.escape(abbreviation))

    protect(r"(?<=\d)\.(?=\d)")

    protect(r"\b[A-Z]\.(?=\s*[A-Z])")

    sentences = re.split(
        r"(?<=[.!?])\s+",
        protected
    )

    results = []

    for sentence in sentences:

        for token, original in tokens.items():

            sentence = sentence.replace(
                token,
                original
            )

        sentence = sentence.strip()

        if sentence:
            results.append(sentence)

    return results


# ==================================================
# 15. CREATE SEGMENTS
# ==================================================

def create_segments(pages_data):

    segments = []

    current_subject = None

    for page_data in pages_data:

        page_number = page_data["page"]

        extraction_method = page_data.get(
            "extraction_method",
            "unknown"
        )

        paragraphs = extract_paragraphs(
            page_data.get("text", "")
        )

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # Subject heading.
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

            # Normal heading.
            if is_heading(paragraph):

                segments.append({

                    "text": paragraph,

                    "page": page_number,

                    "subject": current_subject,

                    "type": "heading",

                    "extraction_method": extraction_method

                })

                continue

            # Keep FAQ question and answer together.
            if FAQ_QUESTION_PATTERN.match(paragraph):

                segments.append({

                    "text": paragraph,

                    "page": page_number,

                    "subject": current_subject,

                    "type": "faq",

                    "extraction_method": extraction_method

                })

                continue

            # Keep complete lists together.
            if LIST_PATTERN.match(paragraph):

                segments.append({

                    "text": paragraph,

                    "page": page_number,

                    "subject": current_subject,

                    "type": "list",

                    "extraction_method": extraction_method

                })

                continue

            # Split normal content into sentences.
            for sentence in split_sentences(paragraph):

                if sentence.strip():

                    segments.append({

                        "text": sentence,

                        "page": page_number,

                        "subject": current_subject,

                        "type": "content",

                        "extraction_method": extraction_method

                    })

        # Add extracted tables separately.
        for table in page_data.get("tables", []):

            table_text = table["text"].strip()

            if not table_text:
                continue

            segments.append({

                "text": (
                    f"Table {table['table_id']} "
                    f"on page {page_number}:\n"
                    f"{table_text}"
                ),

                "page": page_number,

                "subject": current_subject,

                "type": "table",

                "table_id": table["table_id"],

                "extraction_method": "pdfplumber_table"

            })

    logger.info(
        "Created %s segments",
        len(segments)
    )

    return segments


# ==================================================
# 16. DYNAMIC CHUNK SETTINGS
# ==================================================

def get_dynamic_chunk_settings(segments):

    lengths = [

        len(segment["text"])

        for segment in segments

        if segment.get("text", "").strip()

        and segment.get("type") != "table"

    ]

    if not lengths:

        chunk_size = 900

    else:

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

    return {

        "chunk_size": chunk_size,

        "overlap_chars": int(
            chunk_size * 0.15
        )

    }


# ==================================================
# 17. SPLIT LONG TEXT
# ==================================================

def split_long_segment(text, max_size):

    if len(text) <= max_size:
        return [text]

    words = text.split()

    result = []

    current = []

    current_length = 0

    for word in words:

        needed = len(word) + (
            1 if current else 0
        )

        if (
            current
            and current_length + needed > max_size
        ):

            result.append(
                " ".join(current)
            )

            current = []

            current_length = 0

        current.append(word)

        current_length += len(word) + (
            1 if len(current) > 1 else 0
        )

    if current:

        result.append(
            " ".join(current)
        )

    return result


# ==================================================
# 18. SPLIT LARGE TABLES BY ROWS
# ==================================================

def split_large_table(table_text, max_size):

    lines = table_text.splitlines()

    if len(table_text) <= max_size or len(lines) < 3:
        return [table_text]

    header = lines[0]

    separator = (
        lines[1]
        if lines[1].startswith("|")
        else ""
    )

    data_rows = (
        lines[2:]
        if separator
        else lines[1:]
    )

    prefix_lines = [header]

    if separator:
        prefix_lines.append(separator)

    prefix = "\n".join(prefix_lines)

    chunks = []

    current_rows = []

    for row in data_rows:

        candidate = "\n".join(
            [prefix] + current_rows + [row]
        )

        if (
            current_rows
            and len(candidate) > max_size
        ):

            chunks.append(
                "\n".join(
                    [prefix] + current_rows
                )
            )

            current_rows = [row]

        else:

            current_rows.append(row)

    if current_rows:

        chunks.append(
            "\n".join(
                [prefix] + current_rows
            )
        )

    return chunks


# ==================================================
# 19. FINAL CHUNK CREATION
# ==================================================

def chunk_segments(
    segments,
    chunk_size,
    overlap_chars
):

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
            item.get(
                "extraction_method",
                "unknown"
            )
            for item in items
        })

        types = {
            item.get("type", "content")
            for item in items
        }

        if "faq" in types:

            content_type = "faq"

        elif "list" in types:

            content_type = "list"

        elif types == {"table"}:

            content_type = "table"

        else:

            content_type = "text"

        chunks.append({

            "text": "\n".join(
                item["text"]
                for item in items
            ),

            "page_start": min(pages),

            "page_end": max(pages),

            "pages": pages,

            "subject": items[0].get("subject"),

            "content_type": content_type,

            "table_id": (
                items[0].get("table_id")
                if content_type == "table"
                else None
            ),

            "extraction_method": (
                methods[0]
                if len(methods) == 1
                else "mixed"
            )

        })

    def make_overlap(previous):

        overlap = []

        overlap_length = 0

        for item in reversed(previous):

            if item.get("type") != "content":
                continue

            item_length = len(item["text"]) + 1

            if (
                overlap_length + item_length
                > overlap_chars
            ):
                break

            overlap.insert(0, item)

            overlap_length += item_length

        return overlap

    for segment in segments:

        subject = segment.get("subject")

        segment_type = segment.get(
            "type",
            "content"
        )

        # Start a new chunk when subject changes.
        if current and subject != current_subject:

            save_chunk(current)

            current = []

            current_length = 0

        current_subject = subject

        # Every FAQ gets its own chunk.
        if segment_type == "faq":

            if current:

                save_chunk(current)

                current = []

                current_length = 0

            save_chunk([segment])

            continue

        # Keep lists independent.
        if segment_type == "list":

            if current:

                save_chunk(current)

                current = []

                current_length = 0

            save_chunk([segment])

            continue

        # Keep tables independent.
        if segment_type == "table":

            if current:

                save_chunk(current)

                current = []

                current_length = 0

            table_pieces = split_large_table(
                segment["text"],
                chunk_size
            )

            for piece in table_pieces:

                save_chunk([{
                    **segment,
                    "text": piece
                }])

            continue

        pieces = split_long_segment(
            segment["text"],
            chunk_size
        )

        for piece in pieces:

            needed = len(piece) + (
                1 if current else 0
            )

            if (
                current
                and current_length + needed > chunk_size
            ):

                previous = current.copy()

                save_chunk(previous)

                current = make_overlap(previous)

                current_length = sum(
                    len(item["text"]) + 1
                    for item in current
                )

            current.append({

                **segment,

                "text": piece

            })

            current_length += len(piece) + 1

    if current:

        save_chunk(current)

    # Assign unique chunk IDs.
    for chunk_id, chunk in enumerate(chunks):

        chunk["chunk_id"] = chunk_id

    logger.info(
        "Created %s final chunks",
        len(chunks)
    )

    return chunks


# ==================================================
# 20. COMPLETE PDF PROCESSING PIPELINE
# ==================================================

def extract_and_chunk_pdf(pdf_path):

    logger.info(
        "Starting PDF processing: %s",
        pdf_path
    )

    # Step 1: Extract digital text, OCR text and tables.
    pages_data = smart_extract_pdf(pdf_path)

    # Step 2: Create FAQ, list, text and table segments.
    segments = create_segments(pages_data)

    # Step 3: Calculate chunk settings.
    settings = get_dynamic_chunk_settings(
        segments
    )

    # Step 4: Create final chunks.
    chunks = chunk_segments(

        segments=segments,

        chunk_size=settings["chunk_size"],

        overlap_chars=settings["overlap_chars"]

    )

    # Step 5: Add chunk configuration metadata.
    for chunk in chunks:

        chunk["chunk_size_setting"] = (
            settings["chunk_size"]
        )

        chunk["overlap_setting"] = (
            settings["overlap_chars"]
        )

    logger.info(
        "PDF processing completed. "
        "Pages: %s, Segments: %s, Chunks: %s",
        len(pages_data),
        len(segments),
        len(chunks)
    )

    return chunks
