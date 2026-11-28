import io
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import pymupdf
import pytesseract
from PIL import Image, ImageOps


# ============================================================
# 1. CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)

# These are operational defaults, not document-specific rules.
# Override them through environment variables or function arguments
# if you later want to tune the pipeline without editing the code.
DEFAULT_CHUNK_SIZE = 1400
MIN_CHUNK_SIZE = 900
MAX_CHUNK_SIZE = 1800
DEFAULT_OVERLAP = 350

OCR_DPI = 300
GUTTER_RENDER_DPI = 120

PAGE_LAYOUT = "auto"  # "auto", "two_page_spread", or "single"

GUTTER_SEARCH_LEFT_RATIO = 0.30
GUTTER_SEARCH_RIGHT_RATIO = 0.70
MIN_GUTTER_WIDTH_PIXELS = 8
GUTTER_PADDING_POINTS = 3

MIN_WORDS_PER_SIDE = 10
MIN_CHARACTERS_PER_SIDE = 60

DARK_PIXEL_THRESHOLD = 210
MAX_GUTTER_INK_RATIO = 0.025

OCR_CONFIG = "--psm 3"


# ============================================================
# 2. TEXT PRESERVATION AND COMPARISON
# ============================================================

def preserve_text(text: Any) -> str:
    """Preserve extracted wording, punctuation, spaces and line breaks."""
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def normalize_for_comparison(text: str) -> str:
    """Normalize only for detection; never use this for stored source text."""
    return re.sub(r"\s+", " ", preserve_text(text).lower()).strip()


def nonempty_lines(text: str):
    """Return non-empty lines without changing their original contents."""
    return [
        line for line in preserve_text(text).splitlines()
        if line.strip()
    ]


# ============================================================
# 3. QUESTION / ANSWER DETECTION
# ============================================================

EXPLICIT_QUESTION_PATTERN = re.compile(
    r"^\s*(?:"
    r"Q\s*\.?\s*\d*\s*[:.)-]?"
    r"|Question\s*\d*\s*[:.)-]"
    r")\s*",
    re.IGNORECASE,
)

QUESTION_START_PATTERN = re.compile(
    r"^\s*(?:"
    r"what|why|when|where|who|whom|whose|which|how|"
    r"can|could|would|should|is|are|am|do|does|did|"
    r"will|shall|has|have|had|if"
    r")\b",
    re.IGNORECASE,
)

INTERROGATIVE_WORD_PATTERN = re.compile(
    r"\b(?:what|why|when|where|who|whom|whose|which|how)\b",
    re.IGNORECASE,
)

ANSWER_LABEL_PATTERN = re.compile(
    r"^\s*(?:Ans(?:wer)?(?:\s*\d+)?\s*[.:)-]?)\s*",
    re.IGNORECASE,
)


def is_question_line(line: str) -> bool:
    """
    Detect explicit and natural-language questions.

    A wrapped 'If ... what ...' question is detected even when its
    first extracted line does not yet contain the question mark.
    """
    line = line.strip()
    if not line:
        return False

    if EXPLICIT_QUESTION_PATTERN.match(line):
        return True

    if not QUESTION_START_PATTERN.match(line):
        return False

    if "?" in line:
        return True

    # A conditional question can wrap before its question mark:
    # "If I use the car in another city, what"
    # "premium rate is applied?"
    if re.match(r"^\s*if\b", line, re.IGNORECASE):
        return bool(INTERROGATIVE_WORD_PATTERN.search(line))

    return False


def is_answer_line(line: str) -> bool:
    return bool(ANSWER_LABEL_PATTERN.match(line.strip()))


def is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    patterns = [
        r"^(unit|module|chapter|section|part)\s+[\w.-]+",
        r"^(subject|course|paper|topic)\s*:",
        r"^(references|textbooks|contents|syllabus)\s*:?",
        r"^(appendix|bibliography|acknowledgements?)\s*:?",
    ]

    if any(re.match(pattern, line, re.IGNORECASE) for pattern in patterns):
        return True

    return (
        len(line) <= 80
        and line.isupper()
        and len(line.split()) <= 10
    )


# ============================================================
# 4. DETECT A TWO-PAGE SPREAD / CENTRAL GUTTER
# ============================================================

def _merge_intervals(intervals):
    """Merge overlapping horizontal word-coordinate intervals."""
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda item: item[0])
    merged = [[intervals[0][0], intervals[0][1]]]

    for x0, x1 in intervals[1:]:
        if x0 > merged[-1][1]:
            merged.append([x0, x1])
        else:
            merged[-1][1] = max(merged[-1][1], x1)

    return merged


def detect_gutter_from_words(page) -> Optional[float]:
    """Find a central blank gap between selectable text words."""
    page_width = float(page.rect.width)
    words = page.get_text("words")

    if not words:
        return None

    intervals = []
    for word in words:
        x0, _y0, x1, _y1 = word[:4]
        if str(word[4]).strip():
            intervals.append((float(x0), float(x1)))

    merged = _merge_intervals(intervals)
    if len(merged) < 2:
        return None

    search_left = page_width * GUTTER_SEARCH_LEFT_RATIO
    search_right = page_width * GUTTER_SEARCH_RIGHT_RATIO

    candidates = []
    for left, right in zip(merged, merged[1:]):
        gap_start = left[1]
        gap_end = right[0]
        gap_width = gap_end - gap_start
        gap_middle = (gap_start + gap_end) / 2

        if gap_width > 0 and search_left <= gap_middle <= search_right:
            candidates.append((gap_width, gap_middle))

    if not candidates:
        return None

    gap_width, split_x = max(candidates, key=lambda item: item[0])

    logger.debug(
        "Word gutter candidate | x=%.2f | width=%.2f points",
        split_x,
        gap_width,
    )
    return split_x


def detect_gutter_from_image(page) -> Optional[float]:
    """Find a vertical low-ink region for scanned pages."""
    pix = page.get_pixmap(
        dpi=GUTTER_RENDER_DPI,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    width, height = image.size

    x_start = int(width * GUTTER_SEARCH_LEFT_RATIO)
    x_end = int(width * GUTTER_SEARCH_RIGHT_RATIO)

    if x_end <= x_start:
        return None

    pixels = image.load()
    ink_ratios = []

    for x in range(x_start, x_end):
        dark_count = sum(
            1 for y in range(height)
            if pixels[x, y] < DARK_PIXEL_THRESHOLD
        )
        ink_ratios.append(dark_count / max(height, 1))

    runs = []
    run_start = None

    for index, ratio in enumerate(ink_ratios):
        if ratio <= MAX_GUTTER_INK_RATIO:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(ink_ratios) - 1))

    valid_runs = [
        (start, end)
        for start, end in runs
        if end - start + 1 >= MIN_GUTTER_WIDTH_PIXELS
    ]

    if not valid_runs:
        return None

    start, end = max(
        valid_runs,
        key=lambda item: item[1] - item[0] + 1,
    )

    gutter_pixel_x = x_start + (start + end) / 2
    split_x = gutter_pixel_x * page.rect.width / width

    logger.debug(
        "Image gutter candidate | x=%.2f PDF points",
        split_x,
    )
    return split_x


def detect_spread_split_x(page) -> Optional[float]:
    """Try selectable text first, then image analysis."""
    split_x = detect_gutter_from_words(page)
    if split_x is not None:
        return split_x
    return detect_gutter_from_image(page)


def detect_two_page_spread(page, split_x: Optional[float] = None) -> bool:
    """Check that there is meaningful content on both sides of a gutter."""
    if split_x is None:
        split_x = detect_spread_split_x(page)

    if split_x is None:
        return False

    words = page.get_text("words")

    if words:
        left_words = [word for word in words if float(word[2]) <= split_x]
        right_words = [word for word in words if float(word[0]) >= split_x]

        left_chars = sum(len(str(word[4])) for word in left_words)
        right_chars = sum(len(str(word[4])) for word in right_words)

        return (
            len(left_words) >= MIN_WORDS_PER_SIDE
            and len(right_words) >= MIN_WORDS_PER_SIDE
            and left_chars >= MIN_CHARACTERS_PER_SIDE
            and right_chars >= MIN_CHARACTERS_PER_SIDE
        )

    pix = page.get_pixmap(
        dpi=GUTTER_RENDER_DPI,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    width, height = image.size
    split_pixel = int(split_x * width / page.rect.width)

    left_crop = image.crop((0, 0, split_pixel, height))
    right_crop = image.crop((split_pixel, 0, width, height))

    def ink_ratio(crop):
        histogram = crop.histogram()
        dark_pixels = sum(histogram[:DARK_PIXEL_THRESHOLD])
        return dark_pixels / max(crop.width * crop.height, 1)

    return (
        ink_ratio(left_crop) > 0
        and ink_ratio(right_crop) > 0
    )


def should_split_page(page, split_x: Optional[float] = None) -> bool:
    if PAGE_LAYOUT == "two_page_spread":
        return True
    if PAGE_LAYOUT == "single":
        return False
    if PAGE_LAYOUT == "auto":
        return detect_two_page_spread(page, split_x)

    raise ValueError(
        "PAGE_LAYOUT must be 'auto', 'two_page_spread' or 'single'."
    )


# ============================================================
# 5. EXTRACT ONE PRINTED PAGE AT A TIME
# ============================================================

def get_side_clip(page, side: str, split_x: float):
    """Create a crop rectangle that excludes the central gutter."""
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    if side == "left":
        return pymupdf.Rect(
            0,
            0,
            max(0, split_x - GUTTER_PADDING_POINTS),
            page_height,
        )

    if side == "right":
        return pymupdf.Rect(
            min(page_width, split_x + GUTTER_PADDING_POINTS),
            0,
            page_width,
            page_height,
        )

    raise ValueError("side must be 'left' or 'right'.")


def extract_spread_side(page, side: str, split_x: float):
    """Extract selectable text only from the selected printed-page crop."""
    clip = get_side_clip(page, side, split_x)
    text = page.get_text("text", clip=clip, sort=True)

    return {
        "text": preserve_text(text),
        "clip": clip,
        "split_x": split_x,
    }


# ============================================================
# 6. OCR AND EXTRACTION QUALITY
# ============================================================

def perform_ocr(page, clip=None) -> str:
    """OCR either the complete PDF page or one cropped printed page."""
    pixmap = page.get_pixmap(
        dpi=OCR_DPI,
        colorspace=pymupdf.csRGB,
        alpha=False,
        clip=clip,
    )

    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
        image = ImageOps.grayscale(image)
        text = pytesseract.image_to_string(
            image,
            config=OCR_CONFIG,
        )

    return preserve_text(text)


def text_quality(text: str) -> int:
    return sum(
        character.isalnum()
        for character in preserve_text(text)
    )


def needs_ocr(text: str) -> bool:
    text = preserve_text(text)
    meaningful = text_quality(text)

    if meaningful == 0:
        return True

    meaningful_ratio = meaningful / max(len(text), 1)
    return meaningful_ratio < 0.35


def choose_best_extraction(page, digital_text: str, clip=None):
    """Use OCR only when digital extraction appears weak."""
    if not needs_ocr(digital_text):
        return digital_text, "digital"

    logger.info("Weak digital extraction; running OCR on selected region.")

    try:
        ocr_text = perform_ocr(page, clip=clip)

        if text_quality(ocr_text) > text_quality(digital_text):
            return ocr_text, "ocr"

        return digital_text, "digital_fallback"

    except Exception:
        logger.exception("OCR failed.")
        return digital_text, "digital_fallback"


# ============================================================
# 7. EXTRACT ALL PHYSICAL PDF PAGES
# ============================================================

def smart_extract_pdf(pdf_path):
    """
    Extract each physical PDF page.

    If a spread is detected, crop left and right independently.
    Each printed page receives its own logical page number.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages_data = []
    logical_page_number = 0

    with pymupdf.open(pdf_path) as document:
        total_physical_pages = len(document)

        logger.info(
            "Starting extraction | physical pages=%s",
            total_physical_pages,
        )

        for physical_page_number, page in enumerate(document, start=1):
            candidate_split_x = detect_spread_split_x(page)
            split_page = should_split_page(page, candidate_split_x)
            split_x = candidate_split_x if split_page else None

            if split_page and split_x is None:
                if PAGE_LAYOUT == "two_page_spread":
                    split_x = float(page.rect.width) / 2
                    logger.warning(
                        "No gutter found on physical page %s; using midpoint.",
                        physical_page_number,
                    )
                else:
                    split_page = False

            sides = ["left", "right"] if split_page else ["full"]

            logger.info(
                "Physical page %s | layout=%s | split_x=%s",
                physical_page_number,
                "spread" if split_page else "single",
                split_x,
            )

            for side in sides:
                if side == "full":
                    clip = None
                    digital_text = preserve_text(
                        page.get_text("text", sort=True)
                    )
                else:
                    side_result = extract_spread_side(
                        page,
                        side,
                        split_x,
                    )
                    clip = side_result["clip"]
                    digital_text = side_result["text"]

                extracted_text, method = choose_best_extraction(
                    page,
                    digital_text,
                    clip,
                )

                logical_page_number += 1

                page_data = {
                    "page": logical_page_number,
                    "physical_page": physical_page_number,
                    "spread_side": side,
                    "total_pages": 0,
                    "total_physical_pages": total_physical_pages,
                    "text": extracted_text,
                    "body": extracted_text,
                    "tables": [],
                    "extraction_method": method,
                    "is_spread": split_page,
                    "split_x": split_x,
                }
                pages_data.append(page_data)

                logger.info(
                    "\n========== EXTRACTED PRINTED PAGE ==========\n"
                    "Logical page: %s\n"
                    "Physical PDF page: %s\n"
                    "Side: %s\n"
                    "Method: %s\n"
                    "Characters: %s\n"
                    "Text:\n%s\n"
                    "============================================",
                    logical_page_number,
                    physical_page_number,
                    side,
                    method,
                    len(extracted_text),
                    extracted_text,
                )

        for item in pages_data:
            item["total_pages"] = len(pages_data)

    logger.info(
        "Extraction complete | physical=%s | logical=%s",
        total_physical_pages,
        len(pages_data),
    )
    return pages_data


# ============================================================
# 8. PAGE EXTRACTION HELPERS
# ============================================================

def extract_pdf_pages(pdf_path):
    pages_data = smart_extract_pdf(pdf_path)

    return [
        {
            "page": page["page"],
            "physical_page": page["physical_page"],
            "spread_side": page["spread_side"],
            "total_pages": page["total_pages"],
            "text": page["text"],
            "body": page["body"],
            "tables": page["tables"],
            "extraction_method": page["extraction_method"],
            "is_spread": page["is_spread"],
            "split_x": page["split_x"],
        }
        for page in pages_data
    ]


def pages_to_text(pages_data):
    output = []

    for page in pages_data:
        output.append(
            f"\n========== LOGICAL PAGE {page['page']} "
            f"| PDF PAGE {page['physical_page']} "
            f"| {page['spread_side']} =========="
        )
        output.append(preserve_text(page.get("text", "")))

    return "\n".join(output)


def extract_pdf_text(pdf_path):
    return pages_to_text(smart_extract_pdf(pdf_path))


# ============================================================
# 9. SPLIT LONG PAGE CONTENT
# ============================================================

def split_text_preserving_lines(
    text: str,
    max_size: int = DEFAULT_CHUNK_SIZE,
    overlap_chars: int = DEFAULT_OVERLAP,
):
    """Split at line boundaries and keep each chunk on one logical page."""
    text = preserve_text(text)

    if not text.strip():
        return []

    if len(text) <= max_size:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:
        if current_lines and current_length + len(line) > max_size:
            chunks.append("".join(current_lines))

            overlap_lines = []
            overlap_length = 0

            for previous_line in reversed(current_lines):
                if overlap_length + len(previous_line) > overlap_chars:
                    break
                overlap_lines.insert(0, previous_line)
                overlap_length += len(previous_line)

            current_lines = overlap_lines
            current_length = overlap_length

        current_lines.append(line)
        current_length += len(line)

    if current_lines:
        final_chunk = "".join(current_lines)
        if final_chunk.strip():
            chunks.append(final_chunk)

    return chunks


# ============================================================
# 10. BUILD QUESTION-ANSWER GROUPS
# ============================================================

def build_question_answer_groups(page_text: str):
    """
    Group each detected question with its answer until the next question.
    Text before the first detected question is not treated as a Q&A record.
    """
    lines = preserve_text(page_text).splitlines(keepends=True)

    groups = []
    current_group = []
    current_has_question = False

    def save_group():
        nonlocal current_group, current_has_question

        if not current_group:
            return

        group_text = "".join(current_group)

        if current_has_question and group_text.strip():
            groups.append({
                "text": group_text,
                "has_answer_label": any(
                    is_answer_line(line)
                    for line in group_text.splitlines()
                ),
            })

        current_group = []
        current_has_question = False

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            if current_has_question:
                current_group.append(raw_line)
            continue

        if is_question_line(line):
            save_group()
            current_group = [raw_line]
            current_has_question = True
            continue

        if current_has_question:
            current_group.append(raw_line)

    save_group()
    return groups


def get_first_nonempty_line(text: str) -> str:
    for line in preserve_text(text).splitlines():
        if line.strip():
            return line.strip()
    return ""


# ============================================================
# 11. DYNAMIC ANSWER-CONTINUATION DETECTION
# ============================================================
def get_answer_body(group_text: str) -> str:
    """
    Extract the answer body while ignoring trailing printed page numbers.

    Original answer wording and line breaks are preserved.
    """
    lines = preserve_text(group_text).splitlines(keepends=True)

    answer_started = False
    answer_lines = []

    for line in lines:

        if not answer_started:

            if is_answer_line(line):
                answer_started = True

                answer_content = ANSWER_LABEL_PATTERN.sub(
                    "",
                    line,
                    count=1,
                )

                if answer_content.strip():
                    answer_lines.append(answer_content)

            continue

        answer_lines.append(line)

    answer_text = "".join(answer_lines)

    # Remove printed page number and trailing whitespace.
    answer_text = remove_trailing_page_number(answer_text)

    return answer_text.strip()

def get_last_nonempty_line(text: str) -> str:
    for line in reversed(preserve_text(text).splitlines()):
        if line.strip():
            return line.strip()
    return ""


def answer_ends_with_continuation_cue(answer_text: str) -> bool:
    """
    Detect an unfinished answer from its ending, not its character count.

    Examples of continuation cues:
      '... applicable for the'
      '... is based on'
      '... and'
      '... in accordance with'

    A final sentence-ending punctuation mark normally means the answer
    is complete. This is a structural heuristic, not a language model.
    """
    last_line = get_last_nonempty_line(answer_text)
    if not last_line:
        return False

    # Remove trailing closing quotes/brackets before checking punctuation.
    ending = last_line.rstrip().rstrip('"\u201d\u2019)]}')
    if not ending:
        return False

    if ending.endswith((".", "?", "!", ";")):
        return False

    # A dangling hyphen or dash commonly indicates a split word/phrase.
    if ending.endswith(("-", "\u2013", "\u2014", "/", ",")):
        return True

    last_token_match = re.search(r"([A-Za-z][A-Za-z'-]*)$", ending)
    if not last_token_match:
        return False

    last_token = last_token_match.group(1).lower()

    # Grammatical function words at the end strongly suggest continuation.
    continuation_words = {
        "a", "an", "the", "of", "for", "to", "from", "with", "without",
        "in", "on", "at", "by", "as", "into", "onto", "over", "under",
        "between", "among", "and", "or", "but", "because", "although",
        "if", "when", "while", "that", "which", "who", "whose",
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may",
        "might", "must", "not", "than", "such",
    }

    return last_token in continuation_words


def next_page_looks_like_continuation(first_line: str) -> bool:
    """Reject a new question, answer label, or obvious heading."""
    line = first_line.strip()

    if not line:
        return False

    if is_question_line(line) or is_answer_line(line) or is_heading(line):
        return False

    # A lowercase first character is a strong sign that a sentence
    # started on the previous printed page.
    if line[:1].islower():
        return True

    # Continuation can begin with punctuation or a joining word.
    if re.match(r"^[,;:.)\]}]", line):
        return True

    continuation_starters = {
        "and", "or", "but", "because", "which", "that", "who",
        "where", "when", "while", "with", "without", "of", "for",
        "to", "from", "in", "on", "at", "by", "as",
    }
    first_token = re.match(r"[A-Za-z]+", line)
    if first_token and first_token.group(0).lower() in continuation_starters:
        return True

    return False


def remove_trailing_page_number(text: str) -> str:
    """Remove standalone numeric page markers from the end only."""
    lines = preserve_text(text).splitlines(keepends=True)

    while lines:
        last_line = lines[-1].strip()

        if not last_line:
            lines.pop()
            continue

        if re.fullmatch(r"\d{1,4}", last_line):
            lines.pop()
            continue

        break

    return "".join(lines).rstrip()


def get_continuation_until_next_question(page_text: str):
    """
    Collect continuation text from the beginning of the next printed page.
    Stop before the next question so that a new Q&A is never absorbed.
    """
    lines = preserve_text(page_text).splitlines(keepends=True)
    continuation_lines = []
    found_next_question = False

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            if continuation_lines:
                continuation_lines.append(raw_line)
            continue

        if is_question_line(line):
            found_next_question = True
            break

        # A new answer label without a preceding question is not assumed
        # to belong to the previous answer.
        if is_answer_line(line) and continuation_lines:
            break

        continuation_lines.append(raw_line)

    continuation_text = remove_trailing_page_number(
        "".join(continuation_lines)
    )

    return continuation_text, found_next_question

def should_join_answer_to_next_page(
    current_group: dict,
    next_page_text: str,
) -> bool:
    """
    Decide whether the final answer continues onto the next
    logical printed page.

    Uses:
      1. Presence of an answer label.
      2. Answer ending structure.
      3. First meaningful line of the next page.
    """

    if not current_group.get("has_answer_label"):
        return False

    # Extract answer without printed page-number interference.
    answer_body = get_answer_body(
        current_group["text"]
    )

    if not answer_body.strip():
        return False

    # Remove standalone page numbers and inspect the next page.
    cleaned_next_page = preserve_text(next_page_text)

    first_line = get_first_nonempty_line(
        cleaned_next_page
    )

    if not first_line:
        return False

    # Next page must look like continuation text.
    if not next_page_looks_like_continuation(first_line):
        return False

    # Previous answer must look unfinished.
    if not answer_ends_with_continuation_cue(answer_body):
        return False

    logger.info(
        "ANSWER CONTINUATION DETECTED | "
        "Previous ending: %s | Next beginning: %s",
        get_last_nonempty_line(answer_body),
        first_line,
    )

    return True
# ============================================================
# 12. JOIN Q&A CONTINUATIONS ACROSS PRINTED PAGES
# ============================================================

def join_cross_page_answers(pages_data):
    """
    Join an incomplete final Q&A with continuation text from the next
    logical printed page.

    Handles left-to-right spread continuation and adjacent physical-page
    continuation. It never copies the next question into the previous Q&A.
    """
    page_groups = [
        build_question_answer_groups(page.get("text", ""))
        for page in pages_data
    ]

    completed_records = []
    consumed_continuation_pages = set()

    for page_index, page_data in enumerate(pages_data):
        groups = page_groups[page_index]

        for group_index, group in enumerate(groups):
            # A page used only as a continuation source must not generate
            # a duplicate Q&A record from the same continuation lines.
            record_text = group["text"]
            page_start = page_data["page"]
            page_end = page_start
            pages_in_record = [page_start]

            is_last_group = group_index == len(groups) - 1

            if (
                is_last_group
                and page_index + 1 < len(pages_data)
            ):
                next_page = pages_data[page_index + 1]

                same_physical_sheet = (
                    page_data["physical_page"]
                    == next_page["physical_page"]
                )

                valid_spread_order = (
                    same_physical_sheet
                    and page_data["spread_side"] == "left"
                    and next_page["spread_side"] == "right"
                )

                adjacent_logical_page = (
                    next_page["page"] == page_data["page"] + 1
                )

                if (
                    adjacent_logical_page
                    and (valid_spread_order or not same_physical_sheet)
                    and should_join_answer_to_next_page(
                        group,
                        next_page.get("text", ""),
                    )
                ):
                    continuation_text, _ = (
                        get_continuation_until_next_question(
                            next_page.get("text", "")
                        )
                    )

                    if continuation_text.strip():
                        record_text = remove_trailing_page_number(record_text)

                        if not record_text.endswith("\n"):
                            record_text += "\n"

                        record_text += continuation_text

                        page_end = next_page["page"]
                        pages_in_record.append(next_page["page"])
                        consumed_continuation_pages.add(next_page["page"])

                        logger.info(
                            "ANSWER CONTINUATION JOINED | "
                            "physical page=%s -> %s | logical pages=%s-%s | "
                            "answer chars=%s | continuation chars=%s | "
                            "final chars=%s",
                            page_data["physical_page"],
                            next_page["physical_page"],
                            page_start,
                            page_end,
                            len(get_answer_body(group["text"])),
                            len(continuation_text),
                            len(record_text),
                        )

            completed_records.append({
                "text": record_text,
                "page_start": page_start,
                "page_end": page_end,
                "pages": pages_in_record,
                "is_complete_qa": bool(
                    group.get("has_answer_label")
                ),
            })

    return completed_records


# ============================================================
# 13. CREATE PAGE-CONTENT RECORDS
# ============================================================

def create_page_content_records(
    pages_data,
    chunk_size,
    overlap_chars,
):
    records = []

    for page_data in pages_data:
        page_number = page_data["page"]
        page_text = preserve_text(page_data.get("text", ""))

        if not page_text.strip():
            continue

        pieces = split_text_preserving_lines(
            page_text,
            max_size=chunk_size,
            overlap_chars=overlap_chars,
        )

        for piece_number, piece in enumerate(pieces, start=1):
            if not piece.strip():
                continue

            records.append({
                "text": piece,
                "original_text": piece,
                "page_start": page_number,
                "page_end": page_number,
                "pages": [page_number],
                "physical_page": page_data["physical_page"],
                "spread_side": page_data["spread_side"],
                "content_type": "page_content",
                "record_type": "page_chunk",
                "piece_number": piece_number,
                "subject": None,
                "logical_unit_id": None,
                "continuation_part": None,
                "is_continuation": False,
                "continuation_of": None,
                "extraction_method": page_data.get(
                    "extraction_method",
                    "unknown",
                ),
            })

    return records


# ============================================================
# 14. CREATE QUESTION-ANSWER RECORDS
# ============================================================

def create_question_answer_records(pages_data):
    records = []
    question_groups = join_cross_page_answers(pages_data)

    for question_number, group in enumerate(question_groups, start=1):
        group_text = group["text"]

        if not group_text.strip():
            continue

        matching_pages = [
            page for page in pages_data
            if page["page"] in group["pages"]
        ]

        extraction_methods = list(dict.fromkeys(
            page["extraction_method"]
            for page in matching_pages
        ))

        physical_pages = list(dict.fromkeys(
            page["physical_page"]
            for page in matching_pages
        ))

        spread_sides = list(dict.fromkeys(
            page["spread_side"]
            for page in matching_pages
        ))

        records.append({
            "text": group_text,
            "original_text": group_text,
            "page_start": group["page_start"],
            "page_end": group["page_end"],
            "pages": group["pages"],
            "physical_pages": physical_pages,
            "spread_sides": spread_sides,
            "content_type": "question_answer",
            "record_type": "question_answer",
            "piece_number": question_number,
            "subject": None,
            "logical_unit_id": f"qa_{uuid.uuid4().hex}",
            "continuation_part": 1,
            "is_continuation": (
                group["page_start"] != group["page_end"]
            ),
            "continuation_of": None,
            "is_complete_qa": group["is_complete_qa"],
            "extraction_method": ", ".join(extraction_methods),
        })

        logger.info(
            "Q&A record created | pages=%s-%s | characters=%s",
            group["page_start"],
            group["page_end"],
            len(group_text),
        )

    return records


# ============================================================
# 15. DYNAMIC CHUNK SETTINGS
# ============================================================

def get_dynamic_chunk_settings(pages_data):
    """Choose chunk size from the observed logical-page text lengths."""
    page_lengths = [
        len(page.get("text", ""))
        for page in pages_data
        if page.get("text", "").strip()
    ]

    if not page_lengths:
        chunk_size = DEFAULT_CHUNK_SIZE
    else:
        average_length = sum(page_lengths) / len(page_lengths)
        chunk_size = int(
            min(
                MAX_CHUNK_SIZE,
                max(MIN_CHUNK_SIZE, average_length / 2),
            )
        )

    overlap_chars = min(
        DEFAULT_OVERLAP,
        int(chunk_size * 0.15),
    )

    return {
        "chunk_size": chunk_size,
        "overlap_chars": overlap_chars,
    }


# ============================================================
# 16. COMPLETE PDF PROCESSING PIPELINE
# ============================================================

def extract_and_chunk_pdf(pdf_path):
    """
    Full pipeline:
      1. Extract physical PDF pages.
      2. Detect side-by-side printed pages.
      3. Crop each printed page before extraction/OCR.
      4. Keep each printed page independent.
      5. Create page-content chunks.
      6. Create Q&A records.
      7. Join only structurally incomplete answers.
      8. Return Qdrant-ready records.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Starting PDF processing: %s", pdf_path)

    pages_data = smart_extract_pdf(pdf_path)

    if not pages_data:
        raise RuntimeError("No pages extracted from PDF.")

    settings = get_dynamic_chunk_settings(pages_data)
    chunk_size = settings["chunk_size"]
    overlap_chars = settings["overlap_chars"]

    page_records = create_page_content_records(
        pages_data,
        chunk_size,
        overlap_chars,
    )

    qa_records = create_question_answer_records(pages_data)
    all_records = page_records + qa_records

    for chunk_id, record in enumerate(all_records):
        record["chunk_id"] = chunk_id
        record["chunk_size_setting"] = chunk_size
        record["overlap_setting"] = overlap_chars

    for record in page_records:
        if record["page_start"] != record["page_end"]:
            raise RuntimeError(
                "A page-content chunk contains multiple logical pages."
            )

        if record["pages"] != [record["page_start"]]:
            raise RuntimeError("Invalid page-content page metadata.")

    for record in qa_records:
        if not record["text"].strip():
            raise RuntimeError("An empty Q&A record was generated.")

        if record["page_start"] > record["page_end"]:
            raise RuntimeError("Invalid Q&A page range.")

    logger.info(
        "\n========== PDF PROCESSING SUMMARY ==========\n"
        "Logical printed pages: %s\n"
        "Page content records: %s\n"
        "Question-answer records: %s\n"
        "Total records: %s\n"
        "Chunk size: %s\n"
        "Overlap: %s\n"
        "============================================",
        len(pages_data),
        len(page_records),
        len(qa_records),
        len(all_records),
        chunk_size,
        overlap_chars,
    )

    return all_records


# ============================================================
# 17. DEBUG / LOCAL TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    pdf_path = Path(
        r"D:\Ai\pdf_backend\Motor Insurance Handbook (English).pdf"
    )

    pages = smart_extract_pdf(pdf_path)

    print("\n\n========== EXTRACTED LOGICAL PAGES ==========")
    for page in pages:
        print("\n" + "=" * 80)
        print("LOGICAL PAGE:", page["page"])
        print("PHYSICAL PDF PAGE:", page["physical_page"])
        print("SIDE:", page["spread_side"])
        print("SPLIT X:", page["split_x"])
        print("METHOD:", page["extraction_method"])
        print("\nEXTRACTED TEXT:\n")
        print(page["text"])

    print("\n\n========== QUESTION-ANSWER RECORDS ==========")
    qa_records = create_question_answer_records(pages)

    for record in qa_records:
        text_lower = record["text"].lower()

        if (
            "premium rate is applied" in text_lower
            or "period of the policy" in text_lower
        ):
            print("\n" + "=" * 80)
            print("PAGES:", record["page_start"], "-", record["page_end"])
            print("COMPLETE:", record["is_complete_qa"])
            print("CHARACTERS:", len(record["text"]))
            print("TEXT:\n")
            print(record["text"])
