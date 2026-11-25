
import logging
import os
import re
import requests

logger = logging.getLogger(__name__)


# ==================================================
# 1. OLLAMA CONFIGURATION
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

REQUEST_TIMEOUT = 180

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048
CONTEXT_WINDOW = 8192


# ==================================================
# 2. NORMALIZE TEXT
# ==================================================

def normalize_text(text: str) -> str:

    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================================================
# 3. DETECT QUESTION TYPE
# ==================================================

def is_comprehensive_question(query: str) -> bool:

    query = normalize_text(query)

    patterns = [
        r"\ball\b",
        r"\bevery\b",
        r"\bcomplete\b",
        r"\bentire\b",
        r"\bfull\b",
        r"\blist\b",
        r"\btable\b",
        r"\ball the\b",
        r"\bwhat are the\b",
        r"\bhow many\b",
        r"\bmaximum turnaround times\b",
        r"\bturn around times\b",
        r"\bturnaround times\b",
        r"\bservices and their\b",
    ]

    return any(
        re.search(pattern, query)
        for pattern in patterns
    )


# ==================================================
# 4. FORMAT RETRIEVED CONTEXT
# ==================================================

def format_retrieved_context(
    retrieved_chunks: list[dict]
) -> str:

    if not retrieved_chunks:
        return "No relevant PDF content was retrieved."

    context_blocks = []
    seen_texts = set()

    for index, chunk in enumerate(retrieved_chunks, start=1):

        text = (chunk.get("text") or "").strip()

        if not text:
            continue

        normalized_text = normalize_text(text)

        if normalized_text in seen_texts:
            continue

        seen_texts.add(normalized_text)

        source = os.path.basename(
            chunk.get("source") or "Unknown PDF"
        )

        page_start = chunk.get("page_start", "Unknown")
        page_end = chunk.get("page_end", page_start)

        content_type = chunk.get("content_type", "text")

        if page_start == page_end:
            page_info = f"Page {page_start}"
        else:
            page_info = f"Pages {page_start}-{page_end}"

        content_descriptions = {
            "faq": "FAQ QUESTION AND ANSWER",
            "list": "BULLET OR NUMBERED LIST",
            "table": "TABLE CONTENT",
            "text": "NORMAL TEXT"
        }

        content_description = content_descriptions.get(
            content_type,
            "NORMAL TEXT"
        )

        block = f"""
[CONTEXT {index}]
PDF: {source}
LOCATION: {page_info}
CONTENT TYPE: {content_description}

ACTUAL PDF CONTENT:
{text}
"""

        context_blocks.append(block.strip())

    if not context_blocks:
        return "No usable PDF content was retrieved."

    return "\n\n".join(context_blocks)


# ==================================================
# 5. FORMAT PDF INVENTORY
# ==================================================

def format_pdf_inventory(
    uploaded_pdf_names: list[str] | None
) -> str:

    if uploaded_pdf_names is None:
        return "PDF inventory is unavailable."

    unique_names = sorted({
        os.path.basename(name)
        for name in uploaded_pdf_names
        if name
    })

    return (
        f"Total uploaded PDFs: {len(unique_names)}\n"
        f"Uploaded PDF names: "
        f"{', '.join(unique_names) if unique_names else 'None'}"
    )


# ==================================================
# 6. CREATE RAG PROMPT
# ==================================================

def create_rag_prompt(
    query: str,
    formatted_context: str,
    pdf_inventory: str
) -> str:

    comprehensive = is_comprehensive_question(query)

    if comprehensive:

        answering_rules = """
QUESTION TYPE: COMPLETE TABLE / COMPREHENSIVE QUESTION

The user wants a complete answer, not one matching FAQ.

STRICT INSTRUCTIONS:

1. Read all retrieved PDF contexts carefully.

2. Identify every row relevant to the user's question.

3. Include ALL relevant categories, services, and corresponding values.

4. Do not return only the first row or first matching value.

5. Do not omit rows from a table.

6. Preserve the relationship between each category, service,
   and its corresponding value.

7. If a category name appears only once because the original
   PDF uses merged cells, apply that category to its following
   service rows when the table structure clearly indicates this.

8. Preserve exact numerical values, dates and time units.

9. Do not combine values belonging to different services.

10. If the complete table is available, return the complete table.

11. If only part of the table is available in the retrieved
    context, clearly state that the retrieved content is incomplete.
    Never invent missing rows.

12. Do not add facts that are not in the PDF.

OUTPUT FORMAT:

Use a Markdown table with appropriate columns.

Return all available relevant rows.
Do not provide just one matching value.
"""

    else:

        answering_rules = """
QUESTION TYPE: SPECIFIC QUESTION

STRICT INSTRUCTIONS:

1. Answer only the exact question asked by the user.

2. Identify the FAQ or table row that directly matches the question.

3. Use the corresponding answer or value.

4. Do not include neighbouring FAQ questions.

5. Do not answer unrelated questions.

6. Preserve important conditions, dates, fees and exceptions.

7. Do not invent information.

8. Do not repeat the answer.

9. If the answer is unavailable, say:
   "The information is not available in the uploaded PDF."

OUTPUT FORMAT:

Return only the answer.
Do not reproduce the original FAQ question.
Do not explain your reasoning.
"""

    return f"""
You are a precise PDF question-answering assistant.

USER QUESTION:
{query}

RETRIEVED PDF CONTENT:
{formatted_context}

PDF INVENTORY:
{pdf_inventory}

{answering_rules}
"""


# ==================================================
# 7. CALL OLLAMA
# ==================================================

def call_ollama(
    prompt: str,
    num_predict: int = MAX_OUTPUT_TOKENS
) -> str:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": num_predict,
                "num_ctx": CONTEXT_WINDOW
            }
        },
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    result = response.json()

    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty answer."
        )

    if result.get("done_reason") == "length":
        logger.warning(
            "Ollama output may have been truncated."
        )

    return answer


# ==================================================
# 8. GENERATE RAG ANSWER
# ==================================================

def generate_rag_answer(
    query: str,
    retrieved_chunks: list[dict],
    uploaded_pdf_names: list[str] | None = None
) -> str:

    if not query or not query.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return (
            "I could not find relevant information "
            "in the uploaded PDFs to answer this question."
        )

    comprehensive = is_comprehensive_question(query)

    logger.info(
        "Question type: %s",
        "COMPREHENSIVE" if comprehensive else "SPECIFIC"
    )

    logger.info(
        "Retrieved chunk count: %s",
        len(retrieved_chunks)
    )

    # Log the actual retrieved chunks for debugging.
    for index, chunk in enumerate(retrieved_chunks, start=1):

        logger.info(
            "Retrieved chunk %s | Type: %s | Pages: %s-%s | Text: %s",
            index,
            chunk.get("content_type"),
            chunk.get("page_start"),
            chunk.get("page_end"),
            (chunk.get("text") or "")[:500]
        )

    # Step 1: Format context.
    formatted_context = format_retrieved_context(
        retrieved_chunks
    )

    # Step 2: Format PDF inventory.
    pdf_inventory = format_pdf_inventory(
        uploaded_pdf_names
    )

    # Step 3: Create appropriate prompt.
    prompt = create_rag_prompt(
        query=query,
        formatted_context=formatted_context,
        pdf_inventory=pdf_inventory
    )

    try:

        # Step 4: Generate answer.
        answer = call_ollama(prompt)

        logger.info(
            "Answer generated successfully."
        )

        return answer

    except requests.exceptions.RequestException as error:

        logger.exception(
            "Ollama communication failed."
        )

        raise RuntimeError(
            f"Failed to communicate with Ollama: {error}"
        ) from error

    except (ValueError, RuntimeError):

        logger.exception(
            "Failed to generate answer."
        )

        raise
