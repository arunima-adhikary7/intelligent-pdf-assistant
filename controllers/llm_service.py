
import requests
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------
# 1. OLLAMA CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2"


# --------------------------------------------------
# 2. GENERATE RAG ANSWER
# --------------------------------------------------

def generate_rag_answer(
    query: str,
    retrieved_chunks: list[dict]
) -> str:

    # Check whether retrieval returned any chunks
    if not retrieved_chunks:

        return (
            "I could not find any relevant information "
            "in the uploaded PDF to answer your question."
        )

    # ----------------------------------------------
    # STEP 1: FORMAT RETRIEVED CHUNKS AS CONTEXT
    # ----------------------------------------------

    context_blocks = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        # Extract metadata safely
        page_start = chunk.get("page_start", "Unknown")
        page_end = chunk.get("page_end", page_start)

        score = chunk.get("score", 0.0)

        text = chunk.get("text", "")

        source = chunk.get("source", "Unknown")

        # Format page information
        if page_start == page_end:
            page_info = f"Page {page_start}"
        else:
            page_info = f"Pages {page_start}-{page_end}"

        context_blocks.append(
            f"[Source {i} | {source} | {page_info} | Score {score}]\n"
            f"{text}"
        )

    formatted_context = "\n\n---\n\n".join(context_blocks)

    # ----------------------------------------------
    # STEP 2: CONSTRUCT RAG PROMPT
    # ----------------------------------------------

    prompt = f"""
You are a helpful academic assistant.

Answer the user's question using ONLY the provided PDF context.

Instructions:
- Be clear, structured, and accurate.
- Do not use external knowledge.
- Cite page numbers where information is found.
- If the answer is not present in the context, say:
  "The uploaded document does not contain information to answer this question."

PDF CONTEXT:

{formatted_context}

USER QUESTION:

{query}

ANSWER:
"""

    # ----------------------------------------------
    # STEP 3: SEND PROMPT TO OLLAMA
    # ----------------------------------------------

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            # timeout=180
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return answer

    except requests.exceptions.RequestException as e:

        logger.exception("Ollama LLM error.")

        raise RuntimeError(
            f"Failed to communicate with local LLM: {e}"
        )

    except (ValueError, RuntimeError):

        logger.exception("Failed to process Ollama response.")

        raise
