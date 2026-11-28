
import logging
import os
import tempfile

from pathlib import Path
from typing import Optional

import pymupdf

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)

from pydantic import BaseModel, Field

from controllers.action import extract_and_chunk_pdf

from controllers.vector_store import (
    store_chunks_in_qdrant,
    search_similar_chunks,
    get_unique_pdf_files
)

from controllers.llm_service import generate_rag_answer


# ============================================================
# 1. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. APPLICATION CONFIGURATION
# ============================================================

MAX_UPLOAD_SIZE = int(
    os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024))
)

UPLOAD_READ_SIZE = 1024 * 1024


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PDF QnA System",
    description=(
        "PDF extraction, page-wise chunking, "
        "embedding, Qdrant vector storage, "
        "semantic retrieval and Ollama-based RAG QnA."
    ),
    version="1.1.0"
)


# ============================================================
# 4. REQUEST MODELS
# ============================================================

class SearchRequest(BaseModel):

    query: str = Field(
        min_length=1,
        description="Question or text to search in uploaded PDFs."
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=20
    )


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def get_safe_filename(filename: str) -> str:
    """
    Extract only the filename.

    Prevents directory components from being
    passed into the storage layer.
    """

    filename = str(filename or "")

    filename = filename.replace("\\", "/")

    return Path(filename).name


def validate_pdf_file(file_path: str) -> int:
    """
    Open the uploaded PDF with PyMuPDF.

    Returns:
        Total physical PDF pages.

    Raises:
        ValueError if the PDF cannot be opened
        or contains no pages.
    """

    try:

        with pymupdf.open(file_path) as document:

            if document.is_encrypted:

                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            page_count = len(document)

            if page_count == 0:

                raise ValueError(
                    "The uploaded PDF contains no pages."
                )

            return page_count

    except ValueError:
        raise

    except Exception as error:

        raise ValueError(
            "The uploaded file is not a readable PDF."
        ) from error


def get_chunk_statistics(
    chunks: list[dict]
) -> dict:
    """
    Calculate basic statistics without modifying
    the original extracted chunk text.
    """

    if not chunks:

        return {
            "total_chunks": 0,
            "average_chunk_length": 0,
            "total_pages": 0
        }

    chunk_lengths = [

        len(str(chunk.get("text") or ""))

        for chunk in chunks

    ]

    pages = {

        chunk.get("page_start")

        for chunk in chunks

        if chunk.get("page_start") is not None

    }

    return {

        "total_chunks": len(chunks),

        "average_chunk_length": round(
            sum(chunk_lengths) / len(chunk_lengths),
            2
        ),

        "total_pages_with_chunks": len(pages)

    }


def format_source_information(
    retrieved_chunks: list[dict]
) -> list[dict]:
    """
    Return source metadata for each retrieved chunk.

    Do not deduplicate by text. Different chunks
    can contain identical text on different pages.
    """

    sources = []

    for chunk in retrieved_chunks:

        sources.append({

            "point_id": chunk.get("point_id"),

            "chunk_id": chunk.get("chunk_id"),

            "source": chunk.get(
                "source",
                "Unknown"
            ),

            "page_start": chunk.get(
                "page_start",
                "Unknown"
            ),

            "page_end": chunk.get(
                "page_end",
                chunk.get("page_start", "Unknown")
            ),

            "content_type": chunk.get(
                "content_type",
                "text"
            ),

            "score": chunk.get("score"),

            "logical_unit_id": chunk.get(
                "logical_unit_id"
            ),

            "continuation_part": chunk.get(
                "continuation_part"
            ),

            "is_continuation": chunk.get(
                "is_continuation",
                False
            ),

            "retrieval_reason": chunk.get(
                "retrieval_reason",
                "semantic_search"
            )

        })

    return sources


# ============================================================
# 6. HOME ROUTE
# ============================================================

@app.get("/")
async def home():

    return {

        "message": "PDF QnA API is running",

        "status": "healthy",

        "version": app.version

    }


# ============================================================
# 7. PDF UPLOAD ENDPOINT
# ============================================================

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Complete upload pipeline:

        Upload PDF
            |
        Validate filename and file type
            |
        Save temporary PDF
            |
        Validate PDF readability
            |
        Extract physical pages
            |
        Generate chunks
            |
        Generate embeddings
            |
        Store chunks in Qdrant
            |
        Return upload summary
    """

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    filename = get_safe_filename(
        file.filename
    )

    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    temp_path: Optional[str] = None

    total_bytes = 0

    try:

        # ----------------------------------------------------
        # STEP 1: CREATE TEMPORARY PDF FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_path = temp_file.name

            first_chunk = await file.read(UPLOAD_READ_SIZE)

            if not first_chunk:

                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty."
                )

            # Check PDF file signature.
            if not first_chunk.startswith(b"%PDF-"):

                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not a valid PDF."
                )

            total_bytes += len(first_chunk)

            if total_bytes > MAX_UPLOAD_SIZE:

                raise HTTPException(
                    status_code=413,
                    detail="Uploaded PDF exceeds the maximum file size."
                )

            temp_file.write(first_chunk)

            # Read the remaining file in smaller pieces.
            while True:

                file_chunk = await file.read(
                    UPLOAD_READ_SIZE
                )

                if not file_chunk:
                    break

                total_bytes += len(file_chunk)

                if total_bytes > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded PDF exceeds the maximum file size."
                    )

                temp_file.write(file_chunk)

        logger.info(
            "PDF uploaded | Filename: %s | Size: %s bytes",
            filename,
            total_bytes
        )

        # ----------------------------------------------------
        # STEP 2: VALIDATE PDF READABILITY
        # ----------------------------------------------------

        try:

            page_count = validate_pdf_file(
                temp_path
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(error)
            ) from error

        logger.info(
            "PDF validation successful | Pages: %s",
            page_count
        )

        # ----------------------------------------------------
        # STEP 3: EXTRACT AND CHUNK PDF
        # ----------------------------------------------------

        logger.info(
            "Starting PDF extraction and chunking."
        )

        chunks = extract_and_chunk_pdf(
            temp_path
        )

        logger.info(
            "PDF extraction completed | Chunks: %s",
            len(chunks)
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable text found in the PDF. "
                    "Check whether the PDF contains readable "
                    "text or whether OCR is required."
                )
            )

        # ----------------------------------------------------
        # STEP 4: CALCULATE CHUNK STATISTICS
        # ----------------------------------------------------

        statistics = get_chunk_statistics(
            chunks
        )

        logger.info(
            "Chunk statistics | Total: %s | Average length: %s",
            statistics["total_chunks"],
            statistics["average_chunk_length"]
        )

        # ----------------------------------------------------
        # STEP 5: STORE CHUNKS IN QDRANT
        # ----------------------------------------------------

        logger.info(
            "Starting embedding and Qdrant storage."
        )

        stored_count = store_chunks_in_qdrant(

            chunks=chunks,

            filename=filename

        )

        logger.info(
            "Qdrant storage completed | Stored points: %s",
            stored_count
        )

        # ----------------------------------------------------
        # STEP 6: RETURN UPLOAD SUMMARY
        # ----------------------------------------------------

        return {

            "filename": filename,

            "file_size_bytes": total_bytes,

            "total_pdf_pages": page_count,

            "total_chunks": statistics["total_chunks"],

            "total_pages_with_chunks": (
                statistics["total_pages_with_chunks"]
            ),

            "average_chunk_length": (
                statistics["average_chunk_length"]
            ),

            "stored_in_qdrant": stored_count,

            "message": (
                "PDF extracted, chunked, embedded, "
                "and stored successfully."
            )

        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "PDF processing failed | Filename: %s",
            filename
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process PDF. Check server logs."
        )

    finally:

        # ----------------------------------------------------
        # STEP 7: CLEAN UP TEMPORARY FILE
        # ----------------------------------------------------

        if temp_path and os.path.exists(temp_path):

            try:

                os.remove(temp_path)

                logger.info(
                    "Temporary PDF removed."
                )

            except OSError:

                logger.exception(
                    "Could not remove temporary PDF: %s",
                    temp_path
                )

        await file.close()


# ============================================================
# 8. SEMANTIC SEARCH ENDPOINT
# ============================================================

@app.post("/search/")
async def search_pdf(
    request: SearchRequest
):
    """
    Search Qdrant for semantically similar PDF chunks.
    """

    query = request.query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    try:

        logger.info(
            "Semantic search started | Query: %s | top_k: %s",
            query,
            request.top_k
        )

        results = search_similar_chunks(

            query=query,

            top_k=request.top_k

        )

        logger.info(
            "Semantic search completed | Results: %s",
            len(results)
        )

        return {

            "query": query,

            "top_k": request.top_k,

            "total_found": len(results),

            "results": results

        }

    except Exception:

        logger.exception(
            "Semantic search failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to search PDF content."
        )


# ============================================================
# 9. END-TO-END RAG QUESTION-ANSWERING ENDPOINT
# ============================================================

@app.post("/ask/")
async def ask_question(

    query: str = Query(
        ...,
        min_length=1,
        description="Question to ask about uploaded PDFs."
    ),

    top_k: int = Query(
        default=3,
        ge=1,
        le=20
    )

):
    """
    Complete RAG pipeline:

        User question
            |
        Get uploaded PDF inventory
            |
        Search Qdrant
            |
        Retrieve relevant chunks
            |
        Format original PDF content
            |
        Create dynamic RAG prompt
            |
        Call Ollama
            |
        Return answer and sources
    """

    query = query.strip()

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        logger.info(
            "RAG request received | Question: %s | top_k: %s",
            query,
            top_k
        )

        # ----------------------------------------------------
        # STEP 1: GET PDF INVENTORY
        # ----------------------------------------------------

        uploaded_pdf_names = get_unique_pdf_files()

        logger.info(
            "Uploaded PDFs available: %s",
            len(uploaded_pdf_names)
        )

        # ----------------------------------------------------
        # STEP 2: RETRIEVE RELEVANT CHUNKS
        # ----------------------------------------------------

        retrieved_chunks = search_similar_chunks(

            query=query,

            top_k=top_k

        )

        logger.info(
            "Retrieved chunks: %s",
            len(retrieved_chunks)
        )

        if not retrieved_chunks:

            return {

                "question": query,

                "answer": (
                    "I could not find relevant information "
                    "in the uploaded PDFs to answer this question."
                ),

                "total_uploaded_pdfs": len(uploaded_pdf_names),

                "uploaded_pdf_names": uploaded_pdf_names,

                "retrieved_chunks": 0,

                "sources_used": []

            }

        # ----------------------------------------------------
        # STEP 3: GENERATE ANSWER USING OLLAMA
        # ----------------------------------------------------

        answer = generate_rag_answer(

            query=query,

            retrieved_chunks=retrieved_chunks,

            uploaded_pdf_names=uploaded_pdf_names

        )

        # ----------------------------------------------------
        # STEP 4: FORMAT SOURCE INFORMATION
        # ----------------------------------------------------

        sources = format_source_information(
            retrieved_chunks
        )

        # ----------------------------------------------------
        # STEP 5: RETURN FINAL RESPONSE
        # ----------------------------------------------------

        logger.info(
            "RAG answer generated successfully."
        )

        return {

            "question": query,

            "answer": answer,

            "total_uploaded_pdfs": len(uploaded_pdf_names),

            "uploaded_pdf_names": uploaded_pdf_names,

            "retrieved_chunks": len(retrieved_chunks),

            "sources_used": sources

        }

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            "RAG question answering failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check server logs."
        )


# ============================================================
# 10. APPLICATION STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    logger.info(
        "PDF QnA API started successfully."
    )

    logger.info(
        "Maximum upload size: %s MB",
        round(MAX_UPLOAD_SIZE / (1024 * 1024), 2)
    )


# ============================================================
# 11. APPLICATION SHUTDOWN
# ============================================================

@app.on_event("shutdown")
async def shutdown_event():

    logger.info(
        "PDF QnA API shutting down."
    )
