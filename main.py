
import logging
import os
import tempfile

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query
)
from pydantic import BaseModel, Field

from controllers.action import extract_and_chunk_pdf, smart_extract_pdf
from controllers.vector_store import (
    store_chunks_in_qdrant,
    search_similar_chunks,
    get_unique_pdf_files
)

from controllers.llm_service import generate_rag_answer


# --------------------------------------------------
# 1. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="PDF QnA System",
    description=(
        "PDF extraction, chunking, embedding, "
        "vector storage, semantic search and RAG QnA"
    ),
    version="1.0.0"
)


# --------------------------------------------------
# 3. SEARCH REQUEST MODEL
# --------------------------------------------------

class SearchRequest(BaseModel):

    query: str

    top_k: int = Field(
        default=3,
        ge=1,
        le=20
    )


# --------------------------------------------------
# 4. HOME ROUTE
# --------------------------------------------------

@app.get("/")
async def home():

    return {
        "message": "PDF QnA API is running"
    }


# --------------------------------------------------
# 5. PDF UPLOAD ENDPOINT
# --------------------------------------------------

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_contents = await file.read()

    if not file_contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if not file_contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF."
        )

    temp_path = None

    try:

        # Create temporary PDF
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_contents)
            temp_path = temp_file.name

        logger.info(
            "Processing uploaded PDF: %s",
            file.filename
        )

        # STEP 1: Extract text and generate chunks
        chunks = extract_and_chunk_pdf(temp_path)

        logger.info(
            "Generated %d chunks",
            len(chunks)
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in the PDF."
            )

        # STEP 2: Calculate chunk statistics
        chunk_lengths = [
            len(chunk["text"])
            for chunk in chunks
        ]

        average_chunk_length = round(
            sum(chunk_lengths) / len(chunk_lengths),
            2
        )

        # STEP 3: Store chunks in Qdrant
        stored_count = store_chunks_in_qdrant(
            chunks=chunks,
            filename=file.filename
        )

        logger.info(
            "Successfully stored %d chunks in Qdrant",
            stored_count
        )

        # STEP 4: Return processing summary
        return {
            "filename": file.filename,
            "total_chunks": len(chunks),
            "average_chunk_length": average_chunk_length,
            "stored_in_qdrant": stored_count,
            "message": (
                "PDF extracted, chunked, embedded, "
                "and stored successfully."
            )
        }

    except HTTPException:
        raise

    except Exception:

        logger.exception("PDF processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to process PDF."
        )

    finally:

        # Remove temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

            logger.info("Temporary PDF removed.")

        await file.close()


# --------------------------------------------------
# 6. SEMANTIC SEARCH ENDPOINT
# --------------------------------------------------

@app.post("/search/")
async def search_pdf(
    request: SearchRequest
):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    try:

        results = search_similar_chunks(
            query=request.query,
            top_k=request.top_k
        )

        return {
            "query": request.query,
            "top_k": request.top_k,
            "total_found": len(results),
            "results": results
        }

    except Exception:

        logger.exception("Semantic search failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to search PDF content."
        )


# --------------------------------------------------
# 7. END-TO-END RAG QnA ENDPOINT
# --------------------------------------------------

@app.post("/ask/")
async def ask_question(
    query: str,
    top_k: int = Query(default=3, ge=1, le=20)
):

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # STEP 1: Get complete PDF inventory from Qdrant
        uploaded_pdf_names = get_unique_pdf_files()

        # STEP 2: Retrieve relevant PDF chunks
        retrieved_chunks = search_similar_chunks(
            query=query,
            top_k=top_k
        )

        # STEP 3: Generate answer using Ollama
        answer = generate_rag_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            uploaded_pdf_names=uploaded_pdf_names
        )

        # STEP 4: Return final response
        return {
            "question": query,
            "answer": answer,
            "total_uploaded_pdfs": len(uploaded_pdf_names),
            "uploaded_pdf_names": uploaded_pdf_names,
            "retrieved_chunks": len(retrieved_chunks),
            "sources_used": [
                {
                    "page": chunk.get("page_start", "Unknown"),
                    "score": chunk.get("score"),
                    "source": chunk.get("source", "Unknown")
                }
                for chunk in retrieved_chunks
            ]
        }

    except Exception:

        logger.exception("RAG question answering failed.")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer."
        )
