
import logging
import os
import tempfile
from controllers.llm_service import generate_rag_answer

from pydantic import BaseModel, Field

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from controllers.action import extract_and_chunk_pdf

from controllers.vector_store import (
    store_chunks_in_qdrant,
    search_similar_chunks
)


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
    description="PDF extraction, chunking, embedding, vector storage and semantic search",
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

    # Validate filename
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    # Validate extension
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Read uploaded file
    file_contents = await file.read()

    # Validate empty file
    if not file_contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Validate PDF signature
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

        # ------------------------------------------
        # STEP 1: EXTRACT TEXT AND GENERATE CHUNKS
        # ------------------------------------------

        chunks = extract_and_chunk_pdf(
            temp_path
        )

        logger.info(
            "Generated %d chunks",
            len(chunks)
        )

        # ------------------------------------------
        # STEP 2: CALCULATE CHUNK STATISTICS
        # ------------------------------------------

        chunk_lengths = [
            len(chunk["text"])
            for chunk in chunks
        ]

        average_chunk_length = (
            round(
                sum(chunk_lengths) / len(chunk_lengths),
                2
            )
            if chunk_lengths else 0
        )

        # ------------------------------------------
        # STEP 3: STORE CHUNKS IN QDRANT
        # ------------------------------------------

        stored_count = store_chunks_in_qdrant(
            chunks=chunks,
            filename=file.filename
        )

        logger.info(
            "Successfully stored %d chunks in Qdrant",
            stored_count
        )

        # ------------------------------------------
        # STEP 4: PRINT PROCESSING SUMMARY
        # ------------------------------------------

        print("\n========== PDF PROCESSING ==========")

        print(f"Filename: {file.filename}")

        print(f"Total chunks: {len(chunks)}")

        print(
            f"Average chunk length: "
            f"{average_chunk_length}"
        )

        print(
            f"Stored in Qdrant: {stored_count}"
        )

        for chunk in chunks:

            print(
                f"\n--- Chunk {chunk['chunk_id']} "
                f"| Pages {chunk['page_start']}-{chunk['page_end']} "
                f"| Subject {chunk['subject']} "
                f"| Method {chunk['extraction_method']} "
                f"| Characters {len(chunk['text'])} ---"
            )

            print(chunk["text"])

        print("===================================\n")

        # ------------------------------------------
        # STEP 5: RETURN API RESPONSE
        # ------------------------------------------

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

        logger.exception(
            "PDF processing failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process PDF."
        )

    finally:

        # ------------------------------------------
        # STEP 6: CLEANUP
        # ------------------------------------------

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)

            logger.info(
                "Temporary PDF removed."
            )

        await file.close()


# --------------------------------------------------
# 6. SEMANTIC SEARCH ENDPOINT
# --------------------------------------------------

@app.post("/search/")
async def search_pdf(
    request: SearchRequest
):

    # STEP 1: Validate query
    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    # STEP 2: Perform semantic search
    results = search_similar_chunks(
        query=request.query,
        top_k=request.top_k
    )

    # STEP 3: Return search results
    return {
        "query": request.query,
        "top_k": request.top_k,
        "total_found": len(results),
        "results": results
    }



# Add this import at the top of main.py
from controllers.llm_service import generate_rag_answer

# --------------------------------------------------
# 7. END-TO-END RAG QnA ENDPOINT (/ask/)
# --------------------------------------------------

@app.post("/ask/")
async def ask_question(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Part 2: Embed query -> Cosine Similarity Search -> Retrieve Top-K chunks
    retrieved_chunks = search_similar_chunks(
        query=request.query,
        top_k=request.top_k
    )

    # Part 3: Top-K Context + User Query -> LLM -> Generate Final Answer
    try:
        answer = generate_rag_answer(
            query=request.query,
            retrieved_chunks=retrieved_chunks
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "question": request.query,
        "answer": answer,
        "sources_used": [
            {
                "page": c.get("page") or c.get("payload", {}).get("page") or "Unknown",
                "score": c.get("score"),
                "source": c.get("source") or c.get("payload", {}).get("source") or "Unknown"
            }
            for c in retrieved_chunks
        ]
    }


