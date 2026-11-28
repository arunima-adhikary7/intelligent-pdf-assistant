
import os
import uuid
import math
import logging
import json
import re
from typing import Optional, Any

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)


logger = logging.getLogger(__name__)


# ============================================================
# 1. CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6335"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "pdf_syllabus_qna"
)

BATCH_SIZE = max(
    1,
    int(os.getenv("QDRANT_BATCH_SIZE", "64"))
)

DELETE_BATCH_SIZE = max(
    1,
    int(os.getenv("QDRANT_DELETE_BATCH_SIZE", "500"))
)

MIN_RETRIEVAL_LIMIT = max(
    1,
    int(os.getenv("MIN_RETRIEVAL_LIMIT", "12"))
)

MAX_RETRIEVAL_LIMIT = max(
    MIN_RETRIEVAL_LIMIT,
    int(os.getenv("MAX_RETRIEVAL_LIMIT", "200"))
)

RETRIEVAL_SCALE = max(
    1.0,
    float(os.getenv("RETRIEVAL_SCALE", "4"))
)

DEFAULT_MIN_SCORE = None


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

logger.info(
    "Loading embedding model: %s",
    EMBEDDING_MODEL_NAME
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

VECTOR_SIZE = (
    embedding_model.get_sentence_embedding_dimension()
)

if not VECTOR_SIZE:
    raise RuntimeError(
        "Could not determine embedding vector dimension."
    )

logger.info(
    "Embedding model loaded. Vector size: %s",
    VECTOR_SIZE
)


# ============================================================
# 3. CONNECT TO QDRANT
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# ============================================================
# 4. INITIALIZE COLLECTION
# ============================================================

def init_qdrant_collection() -> None:

    if not qdrant_client.collection_exists(
        collection_name=COLLECTION_NAME
    ):

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        logger.info(
            "Created Qdrant collection: %s",
            COLLECTION_NAME
        )

        return

    collection_info = qdrant_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    existing_vectors = (
        collection_info.config.params.vectors
    )

    # This implementation expects one unnamed dense vector.
    existing_size = getattr(
        existing_vectors,
        "size",
        None
    )

    if existing_size is None:

        raise RuntimeError(
            "Existing collection uses named vectors or "
            "an unsupported vector configuration."
        )

    if existing_size != VECTOR_SIZE:

        raise RuntimeError(
            f"Vector dimension mismatch. "
            f"Collection: {existing_size}, "
            f"Model: {VECTOR_SIZE}."
        )

    logger.info(
        "Using existing Qdrant collection: %s",
        COLLECTION_NAME
    )


init_qdrant_collection()


# ============================================================
# 5. COMMON HELPERS
# ============================================================

def get_original_text(chunk: dict) -> str:
    """
    Return the exact text received from extraction.

    IMPORTANT:
    Do not clean, normalize, strip, or rewrite the text here.

    The same text is:
        1. Sent to the embedding model.
        2. Stored in the Qdrant payload.
    """

    text = chunk.get("text")

    if text is None:
        return ""

    return str(text)


def is_valid_text(text: str) -> bool:
    """
    Check whether text contains meaningful characters.

    This function does not modify the original text.
    """

    return bool(text and text.strip())


def create_source_filter(filename: str) -> Filter:

    return Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=filename)
            )
        ]
    )


def get_collection_point_count() -> int:

    info = qdrant_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    return int(info.points_count or 0)


def get_filtered_point_count(
    filename: Optional[str] = None
) -> int:

    if filename:

        result = qdrant_client.count(
            collection_name=COLLECTION_NAME,
            count_filter=create_source_filter(filename),
            exact=True
        )

        return int(result.count)

    return get_collection_point_count()


def normalize_for_comparison(text: Optional[str]) -> str:
    """
    Used only for query-intent detection.

    Never use this function on text being stored or embedded.
    """

    return " ".join(
        str(text or "").lower().split()
    )


# ============================================================
# 6. QUESTION SCOPE DETECTION
# ============================================================

def get_question_scope(query: str) -> str:

    query = normalize_for_comparison(query)

    comprehensive_patterns = [

        r"\ball\b",
        r"\bevery\b",
        r"\beach\b",
        r"\bcomplete\b",
        r"\bentire\b",
        r"\bfull\b",
        r"\blist\b",
        r"\benumerate\b",
        r"\bwhat are\b",
        r"\bwhich are\b",
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bdifferent types\b",
        r"\btypes of\b",
        r"\bcategories\b",
        r"\bfeatures\b",
        r"\bconditions\b",
        r"\brequirements\b",
        r"\bexceptions\b",
        r"\bsteps\b",
        r"\bturnaround times\b",
        r"\bturn around times\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifferences\b",
        r"\btable\b",
        r"\brespectively\b"

    ]

    if any(
        re.search(pattern, query)
        for pattern in comprehensive_patterns
    ):
        return "comprehensive"

    return "specific"


# ============================================================
# 7. DYNAMIC RETRIEVAL LIMIT
# ============================================================

def calculate_retrieval_limit(
    query: str,
    total_points: Optional[int] = None,
    requested_top_k: Optional[int] = None
) -> int:

    if requested_top_k is not None:

        if (
            not isinstance(requested_top_k, int)
            or requested_top_k < 1
        ):

            raise ValueError(
                "top_k must be a positive integer."
            )

        return min(
            requested_top_k,
            MAX_RETRIEVAL_LIMIT
        )

    if total_points is None:
        total_points = get_collection_point_count()

    if total_points <= 0:
        return 0

    calculated_limit = math.ceil(
        math.sqrt(total_points) * RETRIEVAL_SCALE
    )

    calculated_limit = max(
        MIN_RETRIEVAL_LIMIT,
        calculated_limit
    )

    if get_question_scope(query) == "comprehensive":

        calculated_limit = math.ceil(
            calculated_limit * 1.75
        )

    calculated_limit = min(
        calculated_limit,
        total_points,
        MAX_RETRIEVAL_LIMIT
    )

    return max(1, calculated_limit)


# ============================================================
# 8. GET EXISTING POINT IDS FOR A PDF
# ============================================================

def get_pdf_point_ids(filename: str) -> list:

    point_ids = []

    offset = None

    pdf_filter = create_source_filter(filename)

    while True:

        records, offset = qdrant_client.scroll(

            collection_name=COLLECTION_NAME,

            scroll_filter=pdf_filter,

            limit=DELETE_BATCH_SIZE,

            offset=offset,

            with_payload=False,

            with_vectors=False

        )

        point_ids.extend(
            record.id
            for record in records
        )

        if offset is None:
            break

    return point_ids


# ============================================================
# 9. DELETE OLD POINTS IN BATCHES
# ============================================================

def delete_points_in_batches(point_ids: list) -> None:

    if not point_ids:
        return

    for start in range(
        0,
        len(point_ids),
        DELETE_BATCH_SIZE
    ):

        batch = point_ids[
            start:start + DELETE_BATCH_SIZE
        ]

        qdrant_client.delete(

            collection_name=COLLECTION_NAME,

            points_selector=PointIdsList(
                points=batch
            ),

            wait=True

        )

    logger.info(
        "Deleted %s Qdrant points",
        len(point_ids)
    )


# ============================================================
# 10. BUILD QDRANT PAYLOAD
# ============================================================

def create_chunk_payload(
    chunk: dict,
    original_text: str,
    filename: str
) -> dict:
    """
    Store the exact extracted text and its metadata.

    No text cleaning happens here.
    """

    payload = {

        # Exact original extracted text.
        "text": original_text,

        "source": filename,

        "chunk_id": chunk.get("chunk_id"),

        "page_start": chunk.get("page_start"),

        "page_end": chunk.get("page_end"),

        "pages": chunk.get("pages") or [],

        "subject": chunk.get("subject"),

        "content_type": (
            chunk.get("content_type") or "text"
        ),

        "table_id": chunk.get("table_id"),

        "extraction_method": (
            chunk.get("extraction_method") or "unknown"
        ),

        "chunk_size_setting": (
            chunk.get("chunk_size_setting")
        ),

        "overlap_setting": (
            chunk.get("overlap_setting")
        ),

        # Cross-page continuation metadata.
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

        "continuation_of": chunk.get(
            "continuation_of"
        )

    }

    # Preserve other JSON-compatible metadata.
    for key, value in chunk.items():

        if key in payload:
            continue

        if key == "text":
            continue

        if value is None:
            continue

        try:

            json.dumps(value)

            payload[key] = value

        except (
            TypeError,
            ValueError
        ):

            logger.warning(
                "Skipping non-serializable metadata: %s",
                key
            )

    return payload


# ============================================================
# 11. STORE PDF CHUNKS IN QDRANT
# ============================================================

def store_chunks_in_qdrant(
    chunks: list[dict],
    filename: str
) -> int:
    """
    Store every extracted chunk as an independent Qdrant point.

    IMPORTANT:

    - No text cleaning.
    - No text normalization.
    - No deduplication by text.
    - No merging of physical PDF pages.
    - One embedding for every independent chunk.
    - Exact extracted text is stored in the payload.
    """

    if not filename or not filename.strip():

        raise ValueError(
            "Filename cannot be empty."
        )

    if not chunks:

        logger.warning(
            "No chunks received for %s. "
            "Existing records retained.",
            filename
        )

        return 0

    logger.info(
        "Starting storage | PDF: %s | Chunks: %s",
        filename,
        len(chunks)
    )

    # Keep the old PDF records until the new upload succeeds.
    old_point_ids = get_pdf_point_ids(filename)

    inserted_point_ids = []

    total_stored = 0

    try:

        for start in range(
            0,
            len(chunks),
            BATCH_SIZE
        ):

            batch_chunks = chunks[
                start:start + BATCH_SIZE
            ]

            valid_items = []

            for chunk in batch_chunks:

                if not isinstance(chunk, dict):

                    logger.warning(
                        "Skipping invalid chunk."
                    )

                    continue

                original_text = get_original_text(
                    chunk
                )

                if not is_valid_text(original_text):

                    logger.warning(
                        "Skipping empty chunk | Chunk ID: %s",
                        chunk.get("chunk_id")
                    )

                    continue

                # Keep the original text unchanged.
                valid_items.append(
                    (chunk, original_text)
                )

            if not valid_items:
                continue

            valid_chunks = [
                item[0]
                for item in valid_items
            ]

            original_texts = [
                item[1]
                for item in valid_items
            ]

            # ------------------------------------------------
            # EMBED THE ORIGINAL TEXT.
            # ------------------------------------------------

            logger.info(
                "Embedding batch | Items: %s",
                len(original_texts)
            )

            vectors = embedding_model.encode(

                original_texts,

                batch_size=BATCH_SIZE,

                show_progress_bar=False,

                convert_to_numpy=True,

                normalize_embeddings=True

            ).tolist()

            if len(vectors) != len(valid_chunks):

                raise RuntimeError(
                    "Embedding count does not match "
                    "the number of valid chunks."
                )

            # Validate vectors.
            for vector in vectors:

                if len(vector) != VECTOR_SIZE:

                    raise RuntimeError(
                        "Invalid embedding dimension."
                    )

                if not all(
                    math.isfinite(float(value))
                    for value in vector
                ):

                    raise RuntimeError(
                        "Embedding contains invalid numeric values."
                    )

            points = []

            batch_point_ids = []

            for chunk, original_text, vector in zip(
                valid_chunks,
                original_texts,
                vectors
            ):

                point_id = str(uuid.uuid4())

                batch_point_ids.append(point_id)

                payload = create_chunk_payload(

                    chunk=chunk,

                    original_text=original_text,

                    filename=filename

                )

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

                logger.info(
                    "Prepared Qdrant point | "
                    "Chunk: %s | Page: %s | "
                    "Type: %s | Characters: %s",
                    chunk.get("chunk_id"),
                    chunk.get("page_start"),
                    chunk.get("content_type"),
                    len(original_text)
                )

            # ------------------------------------------------
            # STORE INDEPENDENT POINTS.
            # ------------------------------------------------

            qdrant_client.upsert(

                collection_name=COLLECTION_NAME,

                points=points,

                wait=True

            )

            inserted_point_ids.extend(
                batch_point_ids
            )

            total_stored += len(points)

            logger.info(
                "Stored batch | Batch points: %s | "
                "Total stored: %s",
                len(points),
                total_stored
            )

        if total_stored == 0:

            logger.warning(
                "No valid chunks stored for %s. "
                "Old records retained.",
                filename
            )

            return 0

    except Exception:

        logger.exception(
            "Failed to store chunks for %s",
            filename
        )

        # Roll back newly inserted points.
        try:

            delete_points_in_batches(
                inserted_point_ids
            )

        except Exception:

            logger.exception(
                "Rollback failed for %s",
                filename
            )

        raise

    # --------------------------------------------------------
    # DELETE OLD RECORDS AFTER SUCCESSFUL NEW STORAGE.
    # --------------------------------------------------------

    try:

        delete_points_in_batches(
            old_point_ids
        )

    except Exception:

        logger.exception(
            "New records stored, but old records could not "
            "all be deleted for %s.",
            filename
        )

    logger.info(
        "Storage completed | PDF: %s | "
        "New points: %s | Old points: %s",
        filename,
        total_stored,
        len(old_point_ids)
    )

    return total_stored


# ============================================================
# 12. FORMAT SEARCH RESULTS
# ============================================================

def format_search_results(
    search_results,
    min_score: Optional[float] = None
) -> list[dict]:
    """
    Convert Qdrant hits into application dictionaries.

    Preserve the exact text stored in Qdrant.

    Do not deduplicate by text.
    """

    retrieved_chunks = []

    for hit in search_results:

        payload = hit.payload or {}

        score = float(hit.score)

        if (
            min_score is not None
            and score < min_score
        ):
            continue

        # Do not strip or clean the stored text.
        text = payload.get("text")

        if text is None or not str(text).strip():
            continue

        retrieved_chunks.append({

            "point_id": str(hit.id),

            "score": round(score, 5),

            "chunk_id": payload.get("chunk_id"),

            "page_start": payload.get("page_start"),

            "page_end": payload.get("page_end"),

            "pages": payload.get("pages") or [],

            "subject": payload.get("subject"),

            "content_type": (
                payload.get("content_type") or "text"
            ),

            "table_id": payload.get("table_id"),

            "extraction_method": (
                payload.get("extraction_method")
            ),

            "chunk_size_setting": (
                payload.get("chunk_size_setting")
            ),

            "overlap_setting": (
                payload.get("overlap_setting")
            ),

            "source": (
                payload.get("source") or "Unknown PDF"
            ),

            # Exact stored text.
            "text": text,

            # Continuation metadata.
            "logical_unit_id": payload.get(
                "logical_unit_id"
            ),

            "continuation_part": payload.get(
                "continuation_part"
            ),

            "is_continuation": payload.get(
                "is_continuation",
                False
            ),

            "continuation_of": payload.get(
                "continuation_of"
            ),

            "retrieval_reason": "semantic_search"

        })

    return retrieved_chunks


# ============================================================
# 13. EXPAND CROSS-PAGE CONTINUATION CHUNKS
# ============================================================

def expand_continuation_chunks(
    retrieved_chunks: list[dict]
) -> list[dict]:
    """
    If a retrieved chunk belongs to a linked logical unit,
    fetch all other chunks belonging to that unit.

    Physical pages remain independent Qdrant points.
    """

    if not retrieved_chunks:
        return []

    expanded = []

    seen_point_ids = set()

    # Preserve original semantic search results and ranking.
    for chunk in retrieved_chunks:

        point_id = str(
            chunk.get("point_id")
        )

        if point_id in seen_point_ids:
            continue

        seen_point_ids.add(point_id)

        expanded.append(chunk)

    # Find logical continuation groups.
    logical_units = {}

    for chunk in retrieved_chunks:

        logical_unit_id = chunk.get(
            "logical_unit_id"
        )

        if not logical_unit_id:
            continue

        logical_units.setdefault(
            logical_unit_id,
            set()
        ).add(
            chunk.get("source")
        )

    if not logical_units:
        return expanded

    # Fetch all linked parts.
    for logical_unit_id, source_names in logical_units.items():

        for source_name in source_names:

            if not source_name:
                continue

            continuation_filter = Filter(
                must=[

                    FieldCondition(
                        key="logical_unit_id",
                        match=MatchValue(
                            value=logical_unit_id
                        )
                    ),

                    FieldCondition(
                        key="source",
                        match=MatchValue(
                            value=source_name
                        )
                    )

                ]
            )

            offset = None

            fetched_count = 0

            while True:

                records, offset = qdrant_client.scroll(

                    collection_name=COLLECTION_NAME,

                    scroll_filter=continuation_filter,

                    limit=DELETE_BATCH_SIZE,

                    offset=offset,

                    with_payload=True,

                    with_vectors=False

                )

                for record in records:

                    point_id = str(record.id)

                    if point_id in seen_point_ids:
                        continue

                    payload = record.payload or {}

                    text = payload.get("text")

                    if text is None or not str(text).strip():
                        continue

                    continuation_chunk = {

                        "point_id": point_id,

                        # Not independently ranked by semantic search.
                        "score": None,

                        "chunk_id": payload.get("chunk_id"),

                        "page_start": payload.get("page_start"),

                        "page_end": payload.get("page_end"),

                        "pages": payload.get("pages") or [],

                        "subject": payload.get("subject"),

                        "content_type": (
                            payload.get("content_type") or "text"
                        ),

                        "table_id": payload.get("table_id"),

                        "extraction_method": (
                            payload.get("extraction_method")
                        ),

                        "chunk_size_setting": (
                            payload.get("chunk_size_setting")
                        ),

                        "overlap_setting": (
                            payload.get("overlap_setting")
                        ),

                        "source": (
                            payload.get("source") or source_name
                        ),

                        # Exact stored text.
                        "text": text,

                        "logical_unit_id": (
                            payload.get("logical_unit_id")
                        ),

                        "continuation_part": (
                            payload.get("continuation_part")
                        ),

                        "is_continuation": (
                            payload.get(
                                "is_continuation",
                                False
                            )
                        ),

                        "continuation_of": (
                            payload.get("continuation_of")
                        ),

                        "retrieval_reason": (
                            "continuation_expansion"
                        )

                    }

                    expanded.append(
                        continuation_chunk
                    )

                    seen_point_ids.add(
                        point_id
                    )

                    fetched_count += 1

                if offset is None:
                    break

            logger.info(
                "Continuation expansion | "
                "Logical ID: %s | Source: %s | "
                "Additional records: %s",
                logical_unit_id,
                source_name,
                fetched_count
            )

    # Original semantic results remain first.
    original_ids = {
        str(chunk.get("point_id"))
        for chunk in retrieved_chunks
    }

    original_results = [
        chunk
        for chunk in expanded
        if str(chunk.get("point_id")) in original_ids
    ]

    additional_results = [
        chunk
        for chunk in expanded
        if str(chunk.get("point_id")) not in original_ids
    ]

    # Sort additional continuation records by document order.
    additional_results.sort(
        key=lambda chunk: (

            chunk.get("source") or "",

            chunk.get("page_start") or 0,

            chunk.get("continuation_part") or 0,

            chunk.get("chunk_id") or 0

        )
    )

    return original_results + additional_results


# ============================================================
# 14. SEARCH QDRANT
# ============================================================

def search_qdrant(
    query_vector: list,
    limit: int,
    filename: Optional[str] = None
):
    """
    Perform semantic vector search.
    """

    search_filter = None

    if filename:

        search_filter = create_source_filter(
            filename
        )

    response = qdrant_client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        query_filter=search_filter,

        limit=limit,

        with_payload=True,

        with_vectors=False

    )

    return response.points


# ============================================================
# 15. SEARCH SIMILAR CHUNKS
# ============================================================

def search_similar_chunks(
    query: str,
    top_k: Optional[int] = None,
    filename: Optional[str] = None,
    min_score: Optional[float] = DEFAULT_MIN_SCORE
) -> list[dict]:
    """
    Search relevant chunks and expand linked continuations.

    The original semantic ranking is preserved.
    """

    if not query or not query.strip():
        return []

    total_points = get_filtered_point_count(
        filename
    )

    if total_points == 0:

        logger.warning(
            "No Qdrant points found for the requested scope."
        )

        return []

    retrieval_limit = calculate_retrieval_limit(

        query=query,

        total_points=total_points,

        requested_top_k=top_k

    )

    question_scope = get_question_scope(
        query
    )

    logger.info(
        "Retrieval started | Scope: %s | "
        "Points: %s | Retrieval limit: %s",
        question_scope,
        total_points,
        retrieval_limit
    )

    # Embed the user's query.
    query_vector = embedding_model.encode(

        query,

        convert_to_numpy=True,

        normalize_embeddings=True

    ).tolist()

    if len(query_vector) != VECTOR_SIZE:

        raise RuntimeError(
            "Query embedding dimension mismatch."
        )

    # Initial semantic search.
    search_results = search_qdrant(

        query_vector=query_vector,

        limit=retrieval_limit,

        filename=filename

    )

    retrieved_chunks = format_search_results(

        search_results=search_results,

        min_score=min_score

    )

    logger.info(
        "Semantic search returned %s hits. "
        "%s passed score filtering.",
        len(search_results),
        len(retrieved_chunks)
    )

    # Expand linked answer continuations.
    retrieved_chunks = expand_continuation_chunks(
        retrieved_chunks
    )

    # Log final retrieval results.
    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        logger.info(
            "Result %s | Point: %s | Score: %s | "
            "Source: %s | Page: %s | Type: %s | "
            "Logical ID: %s | Part: %s | Reason: %s",
            index,
            chunk.get("point_id"),
            chunk.get("score"),
            chunk.get("source"),
            chunk.get("page_start"),
            chunk.get("content_type"),
            chunk.get("logical_unit_id"),
            chunk.get("continuation_part"),
            chunk.get("retrieval_reason")
        )

    logger.info(
        "Retrieval completed. Final records: %s",
        len(retrieved_chunks)
    )

    return retrieved_chunks


# ============================================================
# 16. GET UNIQUE PDF FILENAMES
# ============================================================

def get_unique_pdf_files() -> list[str]:

    unique_files = set()

    offset = None

    while True:

        records, offset = qdrant_client.scroll(

            collection_name=COLLECTION_NAME,

            limit=DELETE_BATCH_SIZE,

            offset=offset,

            with_payload=["source"],

            with_vectors=False

        )

        for record in records:

            payload = record.payload or {}

            filename = payload.get("source")

            if filename:
                unique_files.add(filename)

        if offset is None:
            break

    return sorted(unique_files)


# ============================================================
# 17. COLLECTION INFORMATION
# ============================================================

def get_collection_info() -> dict:

    info = qdrant_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    return {

        "collection_name": COLLECTION_NAME,

        "total_points": info.points_count,

        "vector_size": VECTOR_SIZE,

        "distance": "Cosine",

        "embedding_model": EMBEDDING_MODEL_NAME

    }


# ============================================================
# 18. INSPECT QDRANT RECORDS
# ============================================================

def inspect_qdrant_points(
    filename: Optional[str] = None,
    limit: int = 10,
    include_vectors: bool = False
) -> list[dict]:
    """
    Inspect the exact text and metadata stored in Qdrant.

    Use this to verify what was actually saved.
    """

    if limit < 1:

        raise ValueError(
            "limit must be at least 1."
        )

    records, _ = qdrant_client.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=(

            create_source_filter(filename)

            if filename

            else None

        ),

        limit=limit,

        with_payload=True,

        with_vectors=include_vectors

    )

    inspected_records = []

    for record in records:

        payload = record.payload or {}

        item = {

            "point_id": str(record.id),

            "source": payload.get("source"),

            "chunk_id": payload.get("chunk_id"),

            "page_start": payload.get("page_start"),

            "page_end": payload.get("page_end"),

            "content_type": payload.get("content_type"),

            "logical_unit_id": payload.get("logical_unit_id"),

            "continuation_part": payload.get("continuation_part"),

            # Exact text stored in Qdrant.
            "text": payload.get("text", ""),

            "payload": payload

        }

        if include_vectors:

            vector = record.vector

            if isinstance(vector, dict):

                vector = next(
                    iter(vector.values()),
                    []
                )

            item["vector_dimension"] = len(
                vector or []
            )

        inspected_records.append(item)

    return inspected_records


# ============================================================
# 19. INSPECT ALL STORED PAGES
# ============================================================

def inspect_pdf_page_records(
    filename: str
) -> list[dict]:
    """
    Return all stored records for one PDF,
    arranged by physical page and chunk ID.

    This is useful for checking page-by-page storage.
    """

    if not filename:
        return []

    all_records = []

    offset = None

    while True:

        records, offset = qdrant_client.scroll(

            collection_name=COLLECTION_NAME,

            scroll_filter=create_source_filter(
                filename
            ),

            limit=DELETE_BATCH_SIZE,

            offset=offset,

            with_payload=True,

            with_vectors=False

        )

        for record in records:

            payload = record.payload or {}

            all_records.append({

                "point_id": str(record.id),

                "chunk_id": payload.get("chunk_id"),

                "page": payload.get("page_start"),

                "content_type": payload.get("content_type"),

                "logical_unit_id": payload.get("logical_unit_id"),

                "continuation_part": payload.get("continuation_part"),

                "text": payload.get("text", "")

            })

        if offset is None:
            break

    all_records.sort(
        key=lambda item: (

            item.get("page") or 0,

            item.get("chunk_id")
            if item.get("chunk_id") is not None
            else -1

        )
    )

    return all_records


# ============================================================
# 20. COUNT RECORDS FOR A PDF
# ============================================================

def count_pdf_points(filename: str) -> int:

    if not filename:
        return 0

    result = qdrant_client.count(

        collection_name=COLLECTION_NAME,

        count_filter=create_source_filter(filename),

        exact=True

    )

    return int(result.count)


# ============================================================
# 21. INSPECT CONTINUATION GROUPS
# ============================================================

def inspect_continuation_groups(
    filename: Optional[str] = None
) -> list[dict]:
    """
    Inspect stored chunks that share a logical_unit_id.
    """

    continuation_groups = {}

    offset = None

    while True:

        records, offset = qdrant_client.scroll(

            collection_name=COLLECTION_NAME,

            scroll_filter=(

                create_source_filter(filename)

                if filename

                else None

            ),

            limit=DELETE_BATCH_SIZE,

            offset=offset,

            with_payload=True,

            with_vectors=False

        )

        for record in records:

            payload = record.payload or {}

            logical_unit_id = payload.get(
                "logical_unit_id"
            )

            if not logical_unit_id:
                continue

            continuation_groups.setdefault(
                logical_unit_id,
                []
            ).append({

                "point_id": str(record.id),

                "source": payload.get("source"),

                "page_start": payload.get("page_start"),

                "page_end": payload.get("page_end"),

                "continuation_part": payload.get(
                    "continuation_part"
                ),

                "is_continuation": payload.get(
                    "is_continuation",
                    False
                ),

                "text": payload.get("text", "")

            })

        if offset is None:
            break

    result = []

    for logical_unit_id, chunks in continuation_groups.items():

        chunks.sort(
            key=lambda item: (

                item.get("continuation_part") or 0,

                item.get("page_start") or 0

            )
        )

        result.append({

            "logical_unit_id": logical_unit_id,

            "chunks": chunks,

            "number_of_parts": len(chunks)

        })

    return result
