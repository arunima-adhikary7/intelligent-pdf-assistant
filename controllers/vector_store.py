
import uuid
import logging
import re
from typing import Optional

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


# ==================================================
# 1. CONFIGURATION
# ==================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

QDRANT_URL = "http://localhost:6335"
COLLECTION_NAME = "pdf_syllabus_qna"

VECTOR_SIZE = 384
BATCH_SIZE = 64

# Internal retrieval configuration.
# These are backend settings, not user settings.

INITIAL_TOP_K = 6
MAX_TOP_K = 24
RETRIEVAL_STEP = 6

# Minimum cosine similarity score.
MIN_RELEVANCE_SCORE = 0.25

# Small preference when relevance scores are similar.
FAQ_LIST_PRIORITY = 0.015


# ==================================================
# 2. LOAD MODEL AND CONNECT TO QDRANT
# ==================================================

logger.info(
    "Loading embedding model: %s",
    EMBEDDING_MODEL_NAME
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

qdrant_client = QdrantClient(
    url=QDRANT_URL
)


# ==================================================
# 3. INITIALIZE COLLECTION
# ==================================================

def init_qdrant_collection():

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
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

    else:

        logger.info(
            "Using existing Qdrant collection: %s",
            COLLECTION_NAME
        )


init_qdrant_collection()


# ==================================================
# 4. GET EXISTING POINT IDS FOR A PDF
# ==================================================

def get_pdf_point_ids(filename: str) -> list:

    point_ids = []
    offset = None

    pdf_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=filename)
            )
        ]
    )

    while True:

        records, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=pdf_filter,
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )

        point_ids.extend(
            record.id for record in records
        )

        if offset is None:
            break

    return point_ids


# ==================================================
# 5. DELETE OLD PDF POINTS
# ==================================================

def delete_points_in_batches(point_ids: list):

    if not point_ids:
        return

    for start in range(0, len(point_ids), 500):

        batch = point_ids[start:start + 500]

        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(
                points=batch
            ),
            wait=True
        )

    logger.info(
        "Deleted %s old Qdrant points",
        len(point_ids)
    )


# ==================================================
# 6. STORE PDF CHUNKS
# ==================================================

def store_chunks_in_qdrant(
    chunks: list[dict],
    filename: str
) -> int:

    if not chunks:

        logger.warning(
            "No chunks received for %s",
            filename
        )

        return 0

    # Collect old point IDs before inserting new data.
    old_point_ids = get_pdf_point_ids(filename)

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

                text = chunk.get(
                    "text",
                    ""
                ).strip()

                if text:
                    valid_items.append(
                        (chunk, text)
                    )

            if not valid_items:
                continue

            valid_chunks = [
                item[0]
                for item in valid_items
            ]

            valid_texts = [
                item[1]
                for item in valid_items
            ]

            vectors = embedding_model.encode(
                valid_texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()

            points = []

            for chunk, vector in zip(
                valid_chunks,
                vectors
            ):

                payload = {

                    "text": chunk.get("text", ""),

                    "chunk_id": chunk.get("chunk_id"),

                    "page_start": chunk.get("page_start"),

                    "page_end": chunk.get("page_end"),

                    "pages": chunk.get("pages", []),

                    "subject": chunk.get("subject"),

                    "content_type": chunk.get(
                        "content_type",
                        "text"
                    ),

                    "table_id": chunk.get("table_id"),

                    "extraction_method": chunk.get(
                        "extraction_method",
                        "unknown"
                    ),

                    "chunk_size_setting": chunk.get(
                        "chunk_size_setting"
                    ),

                    "overlap_setting": chunk.get(
                        "overlap_setting"
                    ),

                    "source": filename

                }

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload
                    )
                )

            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )

            total_stored += len(points)

        # Remove previous version only after successful
        # insertion of all new chunks.

        if total_stored > 0:

            delete_points_in_batches(
                old_point_ids
            )

        logger.info(
            "Stored %s chunks for %s",
            total_stored,
            filename
        )

        return total_stored

    except Exception:

        logger.exception(
            "Failed to store PDF chunks for %s",
            filename
        )

        raise


# ==================================================
# 7. NORMALIZE TEXT FOR DUPLICATE DETECTION
# ==================================================

def normalize_for_comparison(text: str) -> str:

    text = (text or "").lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==================================================
# 8. CONVERT QDRANT RESULTS
# ==================================================

def format_search_results(
    search_results,
    min_score: Optional[float] = None
) -> list[dict]:

    retrieved_chunks = []
    seen_texts = set()

    for hit in search_results:

        payload = hit.payload or {}

        score = float(hit.score)

        if (
            min_score is not None
            and score < min_score
        ):
            continue

        text = payload.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        normalized = normalize_for_comparison(
            text
        )

        if normalized in seen_texts:
            continue

        seen_texts.add(normalized)

        content_type = payload.get(
            "content_type",
            "text"
        )

        # Keep the original relevance score.
        # Use a separate ranking score for ordering.

        ranking_score = score

        if content_type == "faq":
            ranking_score += FAQ_LIST_PRIORITY

        elif content_type == "list":
            ranking_score += FAQ_LIST_PRIORITY

        retrieved_chunks.append({

            "score": round(score, 4),

            "ranking_score": round(
                ranking_score,
                4
            ),

            "chunk_id": payload.get("chunk_id"),

            "page_start": payload.get("page_start"),

            "page_end": payload.get("page_end"),

            "pages": payload.get("pages", []),

            "subject": payload.get("subject"),

            "content_type": content_type,

            "table_id": payload.get("table_id"),

            "extraction_method": payload.get(
                "extraction_method"
            ),

            "source": payload.get("source"),

            "text": text

        })

    # FAQ and list chunks get a small preference
    # when their relevance scores are comparable.

    retrieved_chunks.sort(
        key=lambda item: item["ranking_score"],
        reverse=True
    )

    return retrieved_chunks


# ==================================================
# 9. SEARCH QDRANT
# ==================================================

def search_qdrant(
    query_vector: list,
    limit: int,
    filename: Optional[str] = None
):

    search_filter = None

    if filename:

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filename)
                )
            ]
        )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
        with_payload=True
    )

    return response.points


# ==================================================
# 10. ADAPTIVE SEMANTIC RETRIEVAL
# ==================================================

def search_similar_chunks(
    query: str,
    top_k: Optional[int] = None,
    filename: Optional[str] = None,
    min_score: Optional[float] = None
) -> list[dict]:

    if not query or not query.strip():
        return []

    # Use backend configuration if top_k is omitted.
    initial_limit = (
        top_k
        if top_k is not None
        else INITIAL_TOP_K
    )

    initial_limit = max(
        1,
        min(initial_limit, MAX_TOP_K)
    )

    maximum_limit = max(
        initial_limit,
        MAX_TOP_K
    )

    relevance_threshold = (
        min_score
        if min_score is not None
        else MIN_RELEVANCE_SCORE
    )

    logger.info(
        "Starting adaptive retrieval. "
        "Initial limit=%s, maximum limit=%s",
        initial_limit,
        maximum_limit
    )

    # Generate query embedding only once.

    query_vector = embedding_model.encode(
        query.strip(),
        convert_to_numpy=True
    ).tolist()

    retrieved_chunks = []
    current_limit = initial_limit

    while True:

        search_results = search_qdrant(
            query_vector=query_vector,
            limit=current_limit,
            filename=filename
        )

        retrieved_chunks = format_search_results(
            search_results,
            min_score=relevance_threshold
        )

        logger.info(
            "Retrieved %s candidates; "
            "%s passed relevance filtering "
            "with search limit %s",
            len(search_results),
            len(retrieved_chunks),
            current_limit
        )

        # Stop when enough relevant results are available.
        # Otherwise, increase the search limit.

        minimum_expected = min(
            initial_limit,
            4
        )

        if len(retrieved_chunks) >= minimum_expected:
            break

        if current_limit >= maximum_limit:
            break

        current_limit = min(
            current_limit + RETRIEVAL_STEP,
            maximum_limit
        )

    # Remove internal ranking score from final output.
    # Keep the original similarity score.

    for chunk in retrieved_chunks:
        chunk.pop("ranking_score", None)

    logger.info(
        "Adaptive retrieval completed. "
        "Returning %s relevant chunks",
        len(retrieved_chunks)
    )

    return retrieved_chunks


# ==================================================
# 11. GET UNIQUE PDF FILENAMES
# ==================================================

def get_unique_pdf_files() -> list[str]:

    unique_files = set()
    next_offset = None

    while True:

        records, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=next_offset,
            with_payload=["source"],
            with_vectors=False
        )

        for record in records:

            if record.payload:

                filename = record.payload.get(
                    "source"
                )

                if filename:
                    unique_files.add(filename)

        if next_offset is None:
            break

    return sorted(unique_files)


# ==================================================
# 12. COLLECTION INFORMATION
# ==================================================

def get_collection_info() -> dict:

    info = qdrant_client.get_collection(
        collection_name=COLLECTION_NAME
    )

    return {

        "collection_name": COLLECTION_NAME,

        "total_points": info.points_count,

        "vector_size": VECTOR_SIZE,

        "distance": "Cosine"

    }
