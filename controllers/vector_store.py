import uuid
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

# 1. Load the local embedding model (384 dimensions)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 2. Connect to Qdrant running inside Docker
QDRANT_URL = "http://localhost:6335"
COLLECTION_NAME = "pdf_syllabus_qna"

qdrant_client = QdrantClient(url=QDRANT_URL)

# 3. Create collection with Cosine Similarity if it doesn't exist
def init_qdrant_collection():
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # Matches all-MiniLM-L6-v2 output size
                distance=Distance.COSINE  # Explicit Cosine Similarity
            ),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")

# Initialize collection on startup
init_qdrant_collection()

# 4. Convert text list into embeddings and store in Docker Qdrant
def store_chunks_in_qdrant(chunks: list[dict], filename: str) -> int:


    if not chunks:
        return 0

    # 1. Extract text from every chunk
    texts = [chunk["text"] for chunk in chunks]

    # 2. Generate embeddings
    vectors = embedding_model.encode(texts).tolist()

    # 3. Prepare Qdrant points
    points = []

    for chunk, vector in zip(chunks, vectors):

        payload = {
            "text": chunk["text"],
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "subject": chunk["subject"],
            "extraction_method": chunk["extraction_method"],
            "source": filename
        }

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        )

    # 4. Store points in Qdrant
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    logger.info(
        "Stored %d vectors in Qdrant for %s",
        len(points),
        filename
    )

    return len(points)




# --------------------------------------------------
# 5. SEMANTIC SEARCH / RETRIEVAL
# --------------------------------------------------

def search_similar_chunks(
    query: str,
    top_k: int = 3
) -> list[dict]:

    # STEP A: Convert the user's question into an embedding
    query_vector = embedding_model.encode(query).tolist()

    # STEP B: Search Qdrant for similar vectors
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    # STEP C: Extract text and metadata from retrieved points
    retrieved_chunks = []

    for hit in search_results:

        retrieved_chunks.append({
            "score": round(hit.score, 4),
            "chunk_id": hit.payload.get("chunk_id"),
            "page_start": hit.payload.get("page_start"),
            "page_end": hit.payload.get("page_end"),
            "subject": hit.payload.get("subject"),
            "source": hit.payload.get("source"),
            "text": hit.payload.get("text")
        })

    # STEP D: Return the retrieved chunks
    return retrieved_chunks