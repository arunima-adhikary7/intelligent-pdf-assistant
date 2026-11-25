
from controllers.vector_store import qdrant_client, COLLECTION_NAME

all_points = []
offset = None

while True:
    points, next_offset = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    all_points.extend(points)

    if next_offset is None:
        break

    offset = next_offset

keywords = [
    "certificate of insurance",
    "registration certificate",
    "pollution",
    "driving licence",
    "driving license",
    "public places",
    "plying"
]

pdf_chunks = [
    point.payload
    for point in all_points
    if point.payload
]

print("\nTOTAL STORED CHUNKS:", len(pdf_chunks))

matched_chunks = 0

for chunk in pdf_chunks:
    content = chunk.get("text", "")

    if any(word in content.lower() for word in keywords):
        matched_chunks += 1

        print("\n" + "=" * 70)
        print("PDF:", chunk.get("source"))
        print("PAGE:", chunk.get("page_start"), "-", chunk.get("page_end"))
        print("TYPE:", chunk.get("content_type"))
        print("CHUNK ID:", chunk.get("chunk_id"))
        print("TEXT:\n", content)

print("\nTOTAL MATCHED CHUNKS:", matched_chunks)
