from typing import List
from my_vector_db import Library, VectorDBClient

# Initialize client
client = VectorDBClient(base_url="http://localhost:8000")

libs: List[Library] = client.list_libraries()
print("Libraries in the database:")
for lib in libs:
    print(f" - {lib.name} (ID: {lib.id})")
    documents = client.list_documents(library_id=lib.id)
    print("   Documents in library:")
    for doc in documents:
        print(f"    - {doc.name} (ID: {doc.id})")
        chunks = client.list_chunks(document_id=doc.id)
        print("      Chunks in document:")
        for chunk in chunks:
            print(f"       - Chunk ID: {chunk.id}, Text: {chunk.text[:30]}...")

client.save_snapshot()
libs = client.list_libraries()
print("Libraries in the database:")
for lib in libs:
    client.delete_library(library_id=lib.id)
    print(f"Deleted library: {lib.name} (ID: {lib.id})")

libs = client.list_libraries()
print("Libraries after deletion:")
print(libs)

client.restore_snapshot()
print("Libraries after loading snapshot:")
libs = client.list_libraries()
print("Libraries in the database:")
for lib in libs:
    print(f" - {lib.name} (ID: {lib.id})")
    documents = client.list_documents(library_id=lib.id)
    print("   Documents in library:")
    for doc in documents:
        print(f"    - {doc.name} (ID: {doc.id})")
        chunks = client.list_chunks(document_id=doc.id)
        print("      Chunks in document:")
        for chunk in chunks:
            print(f"       - Chunk ID: {chunk.id}, Text: {chunk.text[:30]}...")


# Cleanup
client.close()
