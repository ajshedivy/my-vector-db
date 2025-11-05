from my_vector_db.db import MyVectorDB

def print_db_info(vector_db: MyVectorDB) -> None:
    chunk_count = vector_db.get_count()
    print(f"✓ Connected to library: {vector_db.library_name}")
    print(f"  Chunks in database: {chunk_count}")

    if chunk_count == 0:
        print("\n⚠ Warning: No data found in library!")
        print("Run scripts/load_data.py first to load Python programming content.")
        return

    print("\n" + "=" * 70)
    print("Querying Agent with Python Knowledge")
    print("=" * 70)

    # Ask questions about Python programming
    questions = [
        "How do I install Python?",
        "What are the basic data types in Python?",
        "How do I define a function?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 70)