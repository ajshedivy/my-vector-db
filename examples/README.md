# Vector Database SDK Examples

This directory contains comprehensive examples demonstrating the Vector Database Python SDK capabilities. Each example is self-contained and focuses on specific features and workflows.

## Prerequisites

1. **API Server Running**: Ensure the Vector Database API server is running on `http://localhost:8000`

   See the [Run the Vector Database Server](../docs/README.md#run-the-vector-database-server) section for detailed setup instructions including Docker, Docker Compose, and Python options.

   Quick start:
   ```bash
   # with Docker Compose (recommended)
   git clone https://github.com/ajshedivy/my-vector-db.git
   cd my-vector-db
   docker compose up -d

   # Or with Docker 
   docker run -d --name my-vector-db -p 8000:8000 ghcr.io/ajshedivy/my-vector-db:latest
   ```

2. **SDK Installed**: Install the Vector Database SDK
   ```bash
   # Install from PyPI
   pip install my-vector-db

   # Or from source
   cd my-vector-db
   uv sync
   source .venv/bin/activate
   ```

3. **Dependencies**: All examples use the SDK and standard library. No additional dependencies required.

## Available Examples

| Example | Description |
|---------|-------------|
| [search_basic.py](search_basic.py) | Fundamental k-NN vector search operations and similarity scoring |
| [search_filters_declarative.py](search_filters_declarative.py) | Server-side metadata filtering with complex logical operators |
| [search_filters_custom.py](search_filters_custom.py) | Client-side filtering using custom Python functions |
| [search_filters_combined.py](search_filters_combined.py) | Combining server-side and client-side filters for maximum flexibility |
| [batch_operations.py](batch_operations.py) | Efficient bulk operations and performance comparison |
| [persistence_workflow.py](persistence_workflow.py) | Database snapshots, backup, and restore workflows |
| [error_handling.py](error_handling.py) | Production-ready error handling, retry logic, and patterns |
| [agno_example.py](agno_example.py) | Integration with Agno AI SDK for automated embeddings |
| [sdk_example.py](sdk_example.py) | Basic SDK usage and fundamental operations |

## Running Examples

All examples can be run directly with Python:

```bash
# Run from project root
python examples/search_basic.py

# Or with uv
uv run examples/search_basic.py
```

---

## Example Details

### Simple SDK Usage

**[sdk_example.py](sdk_example.py)** 
- Basic SDK initialization
- CRUD operations for libraries, documents, and chunks
- Simple search workflow.
  
```python
python sdk_example.py
```

**SDK Docs:** [Quick Start](../docs/README.md#quick-start) • [Client SDK Reference](../docs/README.md#client-sdk-reference)

---

### Basic Vector Search

**[search_basic.py](search_basic.py)** 
- Creating libraries and documents, adding chunks with embeddings
- Performing k-nearest neighbor vector search
- Understanding similarity scores.

```python
python search_basic.py
```

**SDK Docs:** [Search Operations](../docs/README.md#search-operations) • [Vector Indexes](../docs/README.md#vector-indexes)

<details>
<summary><b>Example output (click to expand)</b></summary>

```
======================================================================
Basic Vector Similarity Search Example
======================================================================

✓ Created library: search_demo
✓ Created document: ai_concepts
✓ Added 4 chunks with embeddings

======================================================================
Searching for vectors similar to 'machine learning'
======================================================================

Found 3 results in 0.14ms

1. Score: 0.9995
   Text: Machine learning models learn patterns from data
   Subtopic: machine_learning

2. Score: 0.9987
   Text: Deep learning uses neural networks with multiple layers
   Subtopic: deep_learning

3. Score: 0.9500
   Text: Natural language processing analyzes human language
   Subtopic: nlp

======================================================================
Searching with k=1 (most similar only)
======================================================================

Top result: Machine learning models learn patterns from data
Score: 0.9995

✓ Cleanup complete

```
</details>

---

### Vector Search with Declarative Filters

**[search_filters_declarative.py](search_filters_declarative.py)** 
- Server-side metadata filtering with operators (equals, greater than, in, contains)
- Complex filter composition with AND/OR logic, and nested filter groups.

```python
python search_filters_declarative.py
```

**SDK Docs:** [Declarative Filters](../docs/README.md#declarative-filters) • [Metadata Filters](../docs/README.md#metadata-filters) • [Filter Strategy](../docs/README.md#filter-strategy)

<details>
<summary><b>Example output (click to expand)</b></summary>

```
======================================================================
Declarative Search Filters Example
======================================================================

✓ Created library and document

✓ Added 4 papers

======================================================================
Filter 1: Papers in 'ai' category
======================================================================

- Neural networks achieve state-of-the-art results... (category: ai)
- Deep learning transforms computer vision applicati... (category: ai)

======================================================================
Filter 2: Papers with confidence > 0.9
======================================================================

- Neural networks achieve state-of-the-art results... (confidence: 0.95)
- Quantum computing solves complex optimization prob... (confidence: 0.92)

======================================================================
Filter 3: (category='ai' OR category='quantum') AND confidence > 0.85
======================================================================

- Neural networks achieve state-of-the-art results...
  Category: ai, Confidence: 0.95

- Deep learning transforms computer vision applicati...
  Category: ai, Confidence: 0.88

- Quantum computing solves complex optimization prob...
  Category: quantum, Confidence: 0.92

✓ Cleanup complete
```
</details>

---

### Vector Search with Custom Filters

**[search_filters_custom.py](search_filters_custom.py)** 
- Client-side filtering with Python functions, lambda expressions for simple filters, and complex filter functions combining metadata and text filtering.

```python
python search_filters_custom.py
```

**SDK Docs:** [Custom Filter Functions](../docs/README.md#custom-filter-functions) • [Filter Strategy](../docs/README.md#filter-strategy) • [Performance Considerations](../docs/README.md#performance-considerations)

<details>
<summary><b>Example output (click to expand)</b></summary>

```
======================================================================
Custom Filter Functions Example
======================================================================

✓ Created library and document

✓ Added 4 articles

======================================================================
Filter 1: Articles containing 'neural'
======================================================================

- Machine learning and neural networks transform AI applications
- Neural architecture search automates model design

======================================================================
Filter 2: AI articles with keywords 'learning' OR 'network'
======================================================================

- Machine learning and neural networks transform AI applications
- Deep learning enables advanced pattern recognition

======================================================================
Filter 3: Short articles (word_count < 7) with high relevance
======================================================================

- Neural architecture search automates model design
  Words: 6, Score: 0.9998

- Deep learning enables advanced pattern recognition
  Words: 6, Score: 0.9988

✓ Cleanup complete
```
</details>

---

### Vector Search with Combined Filters

**[search_filters_combined.py](search_filters_combined.py)**
 - Two-stage filtering workflow combining server-side declarative filters with client-side custom filters using SearchFiltersWithCallable.

```python
python search_filters_combined.py
```

**SDK Docs:** [Filter Composition](../docs/README.md#filter-composition) • [SearchFiltersWithCallable](../docs/README.md#searchfilterswithcallable) • [Filter Strategy](../docs/README.md#filter-strategy)

<details>
<summary><b>Example output (click to expand)</b></summary>

```
======================================================================
Combined Declarative + Custom Filters Example
======================================================================

✓ Created library and document

✓ Added 4 papers

======================================================================
Baseline: Server-side filter only (confidence > 0.9)
======================================================================

Found 2 results
- Transformer models revolutionize natural language proce...
- Convolutional neural networks excel at image classifica...

======================================================================
Combined: confidence > 0.9 (server) + contains 'neural' (client)
======================================================================

Found 1 results
- Convolutional neural networks excel at image classification
  Confidence: 0.92

======================================================================
Advanced: (year=2024 AND confidence>0.9) + keyword check
======================================================================

Found 2 results
- Transformer models revolutionize natural language processing
- Convolutional neural networks excel at image classification

✓ Cleanup complete
```
</details>

---

### Batch Operations

**[batch_operations.py](batch_operations.py)** 
- Comparing batch vs individual operations performance, using add_chunks() for bulk inserts, and demonstrating efficient data loading patterns with atomic transactions.

```python
python batch_operations.py
```

**SDK Docs:** [add_chunks](../docs/README.md#add_chunks) • [Performance Considerations](../docs/README.md#performance-considerations) • [Batch Operations](../docs/README.md#batch-operations)

<details>
<summary><b>Example output (click to expand)</b></summary>

```
======================================================================
Batch Operations Example
======================================================================

✓ Created library and document

======================================================================
Method 1: Batch add chunks using dictionaries
======================================================================

✓ Added 5 chunks in 2.22ms
  Average: 0.44ms per chunk

======================================================================
Method 2: Batch add chunks using Chunk objects
======================================================================

✓ Added 4 chunks in 2.13ms

======================================================================
Performance Comparison: Batch vs Individual Operations
======================================================================

Individual operations (10 chunks): 18.01ms
Batch operation (10 chunks):       2.25ms
Speedup: 8.0x faster

======================================================================
Large Batch: 100 chunks in single operation
======================================================================

✓ Added 100 chunks in 8.37ms
  Throughput: 11949 chunks/second

✓ Cleanup complete
```
</details>

---

### Error Handling

**[error_handling.py](error_handling.py)** 
- Production-ready error handling with specific exception handling (ValidationError, NotFoundError)
- Context manager pattern for cleanup, retry logic with exponential backoff, and graceful degradation.

**SDK Docs:** [Error Handling](../docs/README.md#error-handling) • [Connection Management](../docs/README.md#connection-management) • [Best Practices](../docs/README.md#best-practices)

---

### Agno AI Agent Integration

**[agno_example.py](agno_example.py)** 
- Integration with Agno framework for building RAG applications with automatic embedding generation, knowledge base creation, and agent-based question answering.

**Prerequisites:** Agno SDK (`pip install agno`), Cohere API key for embeddings, Anthropic API key for Claude model

**SDK Docs:** [Agno Integration](../docs/README.md#agno-integration) • [Quick Start - Agno](../docs/README.md#quick-start-1) • [Example: Full Workflow](../docs/README.md#example-full-workflow)


---

## Learning Path

**Recommended order for learning:**

1. **Start here**: `sdk_example.py` - Fundamental SDK operations
2. **Core concepts**: `search_basic.py` - Vector search basics
3. **Filtering**:
   - `search_filters_declarative.py` - Server-side filters
   - `search_filters_custom.py` - Client-side filters
   - `search_filters_combined.py` - Combined approach
4. **Performance**: `batch_operations.py` - Efficient operations
5. **Production**:
   - `error_handling.py` - Error patterns
   - `persistence_workflow.py` - Data durability
6. **Integrations**: `agno_example.py` - External services

## Additional Resources

- **[SDK Documentation](../docs/README.md)** - Complete SDK reference
- **[Main README](../README.md)** - Project overview and setup
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when server running)
