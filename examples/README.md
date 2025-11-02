# Vector Database SDK Examples

This directory contains comprehensive examples demonstrating the Vector Database Python SDK capabilities. Each example is self-contained and focuses on specific features and workflows.

## Prerequisites

1. **API Server Running**: Ensure the Vector Database API server is running on `http://localhost:8000`
   ```bash
   # Start the server with Docker
   docker-compose up -d

   # Or run locally
   uvicorn my_vector_db.main:app --reload
   ```

2. **SDK Installed**: Install the Vector Database SDK
   ```bash
    cd my-vector-db
    uv sync
    source .venv/bin/activate
   ```

3. **Dependencies**: All examples use the SDK and standard library. No additional dependencies required.

## Available Examples

| Example | Description |
|---------|-------------|
| [search_basic.py](#search_basicpy) | Fundamental k-NN vector search operations and similarity scoring |
| [search_filters_declarative.py](#search_filters_declarativepy) | Server-side metadata filtering with complex logical operators |
| [search_filters_custom.py](#search_filters_custompy) | Client-side filtering using custom Python functions |
| [search_filters_combined.py](#search_filters_combinedpy) | Combining server-side and client-side filters for maximum flexibility |
| [batch_operations.py](#batch_operationspy) | Efficient bulk operations and performance comparison |
| [persistence_workflow.py](#persistence_workflowpy) | Database snapshots, backup, and restore workflows |
| [error_handling.py](#error_handlingpy) | Production-ready error handling, retry logic, and patterns |
| [agno_example.py](#agno_examplepy) | Integration with Agno AI SDK for automated embeddings |
| [sdk_example.py](#sdk_examplepy) | Basic SDK usage and fundamental operations |

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

```
python sdk_example.py
```

**What it demonstrates:**
- Basic SDK initialization
- CRUD operations for libraries, documents, and chunks
- Simple search workflow
- Essential SDK patterns

**What happens:**
1. Creates client with context manager
2. Creates library, document, and chunks
3. Performs basic search
4. Shows fundamental SDK operations

**Key concepts:**
- VectorDBClient initialization
- Context manager pattern
- Basic CRUD operations
- Simple search workflow

**SDK Docs:**
- [Quick Start](../docs/README.md#quick-start)
- [Client SDK Reference](../docs/README.md#client-sdk-reference)


### Basic Vector Search

```
python search_basic.py
```

**What it demonstrates:**
- Creating libraries and documents
- Adding chunks with embeddings
- Performing k-nearest neighbor (k-NN) vector search
- Understanding similarity scores
- Search result ranking

**What happens:**
1. Creates a library with FLAT index (exact search)
2. Adds 4 document chunks about different AI topics
3. Performs a search with a query embedding
4. Displays top 3 results ranked by similarity score
5. Cleans up resources

**Key concepts:**
- Vector similarity search fundamentals
- Cosine similarity scoring (higher = more similar)
- k parameter controls number of results

**SDK Docs:**
- [Search Operations](../docs/README.md#search-operations)
- [Vector Indexes](../docs/README.md#vector-indexes)

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

```
python search_filters_declarative.py
```

**What it demonstrates:**
- Server-side metadata filtering
- Filter operators (equals, greater than, in, contains)
- Complex filter composition with AND/OR logic
- Nested filter groups
- Numeric and string comparisons

**What happens:**
1. Creates research papers with rich metadata (category, confidence, author)
2. Filters by exact category match (`category == "ai"`)
3. Filters by numeric range (`confidence > 0.9`)
4. Combines filters with nested AND/OR logic
5. Shows how declarative filters are applied server-side for efficiency

**Key concepts:**
- Server-side filtering reduces network transfer
- FilterGroup with logical operators (AND, OR)
- MetadataFilter with comparison operators
- Nested filter composition

**SDK Docs:**
- [Filtering Guide - Declarative Filters](../docs/README.md#declarative-filters)
- [Metadata Filters](../docs/README.md#metadata-filters)
- [Best Practices - Filter Strategy](../docs/README.md#filter-strategy)

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

```
python search_filters_custom.py
```

**What it demonstrates:**
- Client-side filtering with Python functions
- Lambda expressions for simple filters
- Complex filter functions with business logic
- Text-based filtering on chunk content
- Combining metadata and text filtering

**What happens:**
1. Creates articles with text content and metadata
2. Uses lambda filter to find articles containing "neural"
3. Implements complex filter combining metadata and keywords
4. Demonstrates filtering by text patterns and relevance scores

**Key concepts:**
- Client-side filters applied after fetching results
- Over-fetching (k*3) to ensure enough results after filtering
- Access to SearchResult fields (text, metadata, score)
- Flexible Python logic for complex filtering

**SDK Docs:**
- [Filtering Guide - Custom Filter Functions](../docs/README.md#custom-filter-functions)
- [Best Practices - Filter Strategy](../docs/README.md#filter-strategy)
- [Performance Considerations](../docs/README.md#performance-considerations)

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

### Vecor Search with Combined Filters

```
python search_filters_combined.py
```

**What it demonstrates:**
- Two-stage filtering workflow (server + client)
- SearchFiltersWithCallable model
- Performance optimization with combined filters
- When to use each filtering approach

**What happens:**
1. Baseline: Server-side only filter (`confidence > 0.9`)
2. Combined: Server filters by confidence, then client filters by text
3. Advanced: Complex server filters + keyword checking
4. Shows how combining approaches provides efficiency + flexibility

**Key concepts:**
- Declarative filters narrow candidates server-side
- Custom filters refine based on text client-side
- Over-fetching factor (k*9) for combined filters
- Best of both worlds: efficient + flexible

**SDK Docs:**
- [Filtering Guide - Filter Composition](../docs/README.md#filter-composition)
- [SearchFiltersWithCallable](../docs/README.md#searchfilterswithcallable)
- [Best Practices - Filter Strategy](../docs/README.md#filter-strategy)

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

### batch Operations 

```
python batch_operations.py
```

**What it demonstrates:**
- Batch vs individual operations performance
- add_chunks() for bulk inserts
- Performance benchmarking
- Efficient data loading patterns

**What happens:**
1. Compares individual add_chunk() calls vs batch add_chunks()
2. Measures execution time for 100 chunks
3. Shows speedup factor
4. Demonstrates atomic batch operations

**Key concepts:**
- Batch operations use single API request
- Atomic operations (all succeed or all fail)
- Significant performance improvement for bulk data
- Best practice for data import workflows

**SDK Docs:**
- [Chunk Operations - add_chunks](../docs/README.md#add_chunks)
- [Best Practices - Performance Considerations](../docs/README.md#performance-considerations)
- [Batch Operations](../docs/README.md#batch-operations)

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


**What it demonstrates:**
- Specific exception handling (ValidationError, NotFoundError, etc.)
- Context manager pattern for cleanup
- Retry logic with exponential backoff
- Graceful degradation
- Production-ready error patterns

**What happens:**
1. Shows context manager usage (automatic cleanup)
2. Demonstrates handling specific exceptions
3. Implements retry logic for transient errors
4. Shows fallback strategies when operations fail
5. Guaranteed cleanup with try-finally

**Key concepts:**
- VectorDBError exception hierarchy
- Retry strategies for ServerConnectionError
- Don't retry ValidationError (won't succeed)
- Resource cleanup patterns
- Production error handling

**SDK Docs:**
- [Error Handling](../docs/README.md#error-handling)
- [Connection Management](../docs/README.md#connection-management)
- [Best Practices](../docs/README.md#best-practices)


---

### Agno AI Agent Integration Example

```
python agno_example.py
``` 

**What it demonstrates:**
- Integration with Agno AI SDK
- Automatic embedding generation
- Real embeddings from LLM providers
- Production workflow with external services

**What happens:**
1. Initializes Agno client with API credentials
2. Creates chunks with raw text (no manual embeddings)
3. Agno automatically generates embeddings via OpenAI/Anthropic
4. Performs search with real semantic embeddings
5. Shows integration pattern for production use

**Key concepts:**
- External embedding service integration
- Automatic embedding generation
- Production-ready patterns
- Real semantic similarity

**Prerequisites:**
- Agno SDK installed (`pip install agno`)
- API key for embedding provider (OpenAI, Anthropic, etc.)


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
